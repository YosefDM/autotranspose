"""Measure pitch-shift quality objectively, so "it sounds bad" becomes a number.

Three tests, each isolating a different artefact:

1. **Harmonic fidelity.** A sum of harmonics shifted by r should be the same
   waveform with every partial multiplied by r, and that ideal can be
   synthesised exactly. The error against it is a true SNR.
2. **Inharmonic noise.** Phase-vocoder artefacts put energy between the
   harmonics. Measuring harmonic energy against the rest gives a direct
   "phasiness/metallic" score.
3. **Transient smearing.** A click train should stay sharp; a vocoder spreads
   each click over its window. Measured as the loss of peak-to-average ratio.

Plus a reference comparison on real music: Rubber Band (via pedalboard, which
works fine one-shot even though it cannot stream) processes the same audio, and
the log-spectral distance between the two says how far off a good implementation
we are.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 48000
BLOCK = 1024


def through_shifter(mono: np.ndarray, semitones: float, **core_kw) -> np.ndarray:
    """Stream mono audio through our engine in real-time-sized blocks."""
    from autotranspose.shifter import _PhaseVocoderCore

    core = _PhaseVocoderCore(1, core_kw.pop("n_fft", 2048), core_kw.pop("overlap", 4),
                             2 ** (semitones / 12.0), **core_kw)
    out = []
    for i in range(0, len(mono) - BLOCK + 1, BLOCK):
        out.append(core.process(mono[np.newaxis, i : i + BLOCK])[0])
    return np.concatenate(out)


def rubberband(mono: np.ndarray, semitones: float) -> np.ndarray:
    from pedalboard import PitchShift

    y = PitchShift(semitones=semitones)(np.stack([mono, mono]), SR)
    return y[0]


def _best_lag(a: np.ndarray, b: np.ndarray) -> int:
    """Lag of b relative to a, via FFT cross-correlation.

    np.correlate(..., "full") is O(n^2) and takes minutes on a few seconds of
    audio; this is O(n log n).
    """
    from scipy.signal import correlate

    xc = correlate(b - b.mean(), a - a.mean(), mode="full", method="fft")
    return int(np.argmax(np.abs(xc))) - (len(a) - 1)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align b to a by cross-correlating envelopes, then trim to equal length."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    k = 2048
    kernel = np.ones(k) / k
    from scipy.signal import fftconvolve

    ea = fftconvolve(np.abs(a), kernel, mode="same")
    eb = fftconvolve(np.abs(b), kernel, mode="same")
    # Envelopes are smooth, so decimating before correlating costs nothing and
    # keeps this cheap.
    d = 16
    lag = _best_lag(ea[::d], eb[::d]) * d
    if lag > 0:
        b = b[lag:]
    elif lag < 0:
        a = a[-lag:]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def harmonic_tone(f0: float, n_harm: int, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        x += np.sin(2 * np.pi * f0 * h * t + 0.3 * h) / h
    return (x / np.abs(x).max() * 0.5).astype(np.float32)


def test_harmonic_fidelity(semitones: float, **kw) -> dict:
    """SNR against the exactly-known ideal shifted waveform."""
    f0, n_harm = 196.0, 12  # G3, rich
    x = harmonic_tone(f0, n_harm, 4.0)
    ideal = harmonic_tone(f0 * 2 ** (semitones / 12.0), n_harm, 4.0)
    got = through_shifter(x, semitones, **kw)

    # Compare steady-state only; alignment is by envelope, which is flat here, so
    # align on the waveform itself via the best circular lag.
    a, b = ideal[SR : SR * 3], got[SR : SR * 3]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lag = _best_lag(a, b)
    if lag > 0:
        b2 = np.concatenate([b[lag:], np.zeros(lag)])
    else:
        b2 = np.concatenate([np.zeros(-lag), b[: n + lag]])
    scale = float(np.dot(a, b2) / max(np.dot(b2, b2), 1e-12))
    err = a - scale * b2
    snr = 10 * np.log10(np.sum(a**2) / max(np.sum(err**2), 1e-20))
    return {"snr_db": float(snr)}


def test_inharmonic_noise(semitones: float, **kw) -> dict:
    """Energy on the expected harmonics vs everything else: phasiness score."""
    f0 = 196.0
    x = harmonic_tone(f0, 12, 4.0)
    got = through_shifter(x, semitones, **kw)
    seg = got[SR : SR * 3]
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    f0_out = f0 * 2 ** (semitones / 12.0)

    harmonic = np.zeros(len(spec), dtype=bool)
    for h in range(1, 40):
        f = f0_out * h
        if f > SR / 2.2:
            break
        harmonic |= np.abs(freqs - f) < 12.0  # +/-12 Hz around each partial
    total = spec.sum()
    hr = spec[harmonic].sum() / max(total, 1e-20)
    return {"harmonic_fraction": float(hr), "inharmonic_db": float(10 * np.log10(max(1 - hr, 1e-12)))}


def test_transients(semitones: float, **kw) -> dict:
    """Peak-to-average of a click train, before and after. Lower = more smeared."""
    n = SR * 3
    x = np.zeros(n, dtype=np.float32)
    for i in range(10, n - 10, SR // 4):  # a click every 250 ms
        x[i] = 0.9
        x[i + 1] = -0.6
    got = through_shifter(x, semitones, **kw)

    def crest(sig):
        k = 256
        env = np.convolve(np.abs(sig), np.ones(k) / k, mode="same")
        return float(np.abs(sig).max() / max(env.mean(), 1e-12))

    return {"crest_in": crest(x), "crest_out": crest(got),
            "crest_ratio": crest(got) / max(crest(x), 1e-12)}


def test_vs_rubberband(wav: Path, semitones: float, seconds: float = 20.0, **kw) -> dict:
    """Log-spectral distance to Rubber Band on real music. Lower is closer."""
    import soundfile as sf

    y, sr = sf.read(str(wav), dtype="float32", always_2d=True, frames=int(seconds * SR))
    mono = y.mean(axis=1)
    if sr != SR:
        import soxr

        mono = soxr.resample(mono, sr, SR)
    ours = through_shifter(mono, semitones, **kw)
    ref = rubberband(mono, semitones)
    a, b = _align(ref, ours)

    nfft, hop = 2048, 512
    def logspec(sig):
        frames = []
        w = np.hanning(nfft)
        for i in range(0, len(sig) - nfft, hop):
            frames.append(np.abs(np.fft.rfft(sig[i : i + nfft] * w)))
        S = np.array(frames)
        return 20 * np.log10(S + 1e-6)

    A, B = logspec(a), logspec(b)
    n = min(len(A), len(B))
    d = A[:n] - B[:n]
    return {"log_spectral_distance_db": float(np.sqrt((d**2).mean()))}


def bench(label: str, wav: Path | None, semitones: float, **kw) -> None:
    hf = test_harmonic_fidelity(semitones, **kw)
    ih = test_inharmonic_noise(semitones, **kw)
    tr = test_transients(semitones, **kw)
    line = (
        "%-28s snr %6.2f dB   harmonic %5.1f%%   inharmonic %6.1f dB   crest %4.0f%%"
        % (label, hf["snr_db"], ih["harmonic_fraction"] * 100,
           ih["inharmonic_db"], tr["crest_ratio"] * 100)
    )
    if wav is not None:
        rb = test_vs_rubberband(wav, semitones, **kw)
        line += "   vs RubberBand %5.2f dB" % rb["log_spectral_distance_db"]
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None, help="real music for the Rubber Band comparison")
    ap.add_argument("--semitones", type=float, default=4.0)
    args = ap.parse_args()
    wav = Path(args.wav) if args.wav else None

    print("Pitch-shift quality at %+g semitones" % args.semitones)
    print("  snr: vs the exact ideal shifted tone (higher better)")
    print("  harmonic%%: share of energy on the true partials (higher better)")
    print("  inharmonic: everything else, in dB (lower/more negative better)")
    print("  crest: transient sharpness kept, %% of the original (higher better)")
    if wav:
        print("  vs RubberBand: log-spectral distance (lower better)")
    print()
    for n_fft in (1024, 2048, 4096):
        for overlap in (4, 8):
            bench("n_fft %d overlap %d" % (n_fft, overlap), wav, args.semitones,
                  n_fft=n_fft, overlap=overlap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
