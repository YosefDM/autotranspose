"""Diagnose a real recorded output.wav against its input.wav.

Synthetic metrics turned out to be a dead end: on an "ideal shifted waveform"
SNR test our shifter scored 5.2 dB and Rubber Band scored 0.21 dB, which only
proves the metric does not measure what ears care about. So this looks for
defects in the actual audio instead, and reports each one separately rather than
collapsing them into a single score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf  # noqa: E402


def band_energy(x: np.ndarray, sr: int, nfft: int = 2048, hop: int = 1024):
    w = np.hanning(nfft)
    frames = []
    for i in range(0, len(x) - nfft, hop):
        frames.append(np.abs(np.fft.rfft(x[i : i + nfft] * w)))
    S = np.array(frames)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    return S, freqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--trim-end", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=60.0, help="analyse this much")
    args = ap.parse_args()

    d = Path(args.directory)
    a, sr = sf.read(str(d / "input.wav"), dtype="float32", always_2d=True)
    b, _ = sf.read(str(d / "output.wav"), dtype="float32", always_2d=True)
    n = min(len(a), len(b))
    if args.trim_end:
        n -= int(args.trim_end * sr)
    # Skip the first 20 s: the shift is still settling there.
    start = min(int(20 * sr), n // 4)
    lim = min(n, start + int(args.seconds * sr))
    a, b = a[start:lim], b[start:lim]
    print("analysing %.0fs from %.0fs in, %d Hz\n" % ((lim - start) / sr, start / sr, sr))

    am, bm = a.mean(axis=1), b.mean(axis=1)

    print("== levels ==")
    print("  input  peak %.4f  rms %.5f" % (np.abs(a).max(), np.sqrt((am**2).mean())))
    print("  output peak %.4f  rms %.5f" % (np.abs(b).max(), np.sqrt((bm**2).mean())))
    print("  clipped output samples: %d" % int((np.abs(b) >= 0.999).sum()))

    print("\n== dropouts ==")
    win = sr // 100
    nw = len(am) // win
    aw = np.sqrt((am[: nw * win].reshape(nw, win) ** 2).mean(axis=1))
    bw = np.sqrt((bm[: nw * win].reshape(nw, win) ** 2).mean(axis=1))
    loud = aw > aw.max() * 0.02
    holes = int(np.sum(loud & (bw < 1e-4)))
    print("  10 ms windows loud in, silent out: %d (%.2f%% of the song)"
          % (holes, holes / max(nw, 1) * 100))
    # Sudden level collapses that are not in the input
    ratio = bw / np.maximum(aw, 1e-9)
    med = np.median(ratio[loud]) if loud.any() else 1.0
    dips = int(np.sum(loud & (ratio < med * 0.3)))
    print("  windows where output level collapsed vs input: %d (%.2f%%)"
          % (dips, dips / max(nw, 1) * 100))

    print("\n== stereo image ==")
    if a.shape[1] == 2:
        def corr(x):
            l, r = x[:, 0], x[:, 1]
            if l.std() < 1e-9 or r.std() < 1e-9:
                return 1.0
            return float(np.corrcoef(l, r)[0, 1])
        ca, cb = corr(a), corr(b)
        print("  L/R correlation  input %.3f -> output %.3f" % (ca, cb))
        if cb < ca - 0.15:
            print("  -> the image is coming apart: the two channels are shifted")
            print("     independently, so they drift out of phase with each other.")
            print("     Fix: process mid/side, or run --mono.")
        else:
            print("  -> image preserved")

    print("\n== spectrum ==")
    SA, freqs = band_energy(am, sr)
    SB, _ = band_energy(bm, sr)
    nn = min(len(SA), len(SB))
    SA, SB = SA[:nn], SB[:nn]
    for lo, hi, label in [(0, 200, "bass <200Hz"), (200, 2000, "mid 200-2k"),
                          (2000, 8000, "high 2k-8k"), (8000, sr / 2, "top >8k")]:
        m = (freqs >= lo) & (freqs < hi)
        ea, eb = SA[:, m].mean(), SB[:, m].mean()
        print("  %-14s in %.5f  out %.5f  %+.1f dB"
              % (label, ea, eb, 20 * np.log10((eb + 1e-12) / (ea + 1e-12))))

    print("\n== temporal smearing ==")
    def crest(x, k=256):
        env = np.convolve(np.abs(x), np.ones(k) / k, mode="same")
        return float(np.percentile(np.abs(x), 99.9) / max(env.mean(), 1e-12))
    print("  crest (99.9th pct / mean envelope): in %.1f  out %.1f  (%.0f%% kept)"
          % (crest(am), crest(bm), crest(bm) / max(crest(am), 1e-12) * 100))

    print("\n== modulation / warble ==")
    # Slow amplitude modulation added by the shifter shows up as extra energy in
    # the 2-20 Hz band of the envelope, which is heard as warble or flutter.
    for name, sig in (("input", am), ("output", bm)):
        env = np.abs(sig)
        k = 64
        env = np.convolve(env, np.ones(k) / k, mode="same")[::16]
        esr = sr / 16
        env = env - env.mean()
        sp = np.abs(np.fft.rfft(env * np.hanning(len(env))))
        f = np.fft.rfftfreq(len(env), 1 / esr)
        band = (f > 2) & (f < 20)
        tot = (f > 0.1) & (f < 100)
        print("  %-7s envelope energy 2-20 Hz: %.1f%% of 0.1-100 Hz"
              % (name, sp[band].sum() / max(sp[tot].sum(), 1e-12) * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
