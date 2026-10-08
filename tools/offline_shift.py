"""Shift a recorded input.wav offline and report the same defects we measure live.

Lets a change to the shifter be judged against real music in seconds, without
asking anyone to play a song again.

    python tools/offline_shift.py capture/song2/input.wav --semitones 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf  # noqa: E402

from autotranspose.shifter import PitchShifter  # noqa: E402

BLOCK = 1024


def shift_stereo(x: np.ndarray, sr: int, semitones: int, **kw) -> np.ndarray:
    """x is (frames, channels). Returns the same shape, shifted."""
    sh = PitchShifter(sr, x.shape[1], **kw)  # kw selects the engine
    sh.set_semitones(semitones)
    out = []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        out.append(sh.process(np.ascontiguousarray(x[i : i + BLOCK].T)).T)
    return np.concatenate(out, axis=0)


def corr(x: np.ndarray) -> float:
    if x.shape[1] < 2:
        return 1.0
    l, r = x[:, 0], x[:, 1]
    if l.std() < 1e-9 or r.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(l, r)[0, 1])


def band_db(a: np.ndarray, b: np.ndarray, sr: int) -> list[tuple[str, float]]:
    nfft, hop = 2048, 1024
    w = np.hanning(nfft)

    def spec(x):
        f = [np.abs(np.fft.rfft(x[i : i + nfft] * w)) for i in range(0, len(x) - nfft, hop)]
        return np.array(f)

    A, B = spec(a.mean(axis=1)), spec(b.mean(axis=1))
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    out = []
    for lo, hi, label in [
        (0, 200, "bass <200"),
        (200, 2000, "mid 200-2k"),
        (2000, 8000, "high 2k-8k"),
        (8000, sr / 2, "top >8k"),
    ]:
        m = (freqs >= lo) & (freqs < hi)
        ea, eb = A[:, m].mean(), B[:, m].mean()
        out.append((label, 20 * np.log10((eb + 1e-12) / (ea + 1e-12))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--semitones", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--start", type=float, default=30.0)
    args = ap.parse_args()

    x, sr = sf.read(args.wav, dtype="float32", always_2d=True,
                    start=int(args.start * 48000), frames=int(args.seconds * 48000))
    print("input %.0fs @ %d Hz, %d ch   shift %+d\n" % (len(x) / sr, sr, x.shape[1], args.semitones))

    configs = [
        ("numpy vocoder (fallback)", dict(engine="vocoder")),
        ("Signalsmith Stretch (default)", dict(engine="signalsmith")),
    ]

    print("%-40s %10s %9s %9s %9s %9s"
          % ("config", "L/R corr", "rms dB", "bass dB", "mid dB", "top dB"))
    print("-" * 92)
    print("%-40s %10.3f %9s %9s %9s %9s" % ("input (reference)", corr(x), "-", "-", "-", "-"))
    for name, kw in configs:
        try:
            y = shift_stereo(x, sr, args.semitones, **_filter_kw(kw))
        except (TypeError, RuntimeError) as exc:
            print("%-40s  (unavailable: %s)" % (name, exc))
            continue
        n = min(len(x), len(y))
        rms_db = 20 * np.log10(
            (np.sqrt((y[:n].mean(axis=1) ** 2).mean()) + 1e-12)
            / (np.sqrt((x[:n].mean(axis=1) ** 2).mean()) + 1e-12)
        )
        bands = dict(band_db(x[:n], y[:n], sr))
        print(
            "%-40s %10.3f %8.1f %8.1f %8.1f %8.1f"
            % (name, corr(y), rms_db, bands["bass <200"], bands["mid 200-2k"], bands["top >8k"])
        )
    print("-" * 92)
    return 0


def _filter_kw(kw: dict) -> dict:
    """Only pass options the current PitchShifter accepts.

    Silently dropping unknown keys once made two rows of this table identical
    while labelled as different configurations, so anything dropped is reported.
    """
    import inspect

    allowed = set(inspect.signature(PitchShifter.__init__).parameters)
    dropped = sorted(set(kw) - allowed)
    if dropped:
        raise TypeError("PitchShifter does not accept %s" % ", ".join(dropped))
    return kw


if __name__ == "__main__":
    sys.exit(main())
