"""Check the white-key/gain method against the 24 synthetic keys.

The real song is ambiguous by nature, so it cannot tell us whether the new method
is *correct* -- only whether it is stable. The synthetic set has one unambiguous
right answer per key, so it is the test that catches a method that is merely
decisive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synth import synth_song  # noqa: E402

from autotranspose.keydetect import ANALYSIS_SR, KeyEstimator  # noqa: E402
from autotranspose.keyprofiles import (  # noqa: E402
    PITCH_CLASS_NAMES,
    shift_class_for_key,
    signed_shift,
)
from tools.experiment import CYCLE, chroma_frames  # noqa: E402
from tools.gain import GainDecider  # noqa: E402
from tools.whitekey import white_mass_per_shift  # noqa: E402


def run_song(mono, *, window=20.0, min_gain=0.03, min_heard=10.0):
    blocks, silents = chroma_frames(mono, n_octaves=6)
    est = KeyEstimator(window_seconds=window, profile="blend")
    dec = GainDecider(min_gain=min_gain, min_heard=min_heard)
    applied, t, changes = 0, 0.0, []
    for k, block in enumerate(blocks):
        t += CYCLE
        est.add_frames(block)
        white = None
        if not silents[k] and len(est._frames) >= 8:
            frames = np.stack(est._frames, axis=1)
            w = np.asarray(est._weights, dtype=np.float64)
            if w.sum() > 1e-9:
                white = white_mass_per_shift((frames * w).sum(axis=1) / w.sum())
        new = dec.update(white, est.filled_seconds, t)
        if new != applied:
            changes.append((t, applied, new))
            applied = new
    return applied, changes


def main() -> int:
    print("White-key/gain method over the 24 synthetic keys (40 s each):")
    hits, rows = 0, []
    for minor in (False, True):
        for pc in range(12):
            y = synth_song(
                pc, minor_key=minor, duration=40.0, sr=ANALYSIS_SR, seed=pc + 31 * minor
            )
            got, changes = run_song(y)
            want = signed_shift(shift_class_for_key(pc + (12 if minor else 0)))
            ok = got == want
            hits += ok
            label = PITCH_CLASS_NAMES[pc] + (" minor" if minor else " major")
            rows.append((label, want, got, len(changes), ok))
            print(
                "  %s %-9s want %+2d  got %+2d  changes %d"
                % ("ok  " if ok else "MISS", label, want, got, len(changes))
            )
    print("\n  correct: %d/24 (%.0f%%)" % (hits, hits / 24 * 100))
    extra = sum(r[3] for r in rows) - 24
    print("  total changes beyond the first lock: %d (ideally 0)" % max(0, extra))
    return 0 if hits >= 23 else 1


if __name__ == "__main__":
    sys.exit(main())
