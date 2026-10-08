"""The fix: switch only when it measurably improves what you can play.

Every method so far decided by comparing the top candidate against the
runner-up. On the test song the two best shifts are -3 (74.3% of energy on white
keys) and +4 (73.7%) -- a 0.6-point difference. That comparison is therefore a
coin toss, which produced either oscillation (template: 3 changes) or paralysis
(white-key mass: never committed, leaving 54%).

The question that matters is not "which candidate wins?" but "is the best
candidate better than what I am doing right now?" Against the current shift the
numbers are decisive instead of marginal:

    doing nothing -> -3   is +20.2 points   obviously worth it
    -3            -> +4   is  -0.6 points   obviously not

So hysteresis stops being a tuned threshold and becomes a consequence of the
objective: once on a good shift, nothing beats it by enough to be worth a change,
and the oscillation cannot happen. A real key change mid-song still moves it,
because that genuinely costs many points.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotranspose.keydetect import ANALYSIS_SR, KeyEstimator  # noqa: E402
from autotranspose.keyprofiles import PITCH_CLASS_NAMES, signed_shift  # noqa: E402
from tools.experiment import CYCLE, chroma_frames  # noqa: E402
from tools.replay import load_mono  # noqa: E402
from tools.whitekey import WHITE_MASK, achieved_white_fraction, white_mass_per_shift  # noqa: E402


class GainDecider:
    """Switch when the best shift beats the current one by `min_gain`.

    `min_gain` is in the same units as the objective: share of pitch-class energy
    landing on white keys. 0.03 means "worth changing only if it puts at least
    three more percent of the music under the player's hands".
    """

    def __init__(
        self,
        *,
        min_gain: float = 0.03,
        min_heard: float = 10.0,
        stable_cycles: int = 3,
        min_dwell: float = 10.0,
        max_shift: int = 6,
    ):
        self.min_gain = min_gain
        self.min_heard = min_heard
        self.stable_cycles = stable_cycles
        self.min_dwell = min_dwell
        self.max_shift = max_shift
        self.applied_class = 0
        self._candidate: int | None = None
        self._streak = 0
        self._applied_at: float | None = None

    def update(self, white: np.ndarray | None, heard: float, t: float) -> int:
        if white is None:
            return signed_shift(self.applied_class)
        best = int(np.argmax(white))
        gain = float(white[best] - white[self.applied_class])

        if best == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = best, 1

        dwell_ok = self._applied_at is None or (t - self._applied_at) >= self.min_dwell
        if (
            best != self.applied_class
            and gain >= self.min_gain
            and heard >= self.min_heard
            and self._streak >= self.stable_cycles
            and dwell_ok
            and abs(signed_shift(best)) <= self.max_shift
        ):
            self.applied_class = best
            self._applied_at = t
        return signed_shift(self.applied_class)


def run(blocks, silents, *, window: float, min_gain: float, min_heard: float = 10.0):
    est = KeyEstimator(window_seconds=window, profile="blend")
    dec = GainDecider(min_gain=min_gain, min_heard=min_heard)
    applied, t = 0, 0.0
    changes = []
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
    return {"changes": changes, "final": applied}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    args = ap.parse_args()

    mono = load_mono(Path(args.wav))
    t0 = time.perf_counter()
    blocks, silents = chroma_frames(mono, n_octaves=6)
    print("song %.0fs, chroma in %.1fs" % (len(mono) / ANALYSIS_SR, time.perf_counter() - t0))

    total = np.zeros(12)
    for b, sil in zip(blocks, silents):
        if not sil and b.size:
            total += b.sum(axis=1)
    wm = white_mass_per_shift(total)
    order = np.argsort(wm)[::-1]
    ceiling = wm[order[0]]
    print(
        "\nceiling %+d = %.1f%% white; runner-up %+d = %.1f%% (gap %.1f pts); "
        "doing nothing = %.1f%%"
        % (
            signed_shift(int(order[0])), ceiling * 100,
            signed_shift(int(order[1])), wm[order[1]] * 100,
            (ceiling - wm[order[1]]) * 100, wm[0] * 100,
        )
    )

    duration = len(blocks) * CYCLE
    print("\n%-38s %8s %11s" % ("gain-based decider", "changes", "white-key%"))
    print("-" * 60)
    for window in (20.0, 45.0):
        for min_gain in (0.01, 0.02, 0.03, 0.05):
            res = run(blocks, silents, window=window, min_gain=min_gain)
            frac = achieved_white_fraction(blocks, silents, res["changes"], duration)
            label = "window %gs, min_gain %.0f pts" % (window, min_gain * 100)
            print("%-38s %8d %10.1f%%" % (label, len(res["changes"]), frac * 100))
            for ct, o, n in res["changes"]:
                print("        %5.0fs  %+d -> %+d" % (ct, o, n))
    print("-" * 60)
    print("ceiling %.1f%%   doing nothing %.1f%%" % (ceiling * 100, wm[0] * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
