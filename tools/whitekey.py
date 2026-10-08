"""Test a different objective: maximise the energy that lands on white keys.

The template approach asks "what key is this?" and then derives a shift. On the
test song its two best answers (C minor and F minor) sit 0.063 apart, so it
cannot separate them and oscillates.

But naming the key is not the goal. The goal is that the notes land under the
hands, on the naturals. That can be measured directly: for each of the 12
candidate shifts, how much of the song's pitch-class energy would fall on
{C D E F G A B} after shifting? Pick the best. No key profiles, no major/minor
decision, and the measurement rests on seven pitch classes instead of hinging on
the one or two scale degrees that separate neighbouring keys.

It also gives the honest metric for this whole problem: the fraction of the
song's energy the player actually finds on white keys.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotranspose.decide import ShiftDecider  # noqa: E402
from autotranspose.keydetect import ANALYSIS_SR, KeyEstimate, KeyEstimator  # noqa: E402
from autotranspose.keyprofiles import PITCH_CLASS_NAMES, key_label, signed_shift  # noqa: E402
from tools.experiment import CYCLE, Variant, chroma_frames, simulate  # noqa: E402
from tools.replay import load_mono  # noqa: E402

WHITE = np.array([0, 2, 4, 5, 7, 9, 11])
WHITE_MASK = np.zeros(12, dtype=bool)
WHITE_MASK[WHITE] = True


def white_mass_per_shift(chroma_mean: np.ndarray) -> np.ndarray:
    """For each shift class s, the share of energy landing on white keys."""
    total = chroma_mean.sum()
    if total <= 1e-12:
        return np.zeros(12)
    out = np.empty(12)
    for s in range(12):
        # Shifting the audio up by s sends pitch class p to (p + s) % 12.
        landed = (np.arange(12) + s) % 12
        out[s] = chroma_mean[WHITE_MASK[landed]].sum() / total
    return out


def _z(v: np.ndarray) -> np.ndarray:
    """Put a score vector on the same scale the decider's thresholds expect.

    Raw white-key fractions differ by well under a percent between neighbouring
    shifts, far below the decider's margin threshold, so it never committed at
    all. Z-scoring makes the spread comparable to the correlations the thresholds
    were designed around.
    """
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / max(float(v.std()), 1e-12)


class WhiteKeyEstimator(KeyEstimator):
    """Same rolling chroma, different question asked of it."""

    hybrid: float = 0.0  # 0 = white-key only, 1 = template only

    def estimate(self):
        base = super().estimate()
        if base is None:
            return None
        frames = np.stack(self._frames, axis=1)
        weights = np.asarray(self._weights, dtype=np.float64)
        total = weights.sum()
        if total <= 1e-9:
            return None
        mean = (frames * weights).sum(axis=1) / total

        white_raw = white_mass_per_shift(mean)
        scores = _z(white_raw) * 0.1  # 0.1 puts it in the same range as correlations
        if self.hybrid > 0:
            scores = (1.0 - self.hybrid) * scores + self.hybrid * base.shift_scores
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        margin = float(scores[order[0]] - scores[order[1]])
        # Report the key name the template method would give, for display only.
        return KeyEstimate(
            shift_class=best,
            shift_scores=scores,
            best_key=base.best_key,
            key_confidence=float(white_raw[best]),
            margin=margin,
            frames=base.frames,
            novelty=base.novelty,
        )


def achieved_white_fraction(blocks, silents, changes, duration) -> float:
    """The metric that matters: how much energy actually landed on white keys."""
    shifts = []
    cur = 0
    stamps = [(0.0, 0)] + [(c[0], c[2]) for c in changes]
    t = 0.0
    idx = 0
    total_mass = 0.0
    white_mass = 0.0
    for k, block in enumerate(blocks):
        t += CYCLE
        while idx + 1 < len(stamps) and stamps[idx + 1][0] <= t:
            idx += 1
        cur = stamps[idx][1]
        if silents[k] or block.size == 0:
            continue
        frame_sum = block.sum(axis=1)
        total = frame_sum.sum()
        if total <= 1e-12:
            continue
        landed = (np.arange(12) + (cur % 12)) % 12
        white_mass += frame_sum[WHITE_MASK[landed]].sum()
        total_mass += total
    return white_mass / total_mass if total_mass else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--truth", type=int, default=None)
    args = ap.parse_args()

    mono = load_mono(Path(args.wav))
    print("song: %.0fs" % (len(mono) / ANALYSIS_SR))
    t0 = time.perf_counter()
    blocks, silents = chroma_frames(mono, n_octaves=6)
    print("chroma: %d cycles in %.1fs\n" % (len(blocks), time.perf_counter() - t0))

    # Whole-song white-key mass for every shift: the ceiling any method could hit.
    acc = np.zeros(12)
    tot = 0.0
    all_mean = np.zeros(12)
    for b, sil in zip(blocks, silents):
        if sil or b.size == 0:
            continue
        all_mean += b.sum(axis=1)
    wm = white_mass_per_shift(all_mean)
    order = np.argsort(wm)[::-1]
    print("whole-song white-key mass per shift (the ceiling):")
    for c in order[:6]:
        print(
            "  shift %+2d  %5.1f%% of energy on white keys   (%s major / %s minor)"
            % (
                signed_shift(int(c)),
                wm[c] * 100,
                PITCH_CLASS_NAMES[(-c) % 12],
                PITCH_CLASS_NAMES[(9 - c) % 12],
            )
        )
    best_possible = signed_shift(int(order[0]))
    print("\n  best possible fixed shift: %+d (%.1f%%)" % (best_possible, wm[order[0]] * 100))
    print("  doing nothing (shift 0):   %.1f%%" % (wm[0] * 100))

    duration = len(blocks) * CYCLE
    print("\n%-34s %8s %10s %9s" % ("method", "changes", "white-key%", "final"))
    print("-" * 66)

    TUNED = dict(decay_halflife=30.0, commit_share=0.6)
    cases = [
        ("template, current settings", KeyEstimator, 20.0, {}, 0.0),
        ("template, tuned (h30 s0.6)", KeyEstimator, 20.0, TUNED, 0.0),
        ("white-key mass", WhiteKeyEstimator, 20.0, {}, 0.0),
        ("white-key mass, tuned", WhiteKeyEstimator, 20.0, TUNED, 0.0),
        ("white-key mass, 45s + tuned", WhiteKeyEstimator, 45.0, TUNED, 0.0),
        ("hybrid 25% template", WhiteKeyEstimator, 20.0, TUNED, 0.25),
        ("hybrid 50% template", WhiteKeyEstimator, 20.0, TUNED, 0.5),
        ("hybrid 50%, 45s window", WhiteKeyEstimator, 45.0, TUNED, 0.5),
        ("hybrid 75% template", WhiteKeyEstimator, 20.0, TUNED, 0.75),
    ]
    for name, cls, window, dec_kw, hybrid in cases:
        res = _simulate(blocks, silents, cls, window, dec_kw, hybrid)
        frac = achieved_white_fraction(blocks, silents, res["changes"], duration)
        print(
            "%-34s %8d %9.1f%% %9s"
            % (name, len(res["changes"]), frac * 100, "%+d" % res["final"])
        )
        for ct, o, ns, k in res["changes"]:
            print("      %5.0fs  %+d -> %+d" % (ct, o, ns))
    print("-" * 66)
    print("ceiling with a perfect fixed choice: %.1f%%" % (wm[order[0]] * 100))
    return 0


def _simulate(blocks, silents, cls, window, dec_kw, hybrid=0.0):
    est = cls(window_seconds=window, profile="blend")
    if hasattr(est, "hybrid"):
        est.hybrid = hybrid
    dec = ShiftDecider(cycle_seconds=CYCLE, **dec_kw)
    applied, t = 0, 0.0
    changes = []
    for k, block in enumerate(blocks):
        t += CYCLE
        est.add_frames(block)
        e = est.estimate() if not silents[k] else None
        new = dec.update(e, heard_seconds=est.filled_seconds, silent=silents[k], now=t)
        if new != applied:
            changes.append((t, applied, new, dec.state.key_name))
            applied = new
    return {"changes": changes, "final": applied}


if __name__ == "__main__":
    sys.exit(main())
