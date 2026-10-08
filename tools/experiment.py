"""Try detector variants against a recorded song, many per second.

Fidelity first: an earlier version of this reimplemented the estimator and
decider in simplified form and reported 1 shift change where the real pipeline
produced 3 -- useless for tuning. So this uses the **real** `KeyEstimator` and
`ShiftDecider`; only the expensive, knob-independent part (the chroma frames) is
precomputed, once per register. The baseline is asserted against the known real
behaviour before any variant is believed.

    python tools/experiment.py capture/song1/input.wav --truth -3
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotranspose.decide import ShiftDecider  # noqa: E402
from autotranspose.keydetect import ANALYSIS_SR, IncrementalChroma, KeyEstimator  # noqa: E402
from tools.replay import load_mono, whole_song_reference  # noqa: E402

CYCLE = 1.5
SILENT_RMS = 10 ** (-58.0 / 20.0)


def chroma_frames(mono: np.ndarray, *, n_octaves: int = 6, fmin_note: str = "C2"):
    """Stream the song through the real IncrementalChroma, cycle by cycle.

    Returns (list of per-cycle frame blocks, list of per-cycle silent flags).
    """
    inc = IncrementalChroma(sr=ANALYSIS_SR, fmin_note=fmin_note, n_octaves=n_octaves)
    step = int(CYCLE * ANALYSIS_SR)
    blocks, silents = [], []
    for i in range(0, len(mono) - step + 1, step):
        chunk = mono[i : i + step]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        silents.append(rms < SILENT_RMS)
        blocks.append(inc.push(chunk))
    return blocks, silents


@dataclass
class Variant:
    name: str
    window: float = 20.0
    profile: str = "blend"
    bass_weight: float = 0.0
    decider: dict = field(default_factory=dict)


def simulate(full_blocks, bass_blocks, silents, v: Variant) -> dict:
    est = KeyEstimator(window_seconds=v.window, profile=v.profile)
    dec = ShiftDecider(cycle_seconds=CYCLE, **v.decider)

    applied, t = 0, 0.0
    changes = []
    for k, block in enumerate(full_blocks):
        t += CYCLE
        if v.bass_weight > 0 and bass_blocks is not None:
            b = bass_blocks[k]
            if block.size and b.size:
                n = min(block.shape[1], b.shape[1])
                block = (1.0 - v.bass_weight) * block[:, :n] + v.bass_weight * b[:, :n]
        est.add_frames(block)
        estimate = est.estimate() if not silents[k] else None
        new = dec.update(
            estimate, heard_seconds=est.filled_seconds, silent=silents[k], now=t
        )
        if new != applied:
            changes.append((t, applied, new, dec.state.key_name))
            applied = new

    duration = len(full_blocks) * CYCLE
    held: dict[int, float] = {}
    prev_t, prev_s = 0.0, 0
    for ct, _o, ns, _k in changes:
        held[prev_s] = held.get(prev_s, 0.0) + (ct - prev_t)
        prev_t, prev_s = ct, ns
    held[prev_s] = held.get(prev_s, 0.0) + (duration - prev_t)
    return {"changes": changes, "final": applied, "held": held, "duration": duration}


def score(res: dict, truth: int) -> dict:
    duration = res["duration"]
    stamps = [(0.0, 0)] + [(c[0], c[2]) for c in res["changes"]]
    settled = duration
    for i, (ct, sh) in enumerate(stamps):
        if sh == truth and all(s == truth for _, s in stamps[i:]):
            settled = ct
            break
    return {
        "n_changes": len(res["changes"]),
        "correct_pct": res["held"].get(truth, 0.0) / duration * 100,
        "settled_s": settled,
        "final_ok": res["final"] == truth,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--truth", type=int, required=True)
    ap.add_argument("--expect-baseline-changes", type=int, default=None,
                    help="assert the baseline reproduces this many changes")
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args()

    mono = load_mono(Path(args.wav))
    print("song: %.0fs" % (len(mono) / ANALYSIS_SR))
    ref = whole_song_reference(mono)
    print("whole-song reference: %s shift %+d (margin %.3f)\n"
          % (ref["key"], ref["shift"], ref["margin"]))

    t0 = time.perf_counter()
    full_blocks, silents = chroma_frames(mono, n_octaves=6)
    print("full-range chroma: %d cycles in %.1fs" % (len(full_blocks), time.perf_counter() - t0))
    t0 = time.perf_counter()
    bass_blocks, _ = chroma_frames(mono, n_octaves=2)  # C2-C4 only
    print("bass chroma:       %d cycles in %.1fs\n" % (len(bass_blocks), time.perf_counter() - t0))

    STICKY = dict(commit_share=0.55, min_dwell_seconds=30.0)
    variants = [
        Variant("baseline (current)"),
        Variant("window 40s", window=40.0),
        Variant("window 60s", window=60.0),
        Variant("cumulative window", window=1e6),
        Variant("slow vote decay (30s)", decider=dict(decay_halflife=30.0)),
        Variant("no vote decay", decider=dict(decay_halflife=1e9)),
        Variant("cumulative + no decay", window=1e6, decider=dict(decay_halflife=1e9)),
        Variant("dwell 30s", decider=STICKY),
        Variant("share 0.75", decider=dict(commit_share=0.75)),
        Variant("bass 0.3", bass_weight=0.3),
        Variant("bass 0.5", bass_weight=0.5),
        Variant("bass 0.7", bass_weight=0.7),
        Variant("profile shaath", profile="shaath"),
        Variant("profile temperley", profile="temperley"),
        Variant("profile krumhansl", profile="krumhansl"),
        Variant(
            "cum + nodecay + share .75",
            window=1e6,
            decider=dict(decay_halflife=1e9, commit_share=0.75, min_dwell_seconds=30.0),
        ),
        Variant(
            "cum + nodecay + bass .5",
            window=1e6,
            bass_weight=0.5,
            decider=dict(decay_halflife=1e9),
        ),
        Variant(
            "everything",
            window=1e6,
            bass_weight=0.5,
            decider=dict(decay_halflife=1e9, commit_share=0.75, min_dwell_seconds=30.0),
        ),
    ]

    print("truth = %+d" % args.truth)
    print("%-28s %8s %9s %9s %6s" % ("variant", "changes", "correct%", "settled", "final"))
    print("-" * 66)
    results = {}
    t0 = time.perf_counter()
    baseline_changes = None
    for v in variants:
        res = simulate(full_blocks, bass_blocks, silents, v)
        sc = score(res, args.truth)
        results[v.name] = (res, sc)
        if v.name.startswith("baseline"):
            baseline_changes = sc["n_changes"]
        print("%-28s %8d %8.0f%% %8.0fs %6s"
              % (v.name, sc["n_changes"], sc["correct_pct"], sc["settled_s"],
                 "ok" if sc["final_ok"] else "WRONG"))
    print("-" * 66)
    print("%d variants in %.2fs" % (len(variants), time.perf_counter() - t0))

    if args.expect_baseline_changes is not None:
        if baseline_changes != args.expect_baseline_changes:
            print(
                "\nFIDELITY FAIL: baseline gave %d changes, real pipeline gave %d. "
                "The harness does not match the live system; do not trust these rows."
                % (baseline_changes, args.expect_baseline_changes)
            )
            return 1
        print("\nfidelity ok: baseline reproduces the real pipeline (%d changes)"
              % baseline_changes)

    if args.details:
        for name, (res, sc) in results.items():
            if sc["n_changes"]:
                print("\n%s:" % name)
                for ct, o, ns, k in res["changes"]:
                    print("   %5.0fs  %+d -> %+d  (%s)" % (ct, o, ns, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
