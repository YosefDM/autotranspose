"""Grid-search the detector's knobs against one or more recorded songs.

Each song's chroma is computed once and cached to .npz, so a grid of hundreds of
combinations runs in under a minute. Tuning on a single song overfits, so the
score is the worst song's result, not the average, and the number of songs is
printed loudly.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.experiment import CYCLE, Variant, chroma_frames, score, simulate  # noqa: E402
from tools.replay import load_mono, whole_song_reference  # noqa: E402


def cached_chroma(wav: Path, n_octaves: int):
    cache = wav.with_suffix(f".chroma{n_octaves}.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return list(d["blocks"]), list(d["silents"])
    mono = load_mono(wav)
    blocks, silents = chroma_frames(mono, n_octaves=n_octaves)
    np.savez_compressed(
        cache,
        blocks=np.array(blocks, dtype=object),
        silents=np.array(silents),
    )
    return blocks, silents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "songs", nargs="+", help="wav:truth_shift pairs, e.g. capture/song1/input.wav:-3"
    )
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    songs = []
    for spec in args.songs:
        path_str, _, truth = spec.rpartition(":")
        wav = Path(path_str)
        t0 = time.perf_counter()
        full, silents = cached_chroma(wav, 6)
        songs.append(
            {
                "name": wav.parent.name,
                "full": full,
                "silents": silents,
                "truth": int(truth),
                "duration": len(full) * CYCLE,
            }
        )
        print(
            "%s: %.0fs, %d cycles (%.1fs)"
            % (wav.parent.name, len(full) * CYCLE, len(full), time.perf_counter() - t0)
        )

    if len(songs) == 1:
        print(
            "\n*** Only one song. Whatever wins here is tuned to this song and may\n"
            "*** not generalise. Record one or two more before trusting it.\n"
        )

    windows = [20.0, 30.0, 45.0, 60.0, 90.0, 1e6]
    halflives = [8.0, 15.0, 30.0, 1e9]
    min_heards = [12.0, 20.0, 30.0]
    shares = [0.5, 0.6, 0.7]
    dwells = [12.0, 30.0]

    combos = list(itertools.product(windows, halflives, min_heards, shares, dwells))
    print("evaluating %d combinations over %d song(s)..." % (len(combos), len(songs)))

    rows = []
    t0 = time.perf_counter()
    for w, hl, mh, sh, dw in combos:
        v = Variant(
            "w%g h%g m%g s%g d%g" % (w, hl, mh, sh, dw),
            window=w,
            decider=dict(
                decay_halflife=hl,
                min_heard_seconds=mh,
                commit_share=sh,
                min_dwell_seconds=dw,
            ),
        )
        per_song = []
        for s in songs:
            res = simulate(s["full"], None, s["silents"], v)
            per_song.append(score(res, s["truth"]))
        worst_correct = min(p["correct_pct"] for p in per_song)
        total_changes = sum(p["n_changes"] for p in per_song)
        all_final_ok = all(p["final_ok"] for p in per_song)
        worst_settled = max(p["settled_s"] for p in per_song)
        rows.append(
            {
                "params": (w, hl, mh, sh, dw),
                "worst_correct": worst_correct,
                "changes": total_changes,
                "final_ok": all_final_ok,
                "settled": worst_settled,
            }
        )
    print("done in %.1fs\n" % (time.perf_counter() - t0))

    # Rank: must end correct, then maximise time on the right shift, then fewest
    # changes, then settle soonest.
    rows.sort(
        key=lambda r: (not r["final_ok"], -r["worst_correct"], r["changes"], r["settled"])
    )

    print(
        "%-7s %-7s %-6s %-5s %-5s %9s %8s %9s"
        % ("window", "halflf", "mheard", "share", "dwell", "correct%", "changes", "settled")
    )
    print("-" * 68)
    for r in rows[: args.top]:
        w, hl, mh, sh, dw = r["params"]
        print(
            "%-7s %-7s %-6g %-5g %-5g %8.0f%% %8d %8.0fs"
            % (
                "cumul" if w > 1e5 else "%g" % w,
                "none" if hl > 1e8 else "%g" % hl,
                mh,
                sh,
                dw,
                r["worst_correct"],
                r["changes"],
                r["settled"],
            )
        )
    print("-" * 68)
    base = next(
        r for r in rows if r["params"] == (20.0, 8.0, 12.0, 0.5, 12.0)
    ) if any(r["params"] == (20.0, 8.0, 12.0, 0.5, 12.0) for r in rows) else None
    cur = next((r for r in rows if r["params"][:2] == (20.0, 8.0)), None)
    if cur:
        print(
            "for reference, a current-like setting (w20 h8): correct %.0f%%, "
            "%d changes" % (cur["worst_correct"], cur["changes"])
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
