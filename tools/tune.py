"""Tune the white-key/gain detector against several real songs at once.

Settings that win on one song mean nothing, so every combination is scored on
every song and ranked by the worst song, not the average. The metric is the one
the app exists for: the share of the music that actually landed on white keys,
with the number of mid-song changes as the tie-break.

    python tools/tune.py capture/song1/input.wav capture/song2/input.wav:20
    (an optional ":seconds" suffix trims that many seconds off the end)
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotranspose.decide import ShiftDecider  # noqa: E402
from autotranspose.keydetect import ANALYSIS_SR, KeyEstimator  # noqa: E402
from autotranspose.keyprofiles import signed_shift, white_mass_per_shift  # noqa: E402
from tools.experiment import CYCLE, chroma_frames  # noqa: E402
from tools.replay import load_mono  # noqa: E402


def load_song(spec: str) -> dict:
    path_str, _, trim = spec.rpartition(":")
    if not path_str:  # no trim suffix
        path_str, trim = spec, "0"
    wav = Path(path_str)
    cache = wav.with_suffix(".tune-chroma.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        blocks, silents = list(d["blocks"]), list(d["silents"])
    else:
        mono = load_mono(wav, trim_end=float(trim))
        blocks, silents = chroma_frames(mono, n_octaves=6)
        np.savez_compressed(
            cache, blocks=np.array(blocks, dtype=object), silents=np.array(silents)
        )
    total = np.zeros(12)
    for b, sil in zip(blocks, silents):
        if not sil and b.size:
            total += b.sum(axis=1)
    wm = white_mass_per_shift(total)
    return {
        "name": wav.parent.name,
        "blocks": blocks,
        "silents": silents,
        "duration": len(blocks) * CYCLE,
        "ceiling": float(wm.max()),
        "ceiling_shift": signed_shift(int(np.argmax(wm))),
        "nothing": float(wm[0]),
    }


def achieved(song, changes) -> float:
    """Share of energy that landed on white keys, given when the shift changed."""
    stamps = [(0.0, 0)] + [(c[0], c[2]) for c in changes]
    idx, t, white, total = 0, 0.0, 0.0, 0.0
    pcs = np.arange(12)
    from autotranspose.keyprofiles import _WHITE_MASK

    for k, block in enumerate(song["blocks"]):
        t += CYCLE
        while idx + 1 < len(stamps) and stamps[idx + 1][0] <= t:
            idx += 1
        cur = stamps[idx][1] % 12
        if song["silents"][k] or block.size == 0:
            continue
        fs = block.sum(axis=1)
        tot = fs.sum()
        if tot <= 1e-12:
            continue
        white += fs[_WHITE_MASK[(pcs + cur) % 12]].sum()
        total += tot
    return white / total if total else 0.0


def run(song, *, window, min_gain, dwell, min_heard, stable):
    est = KeyEstimator(window_seconds=window, profile="blend", objective="white")
    dec = ShiftDecider(
        cycle_seconds=CYCLE,
        mode="gain",
        min_gain=min_gain,
        min_dwell_seconds=dwell,
        min_heard_seconds=min_heard,
        gain_stable_cycles=stable,
    )
    applied, t, changes = 0, 0.0, []
    for k, block in enumerate(song["blocks"]):
        t += CYCLE
        est.add_frames(block)
        e = est.estimate() if not song["silents"][k] else None
        new = dec.update(e, heard_seconds=est.filled_seconds, silent=song["silents"][k], now=t)
        if new != applied:
            changes.append((t, applied, new))
            applied = new
    return changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("songs", nargs="+")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    songs = []
    for spec in args.songs:
        t0 = time.perf_counter()
        s = load_song(spec)
        songs.append(s)
        print(
            "%-7s %5.0fs  ceiling %+d = %.1f%% white   doing nothing %.1f%%  (%.1fs)"
            % (s["name"], s["duration"], s["ceiling_shift"], s["ceiling"] * 100,
               s["nothing"] * 100, time.perf_counter() - t0)
        )
    print()

    windows = [20.0, 30.0, 45.0, 60.0, 90.0, 1e6]
    gains = [0.02, 0.03, 0.05, 0.08]
    dwells = [12.0, 25.0, 45.0]
    heards = [10.0, 15.0, 20.0]
    stables = [3, 5]

    combos = list(itertools.product(windows, gains, dwells, heards, stables))
    print("%d combinations x %d songs..." % (len(combos), len(songs)))
    rows = []
    t0 = time.perf_counter()
    for w, g, dw, mh, st in combos:
        per = []
        for s in songs:
            ch = run(s, window=w, min_gain=g, dwell=dw, min_heard=mh, stable=st)
            per.append((achieved(s, ch) / s["ceiling"], len(ch), achieved(s, ch)))
        rows.append(
            {
                "p": (w, g, dw, mh, st),
                "worst_ratio": min(x[0] for x in per),
                "changes": sum(x[1] for x in per),
                "per": per,
            }
        )
    print("done in %.1fs\n" % (time.perf_counter() - t0))

    # Closest to the ceiling on the worst song, then fewest changes.
    rows.sort(key=lambda r: (-r["worst_ratio"], r["changes"]))
    hdr = "%-7s %-5s %-6s %-7s %-7s %9s %8s" % (
        "window", "gain", "dwell", "mheard", "stable", "worst%ceil", "changes"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows[: args.top]:
        w, g, dw, mh, st = r["p"]
        detail = "  ".join("%s:%.0f%%/%dch" % (s["name"], p[2] * 100, p[1])
                           for s, p in zip(songs, r["per"]))
        print(
            "%-7s %-5.0f %-6g %-7g %-7d %8.1f%% %7d   %s"
            % ("cumul" if w > 1e5 else "%g" % w, g * 100, dw, mh, st,
               r["worst_ratio"] * 100, r["changes"], detail)
        )
    print("-" * len(hdr))
    cur = next((r for r in rows if r["p"] == (20.0, 0.03, 12.0, 10.0, 3)), None)
    if cur:
        print(
            "current defaults (w20 g3 d12 m10 s3): worst %.1f%% of ceiling, %d changes"
            % (cur["worst_ratio"] * 100, cur["changes"])
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
