"""Replay a recorded song through the detector, offline and fast.

The live pipeline can only be tested in real time, once per attempt. This feeds a
recorded `input.wav` through exactly the same chroma -> estimate -> decide chain,
as fast as the CPU allows, so a change can be judged in seconds instead of
needing the song played again.

    python tools/replay.py capture/song1/input.wav
    python tools/replay.py capture/song1/input.wav --variant cumulative --bass 0.5

What it reports is what the user actually complained about: how many times the
transpose changed during one song.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

from autotranspose.decide import ShiftDecider  # noqa: E402
from autotranspose.keydetect import ANALYSIS_SR, IncrementalChroma, KeyEstimator  # noqa: E402
from autotranspose.keyprofiles import PITCH_CLASS_NAMES, key_label, signed_shift  # noqa: E402


def load_mono(path: Path, sr: int = ANALYSIS_SR, trim_end: float = 0.0) -> np.ndarray:
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    if trim_end > 0:
        cut = int(trim_end * file_sr)
        mono = mono[:-cut] if cut < len(mono) else mono[:0]
    if file_sr != sr:
        mono = soxr.resample(mono, file_sr, sr)
    return mono.astype(np.float32)


def replay(
    mono: np.ndarray,
    *,
    cycle_seconds: float = 1.5,
    window: float = 20.0,
    profile: str = "blend",
    decider_kwargs: dict | None = None,
    estimator_cls=KeyEstimator,
    estimator_kwargs: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Run the detector over `mono` the way the live engine would."""
    chroma = IncrementalChroma(sr=ANALYSIS_SR)
    est = estimator_cls(
        window_seconds=window, profile=profile, **(estimator_kwargs or {})
    )
    dec = ShiftDecider(cycle_seconds=cycle_seconds, **(decider_kwargs or {}))

    step = int(cycle_seconds * ANALYSIS_SR)
    changes: list[tuple[float, int, int, str]] = []
    timeline: list[tuple[float, str, int, str, float, float]] = []
    applied = 0
    t_fake = 0.0
    silent_rms = 10 ** (-58.0 / 20.0)

    for i in range(0, len(mono) - step + 1, step):
        chunk = mono[i : i + step]
        t_fake += cycle_seconds
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        silent = rms < silent_rms

        est.add_frames(chroma.push(chunk))
        estimate = est.estimate() if not silent else None
        new = dec.update(
            estimate, heard_seconds=est.filled_seconds, silent=silent, now=t_fake
        )
        st = dec.state
        if new != applied:
            changes.append((t_fake, applied, new, st.key_name))
            applied = new
        timeline.append(
            (t_fake, st.status, new, st.key_name, st.leader_share, st.margin)
        )
        if verbose:
            print(
                "  %6.1fs %-9s shift %+d %-9s share %3.0f%% margin %.3f"
                % (t_fake, st.status, new, st.key_name, st.leader_share * 100, st.margin)
            )

    held: dict[int, float] = {}
    prev_t, prev_shift = 0.0, 0
    for t, _old, new, _k in changes:
        held[prev_shift] = held.get(prev_shift, 0.0) + (t - prev_t)
        prev_t, prev_shift = t, new
    end = len(mono) / ANALYSIS_SR
    held[prev_shift] = held.get(prev_shift, 0.0) + (end - prev_t)

    return {
        "duration": end,
        "changes": changes,
        "n_changes": len(changes),
        "final": applied,
        "held": held,
        "timeline": timeline,
        "dominant": max(held, key=held.get) if held else 0,
        "dominant_fraction": (max(held.values()) / end) if held else 0.0,
    }


def whole_song_reference(mono: np.ndarray, profile: str = "blend") -> dict:
    """What the detector says with the entire song in hand -- the best it can do."""
    from autotranspose.keydetect import detect_key_offline

    est = detect_key_offline(mono, ANALYSIS_SR, profile=profile)
    return {
        "key": est.key_name,
        "shift_class": est.shift_class,
        "shift": signed_shift(est.shift_class),
        "confidence": est.key_confidence,
        "margin": est.margin,
        "scores": est.shift_scores,
    }


def report(name: str, res: dict) -> None:
    print(
        "%-28s changes=%-2d final=%+d  dominant=%+d held %.0f%% of the song"
        % (
            name,
            res["n_changes"],
            res["final"],
            res["dominant"],
            res["dominant_fraction"] * 100,
        )
    )
    for t, old, new, key in res["changes"]:
        print("      %5.0fs  %+d -> %+d  (%s)" % (t, old, new, key))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--profile", default="blend")
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--trim-end", type=float, default=0.0,
                    help="drop this many seconds off the end (e.g. a different song)")
    args = ap.parse_args()

    path = Path(args.wav)
    t0 = time.perf_counter()
    mono = load_mono(path, trim_end=args.trim_end)
    print(
        "loaded %s: %.1fs at %d Hz (%.1fs to load/resample)"
        % (path.name, len(mono) / ANALYSIS_SR, ANALYSIS_SR, time.perf_counter() - t0)
    )

    ref = whole_song_reference(mono, args.profile)
    print(
        "\nwhole-song reference: %s  shift %+d  (class %d, conf %.0f%%, margin %.3f)"
        % (ref["key"], ref["shift"], ref["shift_class"], ref["confidence"] * 100, ref["margin"])
    )
    order = np.argsort(ref["scores"])[::-1]
    print("  top shift classes:")
    for c in order[:4]:
        maj = PITCH_CLASS_NAMES[(-c) % 12]
        minr = PITCH_CLASS_NAMES[(9 - c) % 12]
        print(
            "    class %2d (%+d)  score %.3f   %s major / %s minor"
            % (c, signed_shift(int(c)), ref["scores"][c], maj, minr)
        )

    print()
    t0 = time.perf_counter()
    res = replay(mono, profile=args.profile, window=args.window, verbose=args.verbose)
    print("current algorithm (replayed in %.1fs):" % (time.perf_counter() - t0))
    report("  live behaviour", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
