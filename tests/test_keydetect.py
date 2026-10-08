"""Accuracy and plumbing tests for the key detector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import synth_song  # noqa: E402

from autotranspose.keydetect import (  # noqa: E402
    ANALYSIS_SR,
    IncrementalChroma,
    KeyEstimator,
    detect_key_offline,
)
from autotranspose.keyprofiles import PITCH_CLASS_NAMES, shift_class_for_key  # noqa: E402


def _expected_shift_class(tonic_pc: int, minor: bool) -> int:
    return shift_class_for_key(tonic_pc + (12 if minor else 0))


def accuracy(profile="blend", duration=24.0, percussion=0.25, verbose=True):
    """Detected-shift accuracy over all 24 keys."""
    hits, rows = 0, []
    for minor in (False, True):
        for pc in range(12):
            y = synth_song(
                pc,
                minor_key=minor,
                duration=duration,
                sr=ANALYSIS_SR,
                percussion=percussion,
                seed=pc + 100 * minor,
            )
            est = detect_key_offline(y, ANALYSIS_SR, profile=profile)
            want = _expected_shift_class(pc, minor)
            ok = est.shift_class == want
            hits += ok
            label = PITCH_CLASS_NAMES[pc] + (" minor" if minor else " major")
            rows.append((label, want, est.shift_class, est.key_name, est.key_confidence, ok))
    if verbose:
        for label, want, got, keyname, conf, ok in rows:
            flag = "ok  " if ok else "MISS"
            print(
                "  %s %-9s want class %2d  got %2d  (%-9s conf %5.1f%%)"
                % (flag, label, want, got, keyname, conf * 100)
            )
    return hits / len(rows), rows


def test_profiles():
    print("Shift-class accuracy over all 24 synthetic keys:")
    best = None
    for profile in ("blend", "shaath", "temperley", "krumhansl", "albrecht"):
        acc, _ = accuracy(profile=profile, verbose=False)
        print("  %-10s %6.1f%%" % (profile, acc * 100))
        if best is None or acc > best[1]:
            best = (profile, acc)
    print("  -> best: %s at %.1f%%" % (best[0], best[1] * 100))
    return best


def test_incremental_matches_offline():
    """Incremental chroma must track a whole-window CQT, or live != offline."""
    y = synth_song(7, minor_key=False, duration=20.0, sr=ANALYSIS_SR, seed=3)
    offline = detect_key_offline(y, ANALYSIS_SR)

    inc, est = IncrementalChroma(), KeyEstimator(window_seconds=1e6)
    block = 4096
    for i in range(0, len(y), block):
        est.add_frames(inc.push(y[i : i + block]))
    live = est.estimate()

    print("\nIncremental vs offline on G major:")
    print("  offline: class %d (%s), frames=%d" % (offline.shift_class, offline.key_name, offline.frames))
    print("  live:    class %d (%s), frames=%d" % (live.shift_class, live.key_name, live.frames))
    corr = float(np.corrcoef(offline.shift_scores, live.shift_scores)[0, 1])
    print("  shift-score correlation: %.4f" % corr)
    assert live.shift_class == offline.shift_class == 5, "G major must want +5"
    assert corr > 0.95, "incremental chroma diverges from offline (r=%.3f)" % corr
    print("  OK")


def test_lock_in_time():
    """How much audio before the estimate settles on the right answer?"""
    print("\nTime-to-lock (G major, 1 s steps):")
    y = synth_song(7, minor_key=False, duration=24.0, sr=ANALYSIS_SR, seed=5)
    inc, est = IncrementalChroma(), KeyEstimator(window_seconds=20.0)
    step = ANALYSIS_SR
    first_correct = None
    for i in range(0, len(y) - step, step):
        est.add_frames(inc.push(y[i : i + step]))
        e = est.estimate()
        if e is None:
            continue
        secs = (i + step) / ANALYSIS_SR
        if e.shift_class == 5 and first_correct is None:
            first_correct = secs
        print(
            "  %5.1fs  class %2d conf %5.1f%% margin %.3f %s"
            % (secs, e.shift_class, e.key_confidence * 100, e.margin, "<-" if e.shift_class == 5 else "")
        )
    print("  first correct at %ss" % first_correct)
    return first_correct


if __name__ == "__main__":
    test_profiles()
    print()
    acc, rows = accuracy(profile="blend")
    print("\nblend accuracy: %.1f%%" % (acc * 100))
    test_incremental_matches_offline()
    test_lock_in_time()
