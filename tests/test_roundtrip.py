"""The test that matters: shift a song in a known key and confirm it lands on C.

This exercises the detector and the shifter together, the same way the live app
does -- detect the key, compute the shift, apply it, then re-detect the output
and check it is now C major / A minor (shift class 0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import synth_song  # noqa: E402

from autotranspose.keydetect import ANALYSIS_SR, detect_key_offline  # noqa: E402
from autotranspose.keyprofiles import (  # noqa: E402
    PITCH_CLASS_NAMES,
    shift_class_for_key,
    signed_shift,
)
from autotranspose.shifter import PitchShifter, _PhaseVocoderCore  # noqa: E402

SR = 48000
BLOCK = 1024


def shift_signal(mono: np.ndarray, semitones: int, sr: int = SR) -> np.ndarray:
    """Push a mono signal through the real shifter in real-time-sized blocks."""
    core = _PhaseVocoderCore(1, 2048, 4, 2 ** (semitones / 12.0))
    out = []
    for i in range(0, len(mono) - BLOCK + 1, BLOCK):
        out.append(core.process(mono[np.newaxis, i : i + BLOCK])[0])
    return np.concatenate(out)


def test_roundtrip_all_keys():
    import soxr

    print("Round trip: detect key -> shift -> re-detect. Target is class 0 (C / Am).")
    ok_count = 0
    rows = []
    for minor in (False, True):
        for pc in range(12):
            y = synth_song(
                pc, minor_key=minor, duration=22.0, sr=SR, percussion=0.25, seed=pc + 7 * minor
            )
            est = detect_key_offline(soxr.resample(y, SR, ANALYSIS_SR), ANALYSIS_SR)
            shift = signed_shift(est.shift_class)

            shifted = shift_signal(y, shift)
            after = detect_key_offline(soxr.resample(shifted, SR, ANALYSIS_SR), ANALYSIS_SR)

            want_class = shift_class_for_key(pc + (12 if minor else 0))
            detected_right = est.shift_class == want_class
            landed = after.shift_class == 0
            ok_count += landed
            rows.append(
                (
                    PITCH_CLASS_NAMES[pc] + (" minor" if minor else " major"),
                    shift,
                    est.key_name,
                    after.key_name,
                    detected_right,
                    landed,
                )
            )

    for label, shift, before, after, det_ok, landed in rows:
        print(
            "  %-9s shift %+2d  %-9s -> %-9s  detect:%s land:%s"
            % (label, shift, before, after, "ok" if det_ok else "MISS", "ok" if landed else "MISS")
        )
    acc = ok_count / len(rows)
    print("\n  landed on the white keys: %d/%d (%.1f%%)" % (ok_count, len(rows), acc * 100))
    assert acc >= 0.95, "round trip only landed %.1f%% of the time" % (acc * 100)
    print("  OK")


def test_cpu_load():
    """Real-time safety margin, measured on blocks the live app actually uses."""
    print("\nCPU load (stereo @ %d Hz):" % SR)
    rng = np.random.default_rng(0)
    for n_fft in (1024, 2048, 4096):
        for engines in (1, 2):
            cores = [_PhaseVocoderCore(2, n_fft, 4, 2 ** (5 / 12.0)) for _ in range(engines)]
            x = (rng.standard_normal((2, BLOCK)) * 0.1).astype(np.float32)
            for _ in range(20):
                for c in cores:
                    c.process(x)
            n = 200
            t0 = time.perf_counter()
            for _ in range(n):
                for c in cores:
                    c.process(x)
            per_block = (time.perf_counter() - t0) / n * 1000
            audio_ms = BLOCK / SR * 1000
            print(
                "  n_fft %5d, %d engine%s: %5.2f ms per %.1f ms block = %4.1f%% of one core"
                % (n_fft, engines, " " if engines == 1 else "s", per_block, audio_ms,
                   per_block / audio_ms * 100)
            )
    print("  (2 engines = crossfade enabled; 1 = --no-crossfade)")


def test_latency_budget():
    sh = PitchShifter(SR, 2, n_fft=2048)
    print("\nShifter algorithmic latency: %.1f ms" % (sh.latency_seconds * 1000))


if __name__ == "__main__":
    test_roundtrip_all_keys()
    test_cpu_load()
    test_latency_budget()
