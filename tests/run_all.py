"""Run every test except the audible live one."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import test_autoroute  # noqa: E402
import test_diag  # noqa: E402
import test_engine  # noqa: E402
import test_keydetect  # noqa: E402
import test_roundtrip  # noqa: E402
import test_shifter  # noqa: E402

SUITES = [
    ("key detection", [
        test_keydetect.test_profiles,
        test_keydetect.test_incremental_matches_offline,
        test_keydetect.test_lock_in_time,
    ]),
    ("pitch shifter", [
        test_shifter.test_pitch_is_correct,
        test_shifter.test_latency_constant_across_shifts,
        test_shifter.test_crossfade_beats_abrupt_change,
    ]),
    ("round trip", [
        test_roundtrip.test_roundtrip_all_keys,
        test_roundtrip.test_cpu_load,
        test_roundtrip.test_latency_budget,
    ]),
    ("diagnostics", [
        test_diag.test_hot_path_is_cheap,
        test_diag.test_series_wraps_without_growing,
        test_diag.test_priming_is_not_reported_as_a_dropout,
        test_diag.test_underrun_rebuilds_the_cushion,
        test_diag.test_verdict_names_the_cause,
        test_diag.test_priming_underruns_are_excused,
    ]),
    ("engine", [
        test_engine.test_full_pipeline_locks_and_shifts,
        test_engine.test_silence_does_not_lock,
    ]),
    # Last: these monkeypatch the devices module (and restore it afterwards).
    ("auto-routing", [
        test_autoroute.test_plan_prefers_your_listening_device,
        test_autoroute.test_plan_refuses_with_one_device,
        test_autoroute.test_applies_and_restores,
        test_autoroute.test_does_not_unmute_what_was_already_muted,
        test_autoroute.test_crash_recovery,
        test_autoroute.test_wont_restore_a_live_instances_settings,
        test_autoroute.test_dead_pid_is_recoverable,
    ]),
]


def main() -> int:
    failures = []
    for name, fns in SUITES:
        print("=" * 72)
        print(name.upper())
        print("=" * 72)
        for fn in fns:
            t0 = time.perf_counter()
            try:
                fn()
            except Exception:
                failures.append(f"{name}: {fn.__name__}")
                traceback.print_exc()
            print("  [%s in %.1fs]\n" % (fn.__name__, time.perf_counter() - t0))

    print("=" * 72)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("all suites passed")
    print("(tests/test_live.py exercises the real sound card and is audible; run it separately)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
