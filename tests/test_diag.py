"""Diagnostics must be cheap on the audio path, and honest about what it sees.

The first test is the important one: if recording metrics costs real time inside
the capture loop or the output callback, the instrumentation causes the dropouts
it is supposed to diagnose.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotranspose.diag import (  # noqa: E402
    BLOCK_FIELDS,
    CB_FIELDS,
    EV_CLIP,
    EV_UNDERRUN,
    Diagnostics,
)
from autotranspose.ring import RingBuffer  # noqa: E402

SR = 48000
BLOCK = 1024


def _diag(**kw):
    return Diagnostics(samplerate=SR, blocksize=BLOCK, log_path=None, **kw)


def test_hot_path_is_cheap():
    """Recording a block must cost a tiny fraction of the audio budget."""
    d = _diag()
    row = np.zeros(len(BLOCK_FIELDS))
    cb = np.zeros(len(CB_FIELDS))
    n = 20000
    for _ in range(200):
        d.record_block(row)
        d.record_callback(cb)
        d.event(EV_CLIP)

    t0 = time.perf_counter()
    for _ in range(n):
        d.record_block(row)
    block_us = (time.perf_counter() - t0) / n * 1e6

    t0 = time.perf_counter()
    for _ in range(n):
        d.record_callback(cb)
    cb_us = (time.perf_counter() - t0) / n * 1e6

    t0 = time.perf_counter()
    for _ in range(n):
        d.event(EV_CLIP)
    ev_us = (time.perf_counter() - t0) / n * 1e6

    budget_us = BLOCK / SR * 1e6
    print("Hot-path instrumentation cost (budget %.0f us per block):" % budget_us)
    print("  record_block    %6.2f us  (%.3f%% of budget)" % (block_us, block_us / budget_us * 100))
    print("  record_callback %6.2f us  (%.3f%% of budget)" % (cb_us, cb_us / budget_us * 100))
    print("  event           %6.2f us" % ev_us)
    assert block_us < budget_us * 0.01, "record_block costs %.1f us" % block_us
    assert cb_us < budget_us * 0.01, "record_callback costs %.1f us" % cb_us
    assert ev_us < budget_us * 0.01, "event costs %.1f us" % ev_us
    print("  OK (all under 1% of the block budget)")


def test_series_wraps_without_growing():
    """The metric store is fixed-size: no unbounded memory in a long session."""
    d = _diag(capacity=128)
    row = np.arange(len(BLOCK_FIELDS), dtype=np.float64)
    for i in range(1000):
        row[0] = i
        d.record_block(row)
    snap = d.blocks.snapshot()
    print("\nRing of 128 after 1000 writes: %d rows kept, first t=%.0f last t=%.0f"
          % (snap["t"].size, snap["t"][0], snap["t"][-1]))
    assert snap["t"].size == 128
    assert snap["t"][-1] == 999, "newest sample missing"
    assert snap["t"][0] == 872, "wrap kept the wrong window"
    print("  OK")


def test_priming_is_not_reported_as_a_dropout():
    """Silence while the ring fills is by design, and must not look like a glitch."""
    d = _diag()
    ring = RingBuffer(BLOCK * 8, 2, prefill_frames=BLOCK * 4, diag=d)
    out = np.zeros((BLOCK, 2), dtype=np.float32)

    for _ in range(4):  # callbacks before any audio exists
        ring.read(BLOCK, out)
    assert ring.underruns == 0, "priming counted as an underrun"
    assert ring.priming_frames == 4 * BLOCK

    for _ in range(4):
        ring.write(np.ones((BLOCK, 2), dtype=np.float32))
    served = ring.read(BLOCK, out)
    print("\nPriming: %d frames of intentional silence, then served %d" % (ring.priming_frames, served))
    assert served == BLOCK
    assert ring.underruns == 0
    print("  OK")


def test_underrun_rebuilds_the_cushion():
    d = _diag()
    ring = RingBuffer(BLOCK * 8, 2, prefill_frames=BLOCK * 4, diag=d)
    out = np.zeros((BLOCK, 2), dtype=np.float32)
    for _ in range(4):
        ring.write(np.ones((BLOCK, 2), dtype=np.float32))
    ring.read(BLOCK, out)  # primes and serves
    # Drain it dry.
    for _ in range(6):
        ring.read(BLOCK, out)
    print("\nAfter draining: underruns=%d, re-primed=%s" % (ring.underruns, not ring._primed))
    assert ring.underruns > 0
    assert not ring._primed, "should re-prime after running dry"
    assert d.event_counts[EV_UNDERRUN] == ring.underruns
    print("  OK")


def test_verdict_names_the_cause():
    """The verdict must turn numbers into the actual advice."""
    print("\nVerdicts from synthetic stats:")
    d = _diag()
    cases = [
        ({"events": {"underrun": 20}, "elapsed_s": 10.0, "dropout_frames": 8192}, "underrun"),
        ({"events": {}, "elapsed_s": 10.0, "clipped": 500, "out_peak": 1.0}, "clip"),
        ({"events": {}, "elapsed_s": 10.0, "load_pct": 85.0}, "load"),
        ({"events": {}, "elapsed_s": 60.0, "drift_ppm": -3000.0, "drift_window_s": 50},
         "drift"),
        ({"events": {}, "elapsed_s": 10.0, "interval_ms": (20, 30, 64), "ring_ms": 43},
         "jitter"),
        ({"events": {}, "elapsed_s": 10.0}, "nothing wrong"),
    ]
    for stats, expect in cases:
        lines = d.verdict(stats)
        joined = " ".join(lines).lower()
        print("  %-14s -> %s" % (expect, lines[0][:78]))
        assert expect in joined, "expected %r in the verdict, got %r" % (expect, lines)
    print("  OK")


def test_priming_underruns_are_excused():
    d = _diag()
    stats = {"events": {"underrun": 3}, "elapsed_s": 10.0, "dropout_frames": 0}
    lines = d.verdict(stats)
    print("\nUnderruns with no lost audio:")
    print("  %s" % lines[0])
    assert "no audio was actually lost" in lines[0]
    print("  OK")


if __name__ == "__main__":
    test_hot_path_is_cheap()
    test_series_wraps_without_growing()
    test_priming_is_not_reported_as_a_dropout()
    test_underrun_rebuilds_the_cushion()
    test_verdict_names_the_cause()
    test_priming_underruns_are_excused()
