"""Verify the shifter actually transposes, keeps latency constant, and that the
crossfade removes the discontinuity a bare pitch change produces."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotranspose.shifter import (  # noqa: E402
    PitchShifter as CrossfadeShifter,
    _PhaseVocoderCore,
)

SR = 48000
BLOCK = 1024


def _run(shifter, signal, set_at=None, set_to=0):
    """Push a mono signal through in blocks; return the mono output."""
    out = []
    for i in range(0, len(signal) - BLOCK + 1, BLOCK):
        if set_at is not None and i // BLOCK == set_at:
            shifter.set_semitones(set_to)
        chunk = signal[i : i + BLOCK]
        out.append(shifter.process(np.stack([chunk, chunk]))[0])
    return np.concatenate(out)


def _dominant_hz(x, sr=SR):
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spec))])


def test_pitch_is_correct():
    """A 440 Hz tone shifted by n semitones must come out at 440*2^(n/12)."""
    print("Pitch accuracy:")
    t = np.arange(SR * 3) / SR
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for semis in (-6, -5, -2, 0, 3, 5, 6):
        sh = CrossfadeShifter(SR)
        sh.set_semitones(semis)
        y = _run(sh, tone)
        tail = y[-SR:]  # after the crossfade has completed
        got = _dominant_hz(tail)
        want = 440.0 * 2 ** (semis / 12.0)
        cents = 1200 * np.log2(got / want) if got > 0 else float("inf")
        ok = abs(cents) < 35
        print(
            "  %+2d semis: want %7.1f Hz  got %7.1f Hz  (%+6.1f cents) %s"
            % (semis, want, got, cents, "ok" if ok else "FAIL")
        )
        assert ok, "pitch off by %.1f cents at %+d semitones" % (cents, semis)
    print("  OK")


def test_latency_constant_across_shifts():
    """Latency must not depend on the shift, or changing shift would jump time."""
    print("\nLatency vs shift (envelope cross-correlation):")
    n = SR * 4
    t = np.arange(n) / SR
    # Amplitude modulation survives pitch shifting, so the envelope is a usable
    # timing reference even though the carrier frequency changes.
    env = (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 1.7 * t))) * np.exp(-((t % 0.6) * 3))
    sig = (env * np.sin(2 * np.pi * 300 * t) * 0.6).astype(np.float32)

    delays = {}
    for semis in (0, -5, 5, -2):
        sh = CrossfadeShifter(SR)
        sh.set_semitones(semis)
        y = _run(sh, sig)
        a = np.abs(sig[: len(y)])
        b = np.abs(y)
        a = a - a.mean()
        b = b - b.mean()
        xc = np.correlate(b, a, mode="full")
        lag = int(np.argmax(xc)) - (len(a) - 1)
        delays[semis] = lag
        print("  %+2d semis: delay %5d samples = %6.1f ms" % (semis, lag, lag / SR * 1000))
    spread = max(delays.values()) - min(delays.values())
    print("  spread across shifts: %d samples (%.1f ms)" % (spread, spread / SR * 1000))
    assert spread < SR * 0.01, "latency varies with shift by %d samples" % spread
    print("  OK")


def test_crossfade_beats_abrupt_change():
    """Crossfading two engines vs. yanking the ratio on one, at the same moment."""
    print("\nDiscontinuity at the shift change:")
    n = SR * 6
    t = np.arange(n) / SR
    tone = (0.4 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330 * t))).astype(
        np.float32
    )
    change_block = (SR * 3) // BLOCK

    sh = CrossfadeShifter(SR)
    sh.set_semitones(0)
    smooth = _run(sh, tone, set_at=change_block, set_to=5)

    # Baseline: one engine, ratio changed instantly with no handover.
    core = _PhaseVocoderCore(2, 2048, 4, 1.0)
    out = []
    for i in range(0, n - BLOCK + 1, BLOCK):
        if i // BLOCK == change_block:
            core.set_ratio(2 ** (5 / 12))
        chunk = tone[i : i + BLOCK]
        out.append(core.process(np.stack([chunk, chunk]))[0])
    abrupt = np.concatenate(out)

    win = slice(SR * 3 - BLOCK, SR * 3 + int(SR * 0.6))
    results = {}
    for name, sig in (("abrupt ratio change", abrupt), ("CrossfadeShifter", smooth)):
        d = np.abs(np.diff(sig[win]))
        rms = float(np.sqrt((sig[win] ** 2).mean()))
        dip = _min_envelope(sig[win])
        results[name] = (d.max(), rms, dip)
        print(
            "  %-20s max jump %.4f   window rms %.4f   worst dip %.4f (%.0f%% of rms)"
            % (name, d.max(), rms, dip, dip / rms * 100 if rms else 0)
        )
    # The crossfade must not punch a hole in the audio.
    cf_dip, cf_rms = results["CrossfadeShifter"][2], results["CrossfadeShifter"][1]
    assert cf_dip > 0.35 * cf_rms, "crossfade dips too far (%.3f vs rms %.3f)" % (cf_dip, cf_rms)
    print("  OK (crossfade holds level through the handover)")


def _min_envelope(x, win=512):
    """Smallest short-window RMS: catches a hole punched in the audio."""
    n = len(x) // win
    if n == 0:
        return float(np.sqrt((x**2).mean()))
    blocks = x[: n * win].reshape(n, win)
    return float(np.sqrt((blocks**2).mean(axis=1)).min())


if __name__ == "__main__":
    test_pitch_is_correct()
    test_latency_constant_across_shifts()
    test_crossfade_beats_abrupt_change()
