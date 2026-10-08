"""Engine wiring test with fake audio devices.

Drives the real audio and analysis threads against a synthetic recorder, so the
whole pipeline is exercised -- capture -> analysis -> decision -> shift ->
playback -- without touching the sound card or making any noise.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import synth_song  # noqa: E402

from autotranspose import devices, engine as engine_mod  # noqa: E402
from autotranspose.engine import Engine, EngineConfig  # noqa: E402
from autotranspose.keyprofiles import shift_class_for_key, signed_shift  # noqa: E402

SR = 48000
BLOCK = 1024


class FakeRecorder:
    """Serves a looping signal in real time, like a sound card would."""

    def __init__(self, signal, channels=2):
        self.signal = signal
        self.pos = 0
        self.channels = channels
        self.next_due = time.perf_counter()

    def record(self, numframes):
        # Pace it so the engine sees real-time arrival rather than a tight loop.
        self.next_due += numframes / SR
        delay = self.next_due - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        out = np.empty((numframes, self.channels), dtype=np.float32)
        for i in range(numframes):
            out[i, :] = self.signal[(self.pos + i) % len(self.signal)]
        self.pos = (self.pos + numframes) % len(self.signal)
        return out


class FakeOutputStream:
    """Drives the engine's output callback in real time, like PortAudio does.

    This exercises the ring buffer for real: if the capture side cannot keep the
    ring fed, the callback records underruns exactly as the sound card would.
    """

    def __init__(self, callback, blocksize, samplerate, channels):
        self.callback = callback
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.channels = channels
        self.frames = 0
        self.blocks = 0
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        buf = np.zeros((self.blocksize, self.channels), dtype=np.float32)
        due = time.perf_counter()
        while not self._stop.is_set():
            due += self.blocksize / self.samplerate
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            buf[:] = 0.0
            self.callback(buf, self.blocksize, None, None)
            self.frames += self.blocksize
            self.blocks += 1
            self.peak = max(self.peak, float(np.abs(buf).max()))

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


def install_fakes(signal, holder):
    rec = FakeRecorder(signal)

    @contextlib.contextmanager
    def fake_recorder(device, samplerate, channels, blocksize):
        yield rec

    devices.open_recorder = fake_recorder
    engine_mod.devices = devices

    def open_output_stream(self):
        stream = FakeOutputStream(
            self._output_callback, self.cfg.blocksize, self.cfg.samplerate, self.cfg.channels
        )
        holder["stream"] = stream
        return stream

    Engine._open_output_stream = open_output_stream


def test_full_pipeline_locks_and_shifts():
    tonic_pc, minor = 7, False  # G major -> wants +5
    want_class = shift_class_for_key(tonic_pc + (12 if minor else 0))
    want_shift = signed_shift(want_class)

    song = synth_song(tonic_pc, minor_key=minor, duration=30.0, sr=SR, seed=11)
    holder = {}
    install_fakes(song, holder)

    cfg = EngineConfig(
        samplerate=SR, blocksize=BLOCK, min_heard_seconds=10.0, min_dwell_seconds=5.0
    )
    cap = devices.Device("fake-loopback", "Fake Loopback", 2, is_loopback=True)
    out = devices.Device("fake-out", "Fake Headphones", 2)
    eng = Engine(cap, out, cfg)

    print("Driving the engine in real time against a G major signal (wants %+d)..." % want_shift)
    locked_at = None
    t_start = time.perf_counter()
    with eng:
        while time.perf_counter() - t_start < 26.0:
            time.sleep(0.5)
            d = eng.status.decision
            if eng.status.error:
                raise AssertionError("engine error: %s" % eng.status.error)
            if d.status == "locked" and locked_at is None:
                locked_at = time.perf_counter() - t_start
                print(
                    "  locked after %.1fs: %s, applying %+d"
                    % (locked_at, d.key_name, eng.status.applied_shift)
                )
            if locked_at and time.perf_counter() - t_start > locked_at + 6:
                break

    st = eng.status
    d = st.decision
    print("  final status:    %s" % d.status)
    print("  detected key:    %s (conf %.1f%%)" % (d.key_name, d.key_confidence * 100))
    print("  applied shift:   %+d (wanted %+d)" % (st.applied_shift, want_shift))
    print("  audio load:      %.1f%% of one block" % (st.audio_load * 100))
    print("  analysis:        %.1f ms per cycle" % st.analysis_ms)
    player = holder["stream"]
    print("  playback:        %d blocks, %d frames, peak %.3f"
          % (player.blocks, player.frames, player.peak))
    print("  ring:            %d frames held, %d underruns, %d overruns"
          % (st.ring_frames, st.underruns, st.overruns))
    print("  dropped blocks:  %d" % st.dropped_analysis_blocks)

    assert st.error is None
    assert player.blocks > 100, "playback thread barely ran (%d blocks)" % player.blocks
    assert player.peak > 0.01, "playback was silent"
    assert d.status == "locked", "never locked (status=%s)" % d.status
    assert st.applied_shift == want_shift, (
        "applied %+d, expected %+d" % (st.applied_shift, want_shift)
    )
    assert st.audio_load < 0.6, "audio thread too slow: %.1f%%" % (st.audio_load * 100)
    assert st.dropped_analysis_blocks == 0, "analysis fell behind"
    # A couple of underruns while the ring primes is fine; a steady stream is not.
    assert st.underruns < 20, "playback kept running dry (%d underruns)" % st.underruns
    print("  OK")


def test_silence_does_not_lock():
    song = np.zeros((SR * 4, 2), dtype=np.float32)
    holder = {}
    install_fakes(song, holder)
    cfg = EngineConfig(samplerate=SR, blocksize=BLOCK, min_heard_seconds=3.0)
    cap = devices.Device("fake-loopback", "Fake Loopback", 2, is_loopback=True)
    eng = Engine(cap, devices.Device("fake-out", "Fake Headphones", 2), cfg)
    print("\nSilence must not produce a lock:")
    with eng:
        time.sleep(8.0)
    print("  status after 8s of silence: %s, shift %+d"
          % (eng.status.decision.status, eng.status.applied_shift))
    assert eng.status.decision.status in ("silent", "listening", "starting")
    assert eng.status.applied_shift == 0
    print("  OK")


if __name__ == "__main__":
    test_full_pipeline_locks_and_shifts()
    test_silence_does_not_lock()
