"""A small lock-protected ring buffer joining the capture thread to playback.

Capture (soundcard, blocking) and playback (sounddevice, callback) run on
different clocks, so one must not be able to stall the other. The producer drops
the oldest audio rather than growing without bound, which keeps latency fixed;
the consumer returns silence rather than blocking when it runs dry. Both events
are counted so the UI can show them instead of hiding a glitch.

**Priming matters more than capacity.** Playback begins draining the instant the
stream opens, so without a prefill target the level sits near empty no matter how
large the buffer is, and any late capture block causes a dropout -- measured here
as a ring sitting at 2048 of 8192 frames with steady underruns. So the consumer
emits silence until `prefill_frames` have accumulated, which buys a cushion equal
to the jitter it has to absorb. After a genuine dropout the cushion is gone, so
it re-primes rather than limping along empty.
"""
from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity_frames: int, channels: int, prefill_frames: int = 0, diag=None):
        self.capacity = int(capacity_frames)
        self.channels = channels
        self.prefill = min(int(prefill_frames), self.capacity - 1)
        self._buf = np.zeros((self.capacity, channels), dtype=np.float32)
        self._read = 0
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()
        self.overruns = 0  # producer outran the consumer; oldest audio dropped
        self.underruns = 0  # consumer ran dry; silence emitted
        self.dropped_frames = 0
        self.starved_frames = 0
        self.priming_frames = 0
        self.primes = 0
        self._primed = self.prefill <= 0
        self._diag = diag

    @property
    def filled(self) -> int:
        with self._lock:
            return self._filled

    def write(self, block: np.ndarray) -> None:
        """block is (frames, channels)."""
        frames = block.shape[0]
        if frames > self.capacity:
            block = block[-self.capacity :]
            frames = block.shape[0]
        with self._lock:
            free = self.capacity - self._filled
            if frames > free:
                # Drop the oldest audio so latency stays bounded.
                drop = frames - free
                self._read = (self._read + drop) % self.capacity
                self._filled -= drop
                self.overruns += 1
                self.dropped_frames += drop
                if self._diag is not None:
                    self._diag.event(2)  # EV_OVERRUN
            end = self._write + frames
            if end <= self.capacity:
                self._buf[self._write : end] = block
            else:
                split = self.capacity - self._write
                self._buf[self._write :] = block[:split]
                self._buf[: end - self.capacity] = block[split:]
            self._write = end % self.capacity
            self._filled += frames

    def read(self, frames: int, out: np.ndarray) -> int:
        """Fill `out` (frames, channels); zero-pad what is missing. Returns frames served.

        While priming, emits silence and reports 0 served without counting an
        underrun -- that is the buffer filling on purpose, not a dropout.
        """
        with self._lock:
            if not self._primed:
                if self._filled < self.prefill:
                    out[:] = 0.0
                    self.priming_frames += frames
                    return 0
                self._primed = True
                self.primes += 1
            n = min(frames, self._filled)
            if n:
                end = self._read + n
                if end <= self.capacity:
                    out[:n] = self._buf[self._read : end]
                else:
                    split = self.capacity - self._read
                    out[:split] = self._buf[self._read :]
                    out[split:n] = self._buf[: end - self.capacity]
                self._read = end % self.capacity
                self._filled -= n
            if n < frames:
                out[n:] = 0.0
                self.underruns += 1
                self.starved_frames += frames - n
                # The cushion is spent; rebuild it instead of running on empty.
                self._primed = self.prefill <= 0
                if self._diag is not None:
                    self._diag.event(1)  # EV_UNDERRUN
            return n
