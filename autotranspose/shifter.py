"""Real-time pitch shifting.

Two engines. **Signalsmith Stretch** (MIT, a native DLL built by
`native/build.bat`) is used when present and is the better one by every measure
taken on real music at +4 semitones: level within 0.22 dB instead of 1.72,
log-spectral distance to Rubber Band 6.61 dB instead of 7.58, added warble 1.4
points instead of 5.1, and 1.3% of a core instead of 14.4%. The numpy phase
vocoder below is the fallback when the DLL has not been built, so the app still
runs anywhere.

Why the fallback is hand-written: pedalboard's `PitchShift` (Rubber Band) only works
when it processes a whole signal at once -- fed block by block with
`reset=False` it emits two blocks and then silence -- and pedalboard's real-time
`AudioStream` takes the output device exclusively, which makes its audio
invisible to the WASAPI loopback this app captures from. The `rubberband` PyPI
package needs the C library built, and Signalsmith Stretch has no Python
binding. So the shift is done here, in numpy.

The algorithm is the classic frequency-domain pitch shifter: estimate each bin's
true frequency from the phase advance between frames, scale those frequencies by
the shift ratio, move the magnitudes to the corresponding bins, and resynthesise
with overlap-add. Analysis and synthesis hops are equal, so a block of N input
frames always yields N output frames and the tempo is untouched -- no separate
time-stretch-then-resample stage, and no resampler to rebuild when the shift
changes.

Two refinements matter far more than any of that, because without them the
output measured 3.7-8 dB SNR against the exact ideal shifted tone while putting
100% of its energy on the correct partials -- right frequencies, scrambled
phases, which is what "metallic" and "phasey" actually are:

* **Identity phase locking** (Laroche & Dolson). Advancing every bin's phase
  independently destroys the phase relationships *within* each harmonic's
  skirt. Instead only spectral peaks advance freely; the bins around a peak are
  rigidly re-pinned to it, preserving their original phase offsets.
* **Transient passthrough.** A vocoder spreads a drum hit across its whole
  window -- measured crest factor fell to 1-3% of the input. When a frame looks
  like an onset (a jump in spectral flux), the synthesis phases are reset to the
  analysis phases, which re-sharpens the attack at the cost of a momentary
  pitch inaccuracy nobody can hear on a transient.

Measuring a real recording found three larger faults than any of that, all now
fixed here:

* **A zero shift was not a no-op.** At ratio 1.0 the vocoder still resynthesised
  from accumulated phase, so a song needing no transposition came out mangled
  anyway. A zero shift now goes through a plain delay line of exactly the same
  latency, so it is sample-accurate, and the crossfade machinery treats that
  delay as just another core to hand over to.
* **The stereo image collapsed.** Shifting the channels independently let their
  phases drift apart: L/R correlation measured 0.654 going in and 0.038 coming
  out, which is the hollow, underwater sound. Phase is now derived once from the
  mid channel and each channel re-applies its own original offset from mid, so
  the image survives.
* **Level fell away as the shift grew**, by 2.3 dB at -5 semitones and 7.4 dB at
  +6, evenly across the spectrum. (First measured as "the bass lost 5.2 dB",
  which was a fixed measurement band catching content that had legitimately moved
  up in pitch -- the real fault was broadband.) Two causes: rounding each source
  bin to the nearest destination bin leaves gaps when shifting up, and splitting
  a magnitude between two bins conserves magnitude but not energy. Magnitudes are
  now interpolated across both neighbouring bins, and each frame is rescaled to
  the energy of the source bins that contributed.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def semitones_to_ratio(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


class _DelayCore:
    """A plain delay matching the vocoder's latency, for a shift of zero.

    Lets the shifter be bit-exact when there is nothing to do, while keeping the
    pipeline's latency identical whether or not a shift is applied -- so a change
    of shift never moves the audio in time.
    """

    def __init__(self, channels: int, delay: int = 2048, overlap: int = 4, ratio: float = 1.0):
        self.channels = channels
        # Must equal the active engine's latency, or changing shift would move
        # the audio in time.
        self.n_fft = delay
        self.ratio = 1.0
        self.transients = 0
        self._buf = np.zeros((channels, delay), dtype=np.float32)

    @property
    def latency_samples(self) -> int:
        return self.n_fft

    def set_ratio(self, ratio: float) -> None:
        pass  # a delay has no ratio; a non-zero shift swaps in a vocoder instead

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(block, dtype=np.float32)
        frames = block.shape[1]
        if frames >= self.n_fft:
            out = np.concatenate([self._buf, block[:, : frames - self.n_fft]], axis=1)
            self._buf = block[:, frames - self.n_fft :].copy()
            return out
        out = self._buf[:, :frames].copy()
        self._buf = np.concatenate([self._buf[:, frames:], block], axis=1)
        return out


class _PhaseVocoderCore:
    """One pitch-shifting engine at a fixed ratio, fed arbitrary block sizes."""

    def __init__(
        self,
        channels: int,
        n_fft: int = 2048,
        overlap: int = 4,
        ratio: float = 1.0,
        phase_lock: bool = True,
        transient_reset: bool = True,
        transient_threshold: float = 0.35,
    ):
        if n_fft % overlap:
            raise ValueError("n_fft must be divisible by overlap")
        self.phase_lock = phase_lock
        self.transient_reset = transient_reset
        self.transient_threshold = transient_threshold
        self.transients = 0
        self.channels = channels
        self.n_fft = n_fft
        self.hop = n_fft // overlap
        self.ratio = float(ratio)
        self.n_bins = n_fft // 2 + 1

        # Periodic Hann, applied on both analysis and synthesis.
        self.window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        self._norm = self._ola_gain()

        k = np.arange(self.n_bins, dtype=np.float64)
        self._expected = TWO_PI * self.hop * k / n_fft
        self._bin_of = k

        self._pending = np.zeros((channels, 0), dtype=np.float32)
        self._olap = np.zeros((channels, n_fft), dtype=np.float32)
        self._ready = np.zeros((channels, 0), dtype=np.float32)
        # Phase state is shared across channels: one set of decisions, applied
        # to both, is what keeps the stereo image intact.
        self._prev_phase = np.zeros(self.n_bins, dtype=np.float64)
        self._sum_phase = np.zeros(self.n_bins, dtype=np.float64)
        self._prev_mag = np.zeros(self.n_bins, dtype=np.float64)
        self._bin_int = np.arange(self.n_bins, dtype=np.int64)

    def _ola_gain(self) -> float:
        """Steady-state sum of the squared window at this hop."""
        acc = np.zeros(self.n_fft * 4, dtype=np.float64)
        w2 = self.window.astype(np.float64) ** 2
        for start in range(0, self.n_fft * 3, self.hop):
            acc[start : start + self.n_fft] += w2
        mid = acc[self.n_fft : self.n_fft * 2]
        return float(mid.mean())

    @property
    def latency_samples(self) -> int:
        return self.n_fft

    def set_ratio(self, ratio: float) -> None:
        self.ratio = float(ratio)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Shift a (channels, frames) float32 block; returns the same shape."""
        block = np.asarray(block, dtype=np.float32)
        frames = block.shape[1]
        self._pending = np.concatenate([self._pending, block], axis=1)

        while self._pending.shape[1] >= self.n_fft:
            frame = self._pending[:, : self.n_fft]
            self._process_frame(frame)
            # The oldest `hop` samples have now received every overlapping
            # contribution they will get, so they are final.
            done = self._olap[:, : self.hop] / self._norm
            self._ready = np.concatenate([self._ready, done], axis=1)
            self._olap = np.concatenate(
                [self._olap[:, self.hop :], np.zeros((self.channels, self.hop), dtype=np.float32)],
                axis=1,
            )
            self._pending = self._pending[:, self.hop :]

        if self._ready.shape[1] >= frames:
            out = self._ready[:, :frames]
            self._ready = self._ready[:, frames:]
            return np.ascontiguousarray(out)

        # Only happens while the first frame fills; pad so timing stays exact.
        have = self._ready.shape[1]
        out = np.zeros((self.channels, frames), dtype=np.float32)
        out[:, frames - have :] = self._ready
        self._ready = self._ready[:, have:]
        return out

    def _process_frame(self, frame: np.ndarray) -> None:
        spec = np.fft.rfft(frame * self.window, axis=1)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # All phase decisions come from the mid (channel sum). Deriving them per
        # channel lets the channels drift apart and destroys the stereo image.
        mid = spec.sum(axis=0)
        mid_mag = np.abs(mid)
        mid_phase = np.angle(mid)

        delta = mid_phase - self._prev_phase - self._expected
        delta = delta - TWO_PI * np.round(delta / TWO_PI)
        self._prev_phase = mid_phase
        true_bin = self._bin_of + delta * self.n_fft / (TWO_PI * self.hop)

        # Onset detection on the mid channel: a jump in positive spectral flux.
        onset = False
        if self.transient_reset:
            flux = np.maximum(mid_mag - self._prev_mag, 0.0).sum()
            prev_energy = self._prev_mag.sum()
            if prev_energy > 1e-9 and flux / prev_energy > self.transient_threshold:
                onset = True
                self.transients += 1
        self._prev_mag = mid_mag

        owner = self._peak_owners(mid_mag) if self.phase_lock else self._bin_int

        omega = TWO_PI * self.hop * (true_bin * self.ratio) / self.n_fft
        if onset:
            self._sum_phase = mid_phase.copy()
        else:
            self._sum_phase += omega
        # Identity locking: a peak advances freely, its neighbours keep the phase
        # offset they had relative to it.
        shared = self._sum_phase[owner] + (mid_phase - mid_phase[owner])

        # Split each source bin between the two neighbouring destination bins, so
        # shifting up does not leave gaps that silently drop energy.
        src = self._bin_int
        exact = src * self.ratio
        lo = np.floor(exact).astype(np.int64)
        frac = exact - lo
        hi = lo + 1
        ok_lo = lo < self.n_bins
        ok_hi = hi < self.n_bins

        # Only source bins whose destination is below Nyquist survive; their
        # energy is the reference for the normalisation below.
        contributes = ok_lo | ok_hi

        syn_mag = np.zeros_like(mag)
        out_phase = np.zeros_like(phase)
        for ch in range(self.channels):
            m = mag[ch]
            np.add.at(syn_mag[ch], lo[ok_lo], (m * (1.0 - frac))[ok_lo])
            np.add.at(syn_mag[ch], hi[ok_hi], (m * frac)[ok_hi])

            # Splitting a bin's magnitude across two bins conserves magnitude but
            # not energy, which cost up to 7 dB at larger shifts. Rescale each
            # frame to the energy of the bins that actually contributed.
            ref = float(np.dot(m[contributes], m[contributes]))
            got = float(np.dot(syn_mag[ch], syn_mag[ch]))
            if got > 1e-20 and ref > 1e-20:
                syn_mag[ch] *= np.sqrt(ref / got)

            # Each channel keeps its own phase offset from mid, which is what
            # carries the stereo image.
            offset = phase[ch] - mid_phase
            ch_phase = shared + offset
            order = np.argsort(m, kind="stable")
            dst_lo, dst_hi = lo[order], hi[order]
            valid_lo, valid_hi = dst_lo < self.n_bins, dst_hi < self.n_bins
            out_phase[ch, dst_lo[valid_lo]] = ch_phase[order][valid_lo]
            out_phase[ch, dst_hi[valid_hi]] = ch_phase[order][valid_hi]

        out_spec = syn_mag * np.exp(1j * out_phase)
        grain = np.fft.irfft(out_spec, n=self.n_fft, axis=1).astype(np.float32)
        self._olap += grain * self.window

    def _peak_owners(self, mag: np.ndarray) -> np.ndarray:
        """For each bin, the index of the spectral peak it belongs to."""
        n = mag.size
        is_peak = np.zeros(n, dtype=bool)
        if n > 2:
            is_peak[1:-1] = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
        floor = mag.max() * 1e-4
        is_peak &= mag > floor
        peaks = np.flatnonzero(is_peak)
        if peaks.size == 0:
            return np.arange(n)
        # Nearest peak for every bin, vectorised.
        idx = np.searchsorted(peaks, np.arange(n))
        left = peaks[np.clip(idx - 1, 0, peaks.size - 1)]
        right = peaks[np.clip(idx, 0, peaks.size - 1)]
        bins = np.arange(n)
        return np.where(np.abs(bins - left) <= np.abs(bins - right), left, right)


class PitchShifter:
    """Pitch shifter whose shift can change mid-stream without a click.

    Two engines run in parallel. On a change the standby is retargeted, given a
    few blocks to settle, and then faded in with an equal-power crossfade. Both
    engines always run, so latency never depends on the shift and a change never
    shifts the audio in time.
    """

    def __init__(
        self,
        samplerate: int,
        channels: int = 2,
        *,
        n_fft: int = 2048,
        overlap: int = 4,
        crossfade_seconds: float = 0.12,
        arm_blocks: int = 3,
        phase_lock: bool = True,
        transient_reset: bool = True,
        level_match: bool = True,
        level_tau: float = 0.25,
        engine: str = "auto",
        diag=None,
    ):
        self._diag = diag
        self._core_kw = dict(phase_lock=phase_lock, transient_reset=transient_reset)
        self.engine = self._choose_engine(engine)
        self.handovers = 0
        self.samplerate = samplerate
        self.channels = channels
        self.crossfade = crossfade_seconds > 0
        self.crossfade_samples = max(1, int(crossfade_seconds * samplerate))
        self.arm_blocks = arm_blocks

        self.n_fft = n_fft
        self.overlap = overlap
        # The bypass delay has to match whichever engine is in use, so measure it
        # once from a throwaway instance.
        self._engine_latency = self._measure_latency()
        # A shift of zero is a delay, not a vocoder pass.
        self._active = self._make_core(0)
        # Without a crossfade there is nothing to hand over to, and a measurement
        # showed an abrupt ratio change on this engine is already click-free, so
        # the second engine is pure cost. Skip it and halve the CPU.
        self._standby = self._make_core(0) if self.crossfade else None

        self._semitones = 0
        self._pending: int | None = None
        self._queued: int | None = None
        self._arm_left = 0
        self._fade_pos: int | None = None

        # Overlap-add of grains whose phases have been altered partly cancels, so
        # the output comes out quieter the larger the shift -- about 3 dB down at
        # -5 semitones and 6 dB at +6, and the amount depends on the material, so
        # no fixed correction fits. Instead the input and output levels are
        # tracked with a slow one-pole filter and the difference is corrected.
        # Slow enough (250 ms) not to pump on transients or squash dynamics.
        self.level_match = level_match
        self._level_tau = level_tau
        self._in_level = 0.0
        self._out_level = 0.0
        self._gain = 1.0

    @staticmethod
    def _choose_engine(requested: str) -> str:
        if requested not in ("auto", "signalsmith", "vocoder"):
            raise ValueError("engine must be 'auto', 'signalsmith' or 'vocoder'")
        if requested == "vocoder":
            return "vocoder"
        from . import signalsmith

        if signalsmith.available():
            return "signalsmith"
        if requested == "signalsmith":
            raise RuntimeError(
                "signalsmith engine requested but unavailable: %s"
                % signalsmith.unavailable_reason()
            )
        return "vocoder"

    def _measure_latency(self) -> int:
        """Latency of the shifting engine, which the bypass delay must match."""
        if self.engine == "signalsmith":
            from .signalsmith import SignalsmithCore

            probe = SignalsmithCore(self.channels, samplerate=self.samplerate)
            return int(probe.latency_samples)
        return self.n_fft

    def _make_core(self, semitones: int):
        """A delay for no shift, the chosen engine otherwise."""
        if semitones == 0:
            return _DelayCore(self.channels, self._engine_latency, self.overlap, 1.0)
        if self.engine == "signalsmith":
            from .signalsmith import SignalsmithCore

            return SignalsmithCore(
                self.channels,
                ratio=semitones_to_ratio(semitones),
                samplerate=self.samplerate,
            )
        return _PhaseVocoderCore(
            self.channels,
            self.n_fft,
            self.overlap,
            semitones_to_ratio(semitones),
            **self._core_kw,
        )

    @property
    def latency_seconds(self) -> float:
        return self._engine_latency / self.samplerate

    @property
    def semitones(self) -> int:
        """The shift being heard, or the one being faded in."""
        if self._queued is not None:
            return self._queued
        return self._semitones if self._pending is None else self._pending

    @property
    def fading(self) -> bool:
        return self._fade_pos is not None

    def set_semitones(self, semitones: int) -> None:
        semitones = int(semitones)
        if semitones == self.semitones:
            return
        if self._diag is not None:
            self._diag.event(4)  # EV_SHIFT_CHANGE
        if self._standby is None:
            self._semitones = semitones
            if isinstance(self._active, _DelayCore) != (semitones == 0):
                self._active = self._make_core(semitones)
            else:
                self._active.set_ratio(semitones_to_ratio(semitones))
            return
        if self._pending is not None:
            # Already handing over; apply this one once that finishes.
            self._queued = semitones
            return
        self._pending = semitones
        # Swap the standby for the right kind of core, then let it settle before
        # fading in: a fresh vocoder emits nothing for its first n_fft samples.
        if isinstance(self._standby, _DelayCore) != (semitones == 0):
            self._standby = self._make_core(semitones)
        else:
            self._standby.set_ratio(semitones_to_ratio(semitones))
        self._arm_left = self.arm_blocks

    def _match_level(self, dry: np.ndarray, wet: np.ndarray) -> np.ndarray:
        """Scale `wet` so its level tracks `dry`, smoothly.

        Skipped entirely while bypassing: a delay already has exactly the input's
        level, and the filter's start-up transient would otherwise scale a path
        that is meant to be sample-exact.
        """
        if not self.level_match or isinstance(self._active, _DelayCore):
            return wet
        frames = dry.shape[1]
        # One-pole smoothing over this block's worth of time.
        a = float(np.exp(-frames / (self._level_tau * self.samplerate)))
        dry_rms = float(np.sqrt(np.mean(np.square(dry, dtype=np.float64)))) if dry.size else 0.0
        wet_rms = float(np.sqrt(np.mean(np.square(wet, dtype=np.float64)))) if wet.size else 0.0
        self._in_level = a * self._in_level + (1 - a) * dry_rms
        self._out_level = a * self._out_level + (1 - a) * wet_rms
        # Only correct once there is a signal worth measuring.
        if self._in_level > 1e-4 and self._out_level > 1e-5:
            target = self._in_level / self._out_level
            self._gain = a * self._gain + (1 - a) * float(np.clip(target, 0.25, 4.0))
        return wet * np.float32(self._gain)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        frames = block.shape[1]

        wet_active = self._active.process(block)
        if self._standby is None:
            return self._match_level(block, wet_active)
        wet_standby = self._standby.process(block)

        if self._fade_pos is None:
            if self._pending is None:
                return self._match_level(block, wet_active)
            if self._arm_left > 0:
                self._arm_left -= 1  # standby still settling, not audible yet
                return self._match_level(block, wet_active)
            self._fade_pos = 0
            self.handovers += 1
            if self._diag is not None:
                self._diag.event(7)  # EV_XFADE_START

        start = self._fade_pos
        x = np.clip(
            (np.arange(start, start + frames, dtype=np.float32) / self.crossfade_samples), 0.0, 1.0
        )
        # Equal power: the two shifted copies are decorrelated, so sin/cos holds
        # loudness steady where a linear fade would dip.
        out = wet_active * np.cos(x * (np.pi / 2.0)) + wet_standby * np.sin(x * (np.pi / 2.0))

        self._fade_pos = start + frames
        if self._fade_pos >= self.crossfade_samples:
            self._active, self._standby = self._standby, self._active
            self._semitones = self._pending if self._pending is not None else self._semitones
            self._pending = None
            self._fade_pos = None
            if isinstance(self._standby, _DelayCore) != (self._semitones == 0):
                self._standby = self._make_core(self._semitones)
            else:
                self._standby.set_ratio(semitones_to_ratio(self._semitones))
            if self._queued is not None:
                nxt, self._queued = self._queued, None
                self.set_semitones(nxt)

        return np.ascontiguousarray(self._match_level(block, out), dtype=np.float32)
