"""The live engine: capture -> analyse -> decide -> shift -> play.

Three threads:

* **audio** -- the only latency-critical one. Records a block from the WASAPI
  loopback, pitch-shifts it, plays it. It never does analysis and never blocks
  on the analyser; it hands audio over through a queue it is allowed to drop.
* **analysis** -- resamples to 22.05 kHz, extracts chroma, estimates the key and
  asks the decider for a shift. Tens of milliseconds of work every 1.5 s, so it
  can afford to be slow without ever touching the audio path.
* the caller's thread, which reads `status` to draw the UI.
"""
from __future__ import annotations

import gc
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import soxr
from pathlib import Path

from . import devices
from .decide import DecisionState, ShiftDecider
from .diag import BLOCK_FIELDS, CB_FIELDS
from .keydetect import ANALYSIS_SR, IncrementalChroma, KeyEstimator
from .ring import RingBuffer
from .shifter import PitchShifter

SILENCE_DBFS = -58.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _dbfs_from_rms(rms: float) -> float:
    return 20.0 * np.log10(max(rms, 1e-5))  # floor at -100 dB so meters stay readable


def _dbfs(x: np.ndarray) -> float:
    return _dbfs_from_rms(_rms(x))


@dataclass
class EngineStatus:
    running: bool = False
    input_dbfs: float = -100.0
    output_dbfs: float = -100.0
    applied_shift: int = 0
    decision: DecisionState = field(default_factory=DecisionState)
    audio_load: float = 0.0  # fraction of real time spent processing
    analysis_ms: float = 0.0
    dropped_analysis_blocks: int = 0
    underruns: int = 0
    overruns: int = 0
    ring_frames: int = 0
    discontinuities: int = 0
    clipped_samples: int = 0
    dropped_frames: int = 0
    starved_frames: int = 0
    handovers: int = 0
    peak_in: float = 0.0
    peak_out: float = 0.0
    error: str | None = None
    started_at: float = 0.0
    shifter_latency_ms: float = 0.0
    block_latency_ms: float = 0.0


@dataclass
class EngineConfig:
    samplerate: int = 48000
    blocksize: int = 1024
    channels: int = 2
    n_fft: int = 2048
    crossfade: bool = True
    analysis_window: float = 30.0
    cycle_seconds: float = 1.5
    profile: str = "blend"
    # "white" maximises the share of the music landing on white keys; "template"
    # is the older key-profile correlation. White is both more accurate on the
    # synthetic set and far more stable on real songs.
    objective: str = "white"
    min_gain: float = 0.05
    max_shift: int = 6
    prefer_down: bool = True
    min_heard_seconds: float = 12.0
    min_dwell_seconds: float = 25.0
    commit_share: float = 0.55
    mono: bool = False
    bypass: bool = False
    # "auto" uses Signalsmith Stretch when its DLL is built, else the numpy
    # vocoder. Measured better on every axis, and ten times cheaper.
    shift_engine: str = "auto"
    output_gain: float = 1.0
    gc_freeze: bool = False
    record_dir: str | None = None
    # Measured against real capture jitter on this machine: 4 blocks lost 4096
    # frames of audio, 8 lost 1024, 12 and 16 lost none. 12 is the cheapest depth
    # that dropped nothing, so it is the default; lower it for less latency.
    ring_blocks: int = 12


class Engine:
    def __init__(
        self,
        capture: devices.Device,
        output: devices.Device | None,
        config: EngineConfig | None = None,
        diagnostics=None,
    ):
        self.capture = capture
        self.output = output
        self.cfg = config or EngineConfig()
        if diagnostics is None:
            from .diag import Diagnostics

            diagnostics = Diagnostics(
                samplerate=self.cfg.samplerate, blocksize=self.cfg.blocksize
            )
        self.diag = diagnostics
        self.log = self.diag.log
        self.recorder = None
        if self.cfg.record_dir:
            from .record import WavRecorder

            self.recorder = WavRecorder(
                Path(self.cfg.record_dir),
                self.cfg.samplerate,
                self.cfg.channels,
                log=self.log,
            )
        self.status = EngineStatus(
            shifter_latency_ms=0.0,
            block_latency_ms=self.cfg.blocksize / self.cfg.samplerate * 1000.0,
        )

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # Must comfortably exceed one analysis cycle, or the analyser can never
        # accumulate a full cycle without the audio thread overflowing it first.
        blocks_per_cycle = self.cfg.cycle_seconds * self.cfg.samplerate / self.cfg.blocksize
        self._analysis_q: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(16, int(blocks_per_cycle * 4))
        )
        self._lock = threading.Lock()
        # Preallocated rows so the audio path never allocates to record metrics.
        self._blk = np.zeros(len(BLOCK_FIELDS), dtype=np.float64)
        self._cb_buf = np.zeros(len(CB_FIELDS), dtype=np.float64)
        self._cb_last_t = 0.0
        self._cb_late_threshold = (
            self.cfg.blocksize / self.cfg.samplerate
        ) * self.diag.thresholds.callback_late_frac
        # Capture and playback run on independent clocks (genuinely so across two
        # devices, e.g. a Bluetooth speaker), so the fill level drifts. The ring
        # has to be deep enough to cover capture jitter -- see ring_blocks.
        self._ring = RingBuffer(
            self.cfg.blocksize * self.cfg.ring_blocks,
            self.cfg.channels,
            prefill_frames=self.cfg.blocksize * max(1, self.cfg.ring_blocks // 2),
            diag=self.diag,
        )

        self.shifter = PitchShifter(
            self.cfg.samplerate,
            self.cfg.channels,
            n_fft=self.cfg.n_fft,
            crossfade_seconds=0.12 if self.cfg.crossfade else 0.0,
            engine=self.cfg.shift_engine,
            diag=self.diag,
        )
        self.decider = ShiftDecider(
            mode="gain" if self.cfg.objective == "white" else "votes",
            min_gain=self.cfg.min_gain,
            cycle_seconds=self.cfg.cycle_seconds,
            max_shift=self.cfg.max_shift,
            prefer_down=self.cfg.prefer_down,
            min_heard_seconds=self.cfg.min_heard_seconds,
            min_dwell_seconds=self.cfg.min_dwell_seconds,
            commit_share=self.cfg.commit_share,
        )
        if self.cfg.bypass:
            self.decider.set_manual(0)
        self.status.shifter_latency_ms = self.shifter.latency_seconds * 1000.0

    # -- lifecycle ----------------------------------------------------------
    def _install_warning_counter(self) -> None:
        """Count soundcard's loopback discontinuity warnings instead of printing them.

        They fire whenever the captured device under-runs -- common and harmless
        when the source app pauses -- but printed warnings would scribble all
        over the live display. Counting them keeps the information without the
        mess.
        """
        self._prev_showwarning = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if "discontinuity" in str(message):
                self.status.discontinuities += 1
                self.diag.event(3)  # EV_CAPTURE_GAP
                return
            self._prev_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = showwarning

    def _restore_warnings(self) -> None:
        if getattr(self, "_prev_showwarning", None) is not None:
            warnings.showwarning = self._prev_showwarning
            self._prev_showwarning = None

    def start(self) -> None:
        self._install_warning_counter()
        if self.cfg.gc_freeze:
            # Move everything alive now into the permanent generation so routine
            # collections have far less to walk; long GC pauses stall the audio
            # thread and show up as dropouts.
            gc.collect()
            gc.freeze()
            gc.disable()
            self.log.info("gc frozen and disabled for this session")
        self.diag.start(self._session_header())
        if self.recorder is not None:
            self.recorder.start()
            self.log.info("recording input/output WAVs to %s", self.cfg.record_dir)
        self.status.running = True
        self.status.started_at = time.monotonic()
        targets = [self._analysis_loop]
        if self.output is not None:
            targets.append(self._audio_loop)
        else:
            targets.append(self._capture_only_loop)
        for fn in targets:
            t = threading.Thread(target=fn, name=fn.__name__, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self.status.running = False
        self._restore_warnings()
        if self.recorder is not None:
            self.recorder.stop()
        try:
            self.diag.stop()
        finally:
            if self.cfg.gc_freeze:
                gc.enable()
                gc.unfreeze()

    def _session_header(self) -> dict:
        cfg = self.cfg
        return {
            "capture": self.capture.name,
            "output": self.output.name if self.output else "(analyse only)",
            "samplerate": cfg.samplerate,
            "blocksize": f"{cfg.blocksize} frames ({cfg.blocksize / cfg.samplerate * 1000:.1f} ms)",
            "channels": cfg.channels,
            "shift_engine": self.shifter.engine,
            "shift_latency": f"{self.shifter.latency_seconds * 1000:.1f} ms",
            "n_fft": f"{cfg.n_fft} (fallback vocoder only)",
            "crossfade": cfg.crossfade,
            "mono": cfg.mono,
            "output_gain": cfg.output_gain,
            "ring_capacity": f"{self._ring.capacity} frames "
            f"({self._ring.capacity / cfg.samplerate * 1000:.1f} ms, "
            f"{cfg.ring_blocks} blocks)",
            "ring_prefill": f"{self._ring.prefill} frames "
            f"({self._ring.prefill / cfg.samplerate * 1000:.1f} ms cushion)",
            "analysis_window": f"{cfg.analysis_window}s",
            "cycle": f"{cfg.cycle_seconds}s",
            "profile": cfg.profile,
            "objective": cfg.objective,
            "min_gain": f"{cfg.min_gain:.3f} ({cfg.min_gain * 100:.1f} points of white-key mass)",
            "max_shift": cfg.max_shift,
            "gc_freeze": cfg.gc_freeze,
            "python": sys.version.split()[0],
        }

    def __enter__(self) -> "Engine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- manual control -----------------------------------------------------
    def nudge(self, delta: int) -> None:
        """Pin the shift and move it by `delta` semitones."""
        current = self.decider.state.applied_shift
        self.decider.set_manual(int(np.clip(current + delta, -12, 12)))

    def set_manual(self, semitones: int | None) -> None:
        self.decider.set_manual(semitones)

    def toggle_auto(self) -> None:
        self.decider.set_manual(None if self.decider.manual else self.decider.state.applied_shift)

    # -- threads ------------------------------------------------------------
    def _submit_for_analysis(self, block: np.ndarray) -> None:
        """block is (frames, channels) as soundcard delivers it."""
        mono = block.mean(axis=1).astype(np.float32, copy=False)
        try:
            self._analysis_q.put_nowait(mono)
        except queue.Full:
            self.status.dropped_analysis_blocks += 1
            self.diag.event(6)  # EV_ANALYSIS_DROP

    def _output_callback(self, outdata, frames, time_info, status) -> None:
        """PortAudio pulls from the ring here. Must stay trivial and never block.

        Only numeric stores happen here -- no formatting, no allocation, no I/O.
        """
        t0 = time.perf_counter()
        flags = 0.0
        if status:
            if getattr(status, "output_underflow", False):
                flags += 1.0
            if getattr(status, "output_overflow", False):
                flags += 2.0
            if getattr(status, "priming_output", False):
                flags += 4.0
        served = self._ring.read(frames, outdata)
        now = time.perf_counter()
        interval = now - self._cb_last_t if self._cb_last_t else 0.0
        self._cb_last_t = now
        self._cb_buf[0] = now
        self._cb_buf[1] = interval
        self._cb_buf[2] = frames
        self._cb_buf[3] = served
        self._cb_buf[4] = flags
        self._cb_buf[5] = now - t0
        self.diag.record_callback(self._cb_buf)
        self.diag.frames_served += frames
        if interval > self._cb_late_threshold:
            self.diag.event(10, now)  # EV_CALLBACK_LATE

    def _open_output_stream(self):
        """The playback stream. Overridden in tests to run without a sound card."""
        import sounddevice as sd

        index = devices.sd_output_index(self.output)
        info = sd.query_devices(index)
        self.log.info(
            "output stream: [%d] %r hostapi=%s max_ch=%d default_sr=%s "
            "default_low_latency=%.1fms",
            index,
            info["name"],
            sd.query_hostapis(info["hostapi"])["name"],
            info["max_output_channels"],
            info["default_samplerate"],
            info["default_low_output_latency"] * 1000,
        )
        stream = sd.OutputStream(
            device=index,
            samplerate=self.cfg.samplerate,
            channels=self.cfg.channels,
            blocksize=self.cfg.blocksize,
            dtype="float32",
            callback=self._output_callback,
        )
        self.log.info(
            "stream opened: sr=%s blocksize=%s latency=%.1fms",
            stream.samplerate, stream.blocksize, stream.latency * 1000,
        )
        return stream

    def _audio_loop(self) -> None:
        cfg = self.cfg
        block_seconds = cfg.blocksize / cfg.samplerate
        try:
            with devices.open_recorder(
                self.capture, cfg.samplerate, cfg.channels, cfg.blocksize
            ) as rec, self._open_output_stream():
                slow_threshold = block_seconds * self.diag.thresholds.slow_block_frac
                clip_level = self.diag.thresholds.clip_level
                gain = float(cfg.output_gain)
                last_t = 0.0
                while not self._stop.is_set():
                    t_wait0 = time.perf_counter()
                    data = rec.record(numframes=cfg.blocksize)
                    t0 = time.perf_counter()
                    wait = t0 - t_wait0
                    self._submit_for_analysis(data)

                    if cfg.mono:
                        m = data.mean(axis=1)
                        data = np.stack([m, m], axis=1)

                    shift = self.decider.state.applied_shift
                    self.shifter.set_semitones(shift)
                    out = self.shifter.process(np.ascontiguousarray(data.T))
                    if gain != 1.0:
                        out = out * gain
                    t_proc = time.perf_counter() - t0

                    in_peak = float(np.abs(data).max()) if data.size else 0.0
                    out_abs = np.abs(out)
                    out_peak = float(out_abs.max()) if out.size else 0.0
                    clipped = int((out_abs >= clip_level).sum())

                    self._ring.write(np.ascontiguousarray(out.T))
                    if self.recorder is not None:
                        self.recorder.submit(data, out.T)

                    now = time.perf_counter()
                    total = now - t_wait0
                    interval = now - last_t if last_t else 0.0
                    last_t = now

                    self._blk[0] = now
                    self._blk[1] = interval
                    self._blk[2] = wait
                    self._blk[3] = t_proc
                    self._blk[4] = total
                    self._blk[5] = self._ring.filled
                    self._blk[6] = in_peak
                    self._blk[7] = out_peak
                    self._blk[8] = _rms(data)
                    self._blk[9] = _rms(out)
                    self._blk[10] = clipped
                    self._blk[11] = shift
                    self.diag.record_block(self._blk)
                    self.diag.frames_captured += cfg.blocksize
                    if t_proc > slow_threshold:
                        self.diag.event(9, now)  # EV_SLOW_BLOCK
                    if clipped:
                        self.diag.event(5, now)  # EV_CLIP

                    with self._lock:
                        self.status.input_dbfs = _dbfs_from_rms(self._blk[8])
                        self.status.output_dbfs = _dbfs_from_rms(self._blk[9])
                        self.status.applied_shift = self.shifter.semitones
                        self.status.ring_frames = self._ring.filled
                        self.status.underruns = self._ring.underruns
                        self.status.overruns = self._ring.overruns
                        self.status.dropped_frames = self._ring.dropped_frames
                        self.status.starved_frames = self._ring.starved_frames
                        self.diag.priming_frames = self._ring.priming_frames
                        self.status.clipped_samples += clipped
                        self.status.handovers = self.shifter.handovers
                        self.status.peak_in = max(self.status.peak_in, in_peak)
                        self.status.peak_out = max(self.status.peak_out, out_peak)
                        # Only the work we do counts; the blocking record() call
                        # is the clock, not load.
                        self.status.audio_load = (
                            0.9 * self.status.audio_load + 0.1 * (t_proc / block_seconds)
                        )
        except Exception as exc:  # surfaced in the UI rather than killing it
            self.status.error = f"{type(exc).__name__}: {exc}"
            self.log.exception("audio thread died")
            self._stop.set()

    def _capture_only_loop(self) -> None:
        """detect mode: listen and analyse, play nothing."""
        cfg = self.cfg
        try:
            with devices.open_recorder(
                self.capture, cfg.samplerate, cfg.channels, cfg.blocksize
            ) as rec:
                last_t = 0.0
                while not self._stop.is_set():
                    t_wait0 = time.perf_counter()
                    data = rec.record(numframes=cfg.blocksize)
                    now = time.perf_counter()
                    self._submit_for_analysis(data)
                    self._blk[:] = 0.0
                    self._blk[0] = now
                    self._blk[1] = now - last_t if last_t else 0.0
                    self._blk[2] = now - t_wait0
                    self._blk[6] = float(np.abs(data).max()) if data.size else 0.0
                    self._blk[8] = _rms(data)
                    last_t = now
                    self.diag.record_block(self._blk)
                    self.diag.frames_captured += cfg.blocksize
                    with self._lock:
                        self.status.input_dbfs = _dbfs_from_rms(self._blk[8])
                        self.status.applied_shift = self.decider.state.applied_shift
        except Exception as exc:
            self.status.error = f"{type(exc).__name__}: {exc}"
            self.log.exception("capture thread died")
            self._stop.set()

    def _analysis_loop(self) -> None:
        cfg = self.cfg
        resampler = soxr.ResampleStream(
            cfg.samplerate, ANALYSIS_SR, 1, dtype="float32", quality="QQ"
        )
        chroma = IncrementalChroma(sr=ANALYSIS_SR)
        estimator = KeyEstimator(
            window_seconds=cfg.analysis_window,
            profile=cfg.profile,
            objective=cfg.objective,
        )

        pending: list[np.ndarray] = []
        pending_frames = 0
        need = int(cfg.cycle_seconds * cfg.samplerate)
        loud_frames = 0

        while not self._stop.is_set():
            try:
                mono = self._analysis_q.get(timeout=0.25)
            except queue.Empty:
                continue
            pending.append(mono)
            pending_frames += mono.size
            if _dbfs(mono) > SILENCE_DBFS:
                loud_frames += mono.size
            if pending_frames < need:
                continue

            chunk = np.concatenate(pending)
            silent = loud_frames < 0.15 * pending_frames
            pending, pending_frames, loud_frames = [], 0, 0

            t0 = time.perf_counter()
            try:
                low = resampler.resample_chunk(chunk)
                t_resample = time.perf_counter()
                frames_added = 0
                if low.size:
                    new = chroma.push(low)
                    frames_added = new.shape[1] if new.size else 0
                    estimator.add_frames(new)
                t_chroma = time.perf_counter()
                estimate = estimator.estimate() if not silent else None
                t_estimate = time.perf_counter()
                # Only the audio thread ever touches the shifter; it picks the
                # new value up from the decision state on its next block.
                before = self.decider.state.applied_shift
                shift = self.decider.update(
                    estimate, heard_seconds=estimator.filled_seconds, silent=silent
                )
                st = self.decider.state
                self.log.debug(
                    "analysis cycle: silent=%s resample=%.1fms chroma=%.1fms(+%d frames) "
                    "estimate=%.1fms heard=%.1fs status=%s key=%s shift=%+d "
                    "share=%.0f%% conf=%.0f%% margin=%.3f novelty=%.2f qdepth=%d "
                    "votes=%s scores=%s",
                    silent,
                    (t_resample - t0) * 1000,
                    (t_chroma - t_resample) * 1000,
                    frames_added,
                    (t_estimate - t_chroma) * 1000,
                    estimator.filled_seconds,
                    st.status,
                    st.key_name,
                    shift,
                    st.leader_share * 100,
                    st.key_confidence * 100,
                    st.margin,
                    estimate.novelty if estimate else -1.0,
                    self._analysis_q.qsize(),
                    # Full 12-way vote and raw correlation per shift class: the
                    # only way to see *why* a key was chosen, after the fact.
                    np.array2string(st.votes, precision=2, suppress_small=True),
                    np.array2string(estimate.shift_scores, precision=3)
                    if estimate is not None
                    else "-",
                )
                if shift != before:
                    self.log.info(
                        "SHIFT %+d -> %+d  (%s, heard %.1fs) white-keys: %.1f%% -> %.1f%% "
                        "(gain %.1f points)",
                        before, shift, st.key_name, estimator.filled_seconds,
                        st.white_now * 100, st.white_best * 100,
                        (st.white_best - st.white_now) * 100,
                    )
                if silent:
                    self.diag.event(11)  # EV_SILENCE
            except Exception as exc:
                self.status.error = f"analysis: {type(exc).__name__}: {exc}"
                self.log.exception("analysis cycle failed")
                continue

            with self._lock:
                self.status.analysis_ms = (time.perf_counter() - t0) * 1000.0
                self.status.decision = self.decider.state
