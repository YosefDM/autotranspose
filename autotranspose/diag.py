"""Diagnostics for choppy, distorted or uneven audio.

The hard rule here: **nothing on the audio path formats a string, allocates, or
touches a file.** Logging from inside the capture loop or the output callback
would itself cause the glitches we are trying to measure. So the hot paths only
write numbers into preallocated arrays and bump integer counters; a reporter
thread does every bit of formatting and I/O.

What each symptom looks like in the log:

* **choppy / clicking** -- `underrun` or `overrun` events, or `ring` dipping to
  0. Underruns mean playback ran dry (capture or processing was late); overruns
  mean the opposite, that capture outran playback and a block was dropped.
* **a click every few seconds, steady** -- clock drift between two devices.
  Watch `drift` in the summary: capture and playback frame counts diverging
  linearly. Expected across two physical devices; a bigger ring trades latency
  for fewer drops.
* **crackly, distorted at peaks** -- `clip` events and `out_peak` at or above
  1.0. The phase vocoder can overshoot on transients; `--output-gain 0.8` fixes
  it without touching anything else.
* **gritty / metallic all the time** -- not a dropout; that is the shifter.
  Check `shift` magnitude (6 semitones sounds much worse than 2) and try a
  different `--fft`.
* **brief stumble whenever the key changes** -- `xfade` events next to
  `shift_change`.
* **periodic hitch, ~1-2 s apart** -- `analysis_ms` spikes or `gc_pause`
  events; the analyser or the garbage collector stalling the audio thread.
"""
from __future__ import annotations

import gc
import logging
import logging.handlers
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER_NAME = "autotranspose"

# Discrete events. Integer codes so the hot path never builds a string.
EV_UNDERRUN = 1
EV_OVERRUN = 2
EV_CAPTURE_GAP = 3
EV_SHIFT_CHANGE = 4
EV_CLIP = 5
EV_ANALYSIS_DROP = 6
EV_XFADE_START = 7
EV_GC_PAUSE = 8
EV_SLOW_BLOCK = 9
EV_CALLBACK_LATE = 10
EV_SILENCE = 11

EVENT_NAMES = {
    EV_UNDERRUN: "underrun",
    EV_OVERRUN: "overrun",
    EV_CAPTURE_GAP: "capture_gap",
    EV_SHIFT_CHANGE: "shift_change",
    EV_CLIP: "clip",
    EV_ANALYSIS_DROP: "analysis_drop",
    EV_XFADE_START: "xfade",
    EV_GC_PAUSE: "gc_pause",
    EV_SLOW_BLOCK: "slow_block",
    EV_CALLBACK_LATE: "callback_late",
    EV_SILENCE: "silence",
}

# Per-block fields recorded by the capture/process thread.
BLOCK_FIELDS = (
    "t",  # monotonic timestamp
    "interval",  # seconds since the previous block (capture jitter)
    "wait",  # time blocked in record()
    "process",  # time spent shifting
    "total",  # whole iteration
    "ring",  # ring fill in frames, after writing
    "in_peak",
    "out_peak",
    "in_rms",
    "out_rms",
    "clipped",  # samples at or beyond full scale
    "shift",
)

# Per-callback fields recorded by the PortAudio output callback.
CB_FIELDS = ("t", "interval", "frames", "served", "status", "dur")


class _Series:
    """Fixed-size circular store of float rows. Written from one thread."""

    def __init__(self, fields: tuple[str, ...], capacity: int):
        self.fields = fields
        self.capacity = capacity
        self._buf = np.zeros((len(fields), capacity), dtype=np.float64)
        self._i = 0
        self._count = 0

    def record(self, values) -> None:
        i = self._i
        self._buf[:, i] = values
        self._i = (i + 1) % self.capacity
        if self._count < self.capacity:
            self._count += 1

    def snapshot(self) -> dict[str, np.ndarray]:
        n = self._count
        if n == 0:
            return {f: np.zeros(0) for f in self.fields}
        if n < self.capacity:
            data = self._buf[:, :n].copy()
        else:
            data = np.concatenate(
                [self._buf[:, self._i :], self._buf[:, : self._i]], axis=1
            )
        return {f: data[k] for k, f in enumerate(self.fields)}


@dataclass
class Thresholds:
    slow_block_frac: float = 0.5  # process time over this fraction of a block
    clip_level: float = 0.999
    gc_pause_ms: float = 5.0
    callback_late_frac: float = 1.5  # callback interval over this x nominal


class Diagnostics:
    """Collects metrics and writes periodic summaries to a log file."""

    def __init__(
        self,
        *,
        samplerate: int,
        blocksize: int,
        log_path: Path | None = None,
        level: str = "INFO",
        console: bool = False,
        summary_seconds: float = 5.0,
        capacity: int = 8192,
        csv_path: Path | None = None,
        thresholds: Thresholds | None = None,
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.block_seconds = blocksize / samplerate
        self.summary_seconds = summary_seconds
        self.thresholds = thresholds or Thresholds()
        self.csv_path = csv_path

        self.blocks = _Series(BLOCK_FIELDS, capacity)
        self.callbacks = _Series(CB_FIELDS, capacity)

        self._events = np.zeros((2, 4096), dtype=np.float64)  # (code, t)
        self._ev_i = 0
        self._ev_count = 0
        self.event_counts = np.zeros(max(EVENT_NAMES) + 1, dtype=np.int64)

        # Running frame tallies, for drift between the two device clocks.
        self.frames_captured = 0
        self.frames_served = 0
        # Silence emitted on purpose while the ring builds its cushion. Counting
        # it as a shortfall would report priming as a dropout.
        self.priming_frames = 0
        # Drift must be a slope, not a total: the output stream starts serving
        # before capture delivers its first block, and that fixed startup skew
        # would otherwise be reported as an enormous clock error.
        self._drift_base: tuple[float, int] | None = None
        self.drift_warmup = 3.0

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gc_t0 = 0.0
        self._gc_hook = None
        self.started_at = 0.0

        self.log = logging.getLogger(LOGGER_NAME)
        self.log.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.log.handlers.clear()
        self.log.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-7s %(threadName)-14s %(message)s",
            datefmt="%H:%M:%S",
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=8 << 20, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            self.log.addHandler(fh)
        if console:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.log.addHandler(sh)
        if not self.log.handlers:
            self.log.addHandler(logging.NullHandler())
        self.log_path = log_path

    # -- hot path: numbers only ------------------------------------------
    def event(self, code: int, t: float | None = None) -> None:
        """Record a discrete event. Safe from the audio thread."""
        i = self._ev_i
        self._events[0, i] = code
        self._events[1, i] = t if t is not None else time.perf_counter()
        self._ev_i = (i + 1) % self._events.shape[1]
        if self._ev_count < self._events.shape[1]:
            self._ev_count += 1
        self.event_counts[code] += 1

    def record_block(self, values) -> None:
        self.blocks.record(values)

    def record_callback(self, values) -> None:
        self.callbacks.record(values)

    # -- lifecycle --------------------------------------------------------
    def start(self, header: dict | None = None) -> None:
        self.started_at = time.perf_counter()
        if header:
            self.log.info("=== session start ===")
            for k, v in header.items():
                self.log.info("  %-22s %s", k, v)
        self._install_gc_hook()
        self._thread = threading.Thread(target=self._report_loop, name="diag", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._remove_gc_hook()
        self.summarise(final=True)
        if self.csv_path:
            self._dump_csv()

    def _install_gc_hook(self) -> None:
        def hook(phase, info):
            if phase == "start":
                self._gc_t0 = time.perf_counter()
            else:
                ms = (time.perf_counter() - self._gc_t0) * 1000.0
                if ms >= self.thresholds.gc_pause_ms:
                    self.event(EV_GC_PAUSE)
                    self._gc_last_ms = ms

        self._gc_hook = hook
        self._gc_last_ms = 0.0
        gc.callbacks.append(hook)

    def _remove_gc_hook(self) -> None:
        if self._gc_hook is not None and self._gc_hook in gc.callbacks:
            gc.callbacks.remove(self._gc_hook)
        self._gc_hook = None

    # -- reporting --------------------------------------------------------
    def _report_loop(self) -> None:
        while not self._stop.wait(self.summary_seconds):
            try:
                self.summarise()
            except Exception:  # never let diagnostics kill the run
                self.log.exception("diagnostics summary failed")

    @staticmethod
    def _pct(a: np.ndarray, q) -> tuple:
        if a.size == 0:
            return tuple(0.0 for _ in q)
        return tuple(float(x) for x in np.percentile(a, q))

    def summarise(self, final: bool = False) -> dict:
        b = self.blocks.snapshot()
        c = self.callbacks.snapshot()
        elapsed = time.perf_counter() - self.started_at if self.started_at else 0.0
        nominal_ms = self.block_seconds * 1000.0

        stats: dict = {"elapsed_s": round(elapsed, 1)}

        if b["t"].size:
            iv = b["interval"][b["interval"] > 0] * 1000.0
            pr = b["process"] * 1000.0
            p50, p95, p99, pmax = self._pct(pr, (50, 95, 99, 100))
            i50, i95, imax = self._pct(iv, (50, 95, 100))
            ring = b["ring"]
            stats.update(
                blocks=int(b["t"].size),
                process_ms=(round(p50, 2), round(p95, 2), round(p99, 2), round(pmax, 2)),
                load_pct=round(p95 / nominal_ms * 100, 1),
                interval_ms=(round(i50, 2), round(i95, 2), round(imax, 2)),
                jitter_ms=round(float(np.std(iv)), 2) if iv.size else 0.0,
                ring_frames=(int(ring.min()), int(np.median(ring)), int(ring.max())),
                ring_ms=round(float(np.median(ring)) / self.samplerate * 1000.0, 1),
                in_peak=round(float(b["in_peak"].max()), 4),
                out_peak=round(float(b["out_peak"].max()), 4),
                gain_db=self._gain_db(b),
                clipped=int(b["clipped"].sum()),
            )
        if c["t"].size:
            cb_iv = c["interval"][c["interval"] > 0] * 1000.0
            short = c["frames"] - c["served"]
            cd50, cd95, cdmax = self._pct(c["dur"] * 1000.0, (50, 95, 100))
            stats.update(
                callbacks=int(c["t"].size),
                cb_interval_ms=self._pct(cb_iv, (50, 95, 100)) if cb_iv.size else (0, 0, 0),
                cb_dur_ms=(round(cd50, 3), round(cd95, 3), round(cdmax, 3)),
                cb_shortfall_frames=int(short.sum()),
                priming_frames=int(self.priming_frames),
                dropout_frames=max(0, int(short.sum()) - int(self.priming_frames)),
                cb_status_flags=int(c["status"].sum()),
            )

        offset = self.frames_captured - self.frames_served
        stats["skew_frames"] = int(offset)
        stats["skew_ms"] = round(offset / self.samplerate * 1000.0, 1)
        now = time.perf_counter()
        if self._drift_base is None:
            if elapsed >= self.drift_warmup:
                self._drift_base = (now, offset)
        else:
            t0, off0 = self._drift_base
            dt = now - t0
            # Over a couple of seconds the figure is dominated by start-up slop
            # and reports thousands of ppm that mean nothing, so wait for a
            # window long enough for a real slope to show.
            if dt >= 15.0:
                drift = offset - off0
                stats["drift_frames"] = int(drift)
                stats["drift_ms"] = round(drift / self.samplerate * 1000.0, 1)
                stats["drift_ppm"] = round(drift / (self.samplerate * dt) * 1e6, 1)
                stats["drift_window_s"] = round(dt, 1)

        counts = {
            EVENT_NAMES[k]: int(v)
            for k, v in enumerate(self.event_counts)
            if v and k in EVENT_NAMES
        }
        stats["events"] = counts

        tag = "FINAL" if final else "summary"
        self.log.info(
            "%s t=%.1fs blocks=%s process_ms(p50/p95/p99/max)=%s load_p95=%s%% "
            "interval_ms(p50/p95/max)=%s jitter=%sms ring(min/med/max)=%s (%sms) "
            "in_peak=%s out_peak=%s gain=%sdB clipped=%s skew=%sms drift=%s frames "
            "(%sms, %s ppm over %ss) dropout_frames=%s (priming %s) events=%s",
            tag,
            stats.get("elapsed_s", 0),
            stats.get("blocks", 0),
            stats.get("process_ms", "-"),
            stats.get("load_pct", "-"),
            stats.get("interval_ms", "-"),
            stats.get("jitter_ms", "-"),
            stats.get("ring_frames", "-"),
            stats.get("ring_ms", "-"),
            stats.get("in_peak", "-"),
            stats.get("out_peak", "-"),
            stats.get("gain_db", "-"),
            stats.get("clipped", 0),
            stats.get("skew_ms", 0),
            stats.get("drift_frames", "-"),
            stats.get("drift_ms", "-"),
            stats.get("drift_ppm", "-"),
            stats.get("drift_window_s", "-"),
            stats.get("dropout_frames", 0),
            stats.get("priming_frames", 0),
            counts or "{}",
        )
        for line in self.verdict(stats):
            self.log.warning("  -> %s", line)
        return stats

    @staticmethod
    def _gain_db(b) -> float:
        in_rms = b["in_rms"]
        out_rms = b["out_rms"]
        mask = in_rms > 1e-5
        if not mask.any():
            return 0.0
        return round(float(20 * np.log10(np.median(out_rms[mask] / in_rms[mask]) + 1e-12)), 2)

    def verdict(self, stats: dict | None = None) -> list[str]:
        """Plain-language read on what is wrong, from the numbers collected."""
        s = stats if stats is not None else self.summarise()
        out: list[str] = []
        ev = s.get("events", {})
        elapsed = max(s.get("elapsed_s", 0.0), 1e-6)

        under = ev.get("underrun", 0)
        over = ev.get("overrun", 0)
        gaps = ev.get("capture_gap", 0)

        dropped_out = s.get("dropout_frames", 0)
        if under and dropped_out == 0:
            out.append(
                f"{under} underrun(s), but no audio was actually lost -- that is the "
                f"buffer priming at startup, which is silent by design."
            )
            under = 0
        if under:
            rate = under / elapsed
            out.append(
                f"{under} playback underruns ({rate:.2f}/s): audio ran dry, so you hear "
                f"gaps. Processing or capture is arriving late -- check load_p95 and "
                f"jitter below, and try a larger --blocksize."
            )
        if over:
            out.append(
                f"{over} overruns: capture outran playback and blocks were dropped. "
                f"Usually clock drift between two devices (see drift_ppm)."
            )
        if gaps:
            out.append(
                f"{gaps} capture gaps: the captured device under-ran, often because the "
                f"source app paused. Harmless unless it keeps climbing."
            )
        if s.get("load_pct", 0) > 70:
            out.append(
                f"load_p95 is {s['load_pct']}% of the block budget: too close to the "
                f"limit. Use --no-crossfade, a larger --fft, or --mono."
            )
        if s.get("clipped", 0):
            out.append(
                f"{s['clipped']} clipped samples (out_peak {s.get('out_peak')}): the "
                f"shifter overshot full scale and the result is crackly. "
                f"Use --output-gain 0.8."
            )
        drift_ppm = s.get("drift_ppm")
        if drift_ppm is not None and abs(drift_ppm) > 500:
            out.append(
                f"clock drift {drift_ppm} ppm between capture and playback (measured over "
                f"{s.get('drift_window_s')}s): expect a click every so often. Inherent to "
                f"two devices; a larger --ring-blocks trades latency for fewer clicks."
            )
        iv = s.get("interval_ms")
        ring_ms = s.get("ring_ms")
        if iv and ring_ms:
            jitter_max = iv[-1]
            if jitter_max > ring_ms:
                out.append(
                    f"capture jitter reaches {jitter_max:.0f} ms but the ring only holds "
                    f"{ring_ms:.0f} ms of audio, so a late block leaves playback with "
                    f"nothing. Raise --ring-blocks (or --blocksize)."
                )
        if ev.get("gc_pause", 0) > 2:
            out.append(
                f"{ev['gc_pause']} long garbage-collection pauses: these stall the "
                f"audio thread. Try --gc-freeze."
            )
        if ev.get("slow_block", 0):
            out.append(
                f"{ev['slow_block']} blocks took over "
                f"{self.thresholds.slow_block_frac:.0%} of their budget to process."
            )
        if not out:
            out.append("nothing wrong in the numbers: no dropouts, no clipping, load fine.")
        return out

    def recent_events(self, limit: int = 40) -> list[tuple[str, float]]:
        n = self._ev_count
        if n == 0:
            return []
        cap = self._events.shape[1]
        if n < cap:
            codes, times = self._events[0, :n], self._events[1, :n]
        else:
            codes = np.concatenate([self._events[0, self._ev_i :], self._events[0, : self._ev_i]])
            times = np.concatenate([self._events[1, self._ev_i :], self._events[1, : self._ev_i]])
        out = []
        for code, t in zip(codes[-limit:], times[-limit:]):
            out.append((EVENT_NAMES.get(int(code), str(int(code))), t - self.started_at))
        return out

    def _dump_csv(self) -> None:
        try:
            b = self.blocks.snapshot()
            if not b["t"].size:
                return
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            arr = np.vstack([b[f] for f in BLOCK_FIELDS]).T
            arr[:, 0] -= self.started_at
            np.savetxt(
                self.csv_path,
                arr,
                delimiter=",",
                header=",".join(BLOCK_FIELDS),
                comments="",
                fmt="%.6f",
            )
            self.log.info("per-block metrics written to %s (%d rows)", self.csv_path, len(arr))
        except Exception:
            self.log.exception("could not write the metrics CSV")


def default_log_path(root: Path | None = None) -> Path:
    base = root or Path(
        os.environ.get("LOCALAPPDATA") or Path.home()
    ) / "autotranspose" / "logs"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(base) / f"session-{stamp}.log"
