"""Capture input and output to WAV files, for judging quality off-line.

When the complaint is "it sounds bad" rather than "it drops out", numbers only
go so far -- you need to hear the two side by side. These files are written by a
separate thread fed through a bounded queue, so the audio thread only ever does
a non-blocking put and is never held up by disk I/O.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np


class WavRecorder:
    """Writes two synchronised WAV files: what came in, and what went out."""

    def __init__(self, directory: Path, samplerate: int, channels: int, log=None):
        self.dir = Path(directory)
        self.samplerate = samplerate
        self.channels = channels
        self.log = log
        self._q: queue.Queue[tuple[np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.frames = 0
        self.in_path = self.dir / "input.wav"
        self.out_path = self.dir / "output.wav"

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="wav-writer", daemon=True)
        self._thread.start()

    def submit(self, in_block: np.ndarray, out_block: np.ndarray) -> None:
        """Called from the audio thread. Never blocks; drops if the writer lags."""
        try:
            self._q.put_nowait((in_block.copy(), out_block.copy()))
        except queue.Full:
            self.dropped += 1

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=10.0)
        if self.log:
            self.log.info(
                "wav capture: %d frames (%.1fs) to %s and %s, %d blocks dropped",
                self.frames,
                self.frames / self.samplerate,
                self.in_path.name,
                self.out_path.name,
                self.dropped,
            )

    def _run(self) -> None:
        import soundfile as sf

        with sf.SoundFile(
            self.in_path, "w", self.samplerate, self.channels, subtype="FLOAT"
        ) as fin, sf.SoundFile(
            self.out_path, "w", self.samplerate, self.channels, subtype="FLOAT"
        ) as fout:
            while True:
                item = self._q.get()
                if item is None:
                    return
                a, b = item
                fin.write(a)
                fout.write(b)
                self.frames += len(a)


def compare(in_path: Path, out_path: Path) -> dict:
    """Measure the output against the input: level, dropouts, clipping, noise."""
    import soundfile as sf

    a, sr = sf.read(str(in_path), dtype="float32", always_2d=True)
    b, sr2 = sf.read(str(out_path), dtype="float32", always_2d=True)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    am, bm = a.mean(axis=1), b.mean(axis=1)

    win = max(1, sr // 100)  # 10 ms windows
    nw = n // win
    aw = np.sqrt((am[: nw * win].reshape(nw, win) ** 2).mean(axis=1))
    bw = np.sqrt((bm[: nw * win].reshape(nw, win) ** 2).mean(axis=1))

    loud = aw > (aw.max() * 0.02 if aw.size else 0)
    # A window that is loud going in but silent coming out is a dropout.
    holes = int(np.sum(loud & (bw < 1e-4)))

    res = {
        "samplerate": sr,
        "seconds": round(n / sr, 2),
        "in_peak": round(float(np.abs(a).max()), 4),
        "out_peak": round(float(np.abs(b).max()), 4),
        "in_rms": round(float(np.sqrt((am**2).mean())), 5),
        "out_rms": round(float(np.sqrt((bm**2).mean())), 5),
        "clipped_out": int((np.abs(b) >= 0.999).sum()),
        "dropout_windows_10ms": holes,
        "silent_out_fraction": round(float((bw < 1e-4).mean()), 4) if bw.size else 0.0,
    }
    if res["in_rms"] > 0:
        res["gain_db"] = round(20 * np.log10(res["out_rms"] / res["in_rms"] + 1e-12), 2)
    return res
