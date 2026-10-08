"""ctypes binding to the Signalsmith Stretch wrapper DLL.

Signalsmith Stretch (MIT) is a production-quality real-time pitch shifter. The
numpy phase vocoder in `shifter.py` stays as the fallback: this module reports
`available()` as False when the DLL has not been built, and nothing else needs to
care.

Build the DLL with `native\\build.bat`.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

DLL_NAME = "signalsmith_stretch.dll"
ABI_VERSION = 1


class _Lib:
    """Lazily loaded, so importing this module never fails."""

    _lib: ctypes.CDLL | None = None
    _error: str | None = None
    _tried = False

    @classmethod
    def get(cls) -> ctypes.CDLL | None:
        if cls._tried:
            return cls._lib
        cls._tried = True
        path = Path(__file__).with_name(DLL_NAME)
        if not path.exists():
            cls._error = f"{path.name} not built (run native\\build.bat)"
            return None
        try:
            lib = ctypes.CDLL(str(path))
            lib.ss_abi_version.restype = ctypes.c_int
            version = lib.ss_abi_version()
            if version != ABI_VERSION:
                cls._error = f"{path.name} has ABI {version}, expected {ABI_VERSION}"
                return None

            lib.ss_create.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int]
            lib.ss_create.restype = ctypes.c_void_p
            lib.ss_destroy.argtypes = [ctypes.c_void_p]
            lib.ss_destroy.restype = None
            lib.ss_set_semitones.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
            lib.ss_set_semitones.restype = None
            lib.ss_reset.argtypes = [ctypes.c_void_p]
            lib.ss_reset.restype = None
            lib.ss_input_latency.argtypes = [ctypes.c_void_p]
            lib.ss_input_latency.restype = ctypes.c_int
            lib.ss_output_latency.argtypes = [ctypes.c_void_p]
            lib.ss_output_latency.restype = ctypes.c_int
            lib.ss_process.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.ss_process.restype = ctypes.c_int
            cls._lib = lib
        except OSError as exc:
            cls._error = f"could not load {path.name}: {exc}"
        return cls._lib

    @classmethod
    def error(cls) -> str | None:
        cls.get()
        return cls._error


def available() -> bool:
    return _Lib.get() is not None


def unavailable_reason() -> str | None:
    return _Lib.error()


class SignalsmithCore:
    """Drop-in replacement for `_PhaseVocoderCore`, backed by the DLL.

    Presents the same interface the shifter already uses -- `process` on a
    (channels, frames) float32 block, `set_ratio`, `latency_samples` -- so it
    slots into the existing crossfade machinery unchanged.
    """

    def __init__(
        self,
        channels: int,
        n_fft: int = 2048,  # accepted for interface compatibility; engine picks its own
        overlap: int = 4,
        ratio: float = 1.0,
        cheaper: bool = False,
        samplerate: int = 48000,
        tonality_limit: float = 0.0,
        **_ignored,
    ):
        lib = _Lib.get()
        if lib is None:
            raise RuntimeError(unavailable_reason() or "signalsmith stretch unavailable")
        self._lib = lib
        self.channels = channels
        self.samplerate = samplerate
        self.tonality_limit = tonality_limit
        self.transients = 0  # the library handles transients internally
        self._handle = lib.ss_create(channels, float(samplerate), 1 if cheaper else 0)
        if not self._handle:
            raise RuntimeError("ss_create failed")
        self.ratio = 1.0
        self.set_ratio(ratio)
        self._in = np.zeros((channels, 0), dtype=np.float32)
        self._out = np.zeros((channels, 0), dtype=np.float32)

    # -- interface used by PitchShifter ----------------------------------
    @property
    def latency_samples(self) -> int:
        """Total group delay: the engine reports input and output legs separately
        and the audio is delayed by both."""
        return int(
            self._lib.ss_input_latency(self._handle)
            + self._lib.ss_output_latency(self._handle)
        )

    def set_ratio(self, ratio: float) -> None:
        ratio = float(ratio)
        if ratio <= 0:
            return
        self.ratio = ratio
        semitones = 12.0 * np.log2(ratio)
        self._lib.ss_set_semitones(
            self._handle, ctypes.c_float(semitones), ctypes.c_float(self.tonality_limit)
        )

    def reset(self) -> None:
        self._lib.ss_reset(self._handle)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(block, dtype=np.float32)
        if block.ndim != 2 or block.shape[0] != self.channels:
            raise ValueError(
                f"expected ({self.channels}, frames) float32, got {block.shape}"
            )
        frames = block.shape[1]
        if frames == 0:
            return block
        if self._out.shape[1] != frames:
            self._out = np.zeros((self.channels, frames), dtype=np.float32)
        rc = self._lib.ss_process(
            self._handle,
            block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            frames,
        )
        if rc != 0:
            raise RuntimeError(f"ss_process failed ({rc})")
        return self._out.copy()

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            try:
                self._lib.ss_destroy(handle)
            except Exception:
                pass
            self._handle = None
