"""Live key detection: incremental chroma extraction + rolling key estimation.

A full CQT over the whole analysis window every cycle is too slow to run live
(~320 ms for 24 s of audio, and HPSS is ~4 s, which rules it out entirely).
Instead we extract chroma frames incrementally from each new hop of audio and
keep a rolling deque of frames, so per-cycle cost stays in the low tens of ms.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import librosa
import numpy as np

from .keyprofiles import (
    collapse_to_shift_classes,
    key_label,
    key_scores,
    shift_class_for_key,
    template_matrix,
    white_mass_per_shift,
)

ANALYSIS_SR = 22050
HOP_LENGTH = 2048
# The lowest CQT bin needs roughly Q * sr / fmin samples of support, which is
# about 0.8 s at C2 with 36 bins/octave. Keep a bit more than that as context so
# incrementally computed frames match what a whole-window CQT would produce.
CONTEXT_SECONDS = 1.2
# Pinned: estimating tuning per window is both jittery on 2-3 s of audio and
# needless for commercial music, which is cut at A440.
TUNING = 0.0
FMIN_NOTE = "C2"
N_OCTAVES = 6


class IncrementalChroma:
    """Turns a stream of mono audio into a stream of chroma frames."""

    def __init__(
        self,
        sr: int = ANALYSIS_SR,
        hop_length: int = HOP_LENGTH,
        fmin_note: str = FMIN_NOTE,
        n_octaves: int = N_OCTAVES,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.fmin = librosa.note_to_hz(fmin_note)
        self._context = int(CONTEXT_SECONDS * sr)
        # Round context to a whole number of hops so frame centres stay aligned.
        self._context -= self._context % hop_length
        self._tail = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._tail = np.zeros(0, dtype=np.float32)

    def push(self, mono: np.ndarray) -> np.ndarray:
        """Feed new audio; return the chroma frames it produced, shape (12, n)."""
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._tail, mono]) if self._tail.size else mono

        if buf.size < self._context + self.hop_length:
            self._tail = buf
            return np.zeros((12, 0), dtype=np.float32)

        chroma = librosa.feature.chroma_cqt(
            y=buf,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_octaves=self.n_octaves,
            bins_per_octave=36,
            tuning=0.0,
        )
        # Frame k is centred at k * hop. Keep only frames centred inside the
        # newly arrived audio; everything earlier was already emitted, and
        # frames near the buffer edges are contaminated by zero padding.
        n_new = buf.size - self._tail.size if self._tail.size else buf.size
        first_new = max(0, int(np.ceil((buf.size - n_new) / self.hop_length)))
        last_safe = max(first_new, chroma.shape[1] - 1)
        fresh = chroma[:, first_new:last_safe]

        self._tail = buf[-self._context:].copy()
        return fresh.astype(np.float32)


@dataclass(frozen=True)
class KeyEstimate:
    """One estimate from the rolling window."""

    shift_class: int
    shift_scores: np.ndarray  # (12,) correlation per shift class
    best_key: int  # 0-23
    key_confidence: float  # softmax probability of the winning shift class
    margin: float  # best minus runner-up correlation
    frames: int
    novelty: float  # 0 = chroma matches recent history, 1 = total change
    # Share of energy that would land on white keys, per shift class. This is the
    # quantity the gain-based decider compares; it is in real units (fraction of
    # the music the player can reach), not correlation units.
    white_mass: np.ndarray | None = None

    @property
    def key_name(self) -> str:
        return key_label(self.best_key)


class KeyEstimator:
    """Rolling energy-weighted chroma aggregation scored against key profiles."""

    def __init__(
        self,
        window_seconds: float = 20.0,
        profile: str | tuple[str, ...] = "blend",
        sr: int = ANALYSIS_SR,
        hop_length: int = HOP_LENGTH,
        softmax_temperature: float = 0.06,
        objective: str = "white",
    ):
        if objective not in ("white", "template"):
            raise ValueError("objective must be 'white' or 'template'")
        self.objective = objective
        self.templates = template_matrix(profile)
        self.temperature = softmax_temperature
        self.frame_seconds = hop_length / sr
        self.max_frames = max(8, int(window_seconds / self.frame_seconds))
        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=self.max_frames)
        self._weights: collections.deque[float] = collections.deque(maxlen=self.max_frames)
        self._prev_mean: np.ndarray | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._weights.clear()
        self._prev_mean = None

    @property
    def filled_seconds(self) -> float:
        return len(self._frames) * self.frame_seconds

    def add_frames(self, chroma: np.ndarray) -> None:
        if chroma.size == 0:
            return
        # chroma_cqt frames are already normalised per frame, so use the raw
        # magnitude sum as a crude "is there harmonic content here" weight.
        energies = chroma.sum(axis=0)
        for i in range(chroma.shape[1]):
            self._frames.append(chroma[:, i].astype(np.float64))
            self._weights.append(float(energies[i]))

    def estimate(self) -> KeyEstimate | None:
        if len(self._frames) < 8:
            return None

        frames = np.stack(self._frames, axis=1)  # (12, n)
        weights = np.asarray(self._weights, dtype=np.float64)
        total = weights.sum()
        if total <= 1e-9:
            return None
        mean = (frames * weights).sum(axis=1) / total

        novelty = 0.0
        if self._prev_mean is not None:
            a, b = mean, self._prev_mean
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 1e-12:
                novelty = float(np.clip(1.0 - (a @ b) / denom, 0.0, 1.0))
        self._prev_mean = mean

        scores24 = key_scores(mean, self.templates)
        template_scores = collapse_to_shift_classes(scores24)
        white = white_mass_per_shift(mean)
        shift_scores = template_scores if self.objective == "template" else white

        order = np.argsort(shift_scores)[::-1]
        best_class = int(order[0])
        margin = float(shift_scores[order[0]] - shift_scores[order[1]])

        exp = np.exp((shift_scores - shift_scores.max()) / self.temperature)
        probs = exp / exp.sum()

        # Report whichever of the two relative keys scored higher, for display.
        from .keyprofiles import key_indices_for_shift_class

        maj, minr = key_indices_for_shift_class(best_class)
        best_key = maj if scores24[maj] >= scores24[minr] else minr

        return KeyEstimate(
            shift_class=best_class,
            shift_scores=shift_scores,
            best_key=int(best_key),
            key_confidence=float(probs[best_class]),
            margin=margin,
            frames=len(self._frames),
            novelty=novelty,
            white_mass=white,
        )


def detect_key_offline(y: np.ndarray, sr: int, profile: str | tuple[str, ...] = "blend") -> KeyEstimate:
    """Whole-signal key estimate. Used by the tests and by `autotranspose check`."""
    if sr != ANALYSIS_SR:
        import soxr

        y = soxr.resample(np.asarray(y, dtype=np.float32), sr, ANALYSIS_SR)
    est = KeyEstimator(window_seconds=1e6, profile=profile)
    chroma = librosa.feature.chroma_cqt(
        y=np.asarray(y, dtype=np.float32),
        sr=ANALYSIS_SR,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz(FMIN_NOTE),
        n_octaves=N_OCTAVES,
        bins_per_octave=36,
        tuning=0.0,
    )
    est.add_frames(chroma)
    result = est.estimate()
    if result is None:
        raise ValueError("signal too short or silent to estimate a key")
    return result


__all__ = [
    "ANALYSIS_SR",
    "IncrementalChroma",
    "KeyEstimate",
    "KeyEstimator",
    "detect_key_offline",
    "shift_class_for_key",
]
