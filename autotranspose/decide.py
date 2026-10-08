"""Turns a stream of noisy key estimates into a stable transpose decision.

A single estimate is not trustworthy. On test material the detector sits at 94%
confidence on the *wrong* answer (a fifth away) for the first several seconds of
a track, before settling correctly once it has heard a full progression. So we
never act on one estimate: votes accumulate with exponential decay, and a shift
is only committed once a leader holds a clear majority for a sustained stretch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .keydetect import KeyEstimate
from .keyprofiles import key_indices_for_shift_class, key_label, signed_shift


@dataclass
class DecisionState:
    """Everything the UI needs to render, published once per analysis cycle."""

    status: str = "starting"  # starting | listening | locked | silent | bypassed
    applied_shift: int = 0  # signed semitones currently being applied
    candidate_class: int | None = None
    candidate_shift: int | None = None
    leader_share: float = 0.0
    key_name: str = "--"
    key_confidence: float = 0.0
    margin: float = 0.0
    heard_seconds: float = 0.0
    votes: np.ndarray = field(default_factory=lambda: np.zeros(12))
    # Fraction of the music landing on white keys: with the shift in use now, and
    # with the best available shift. The gap is what a change would buy.
    white_now: float = 0.0
    white_best: float = 0.0
    locked_at: float | None = None
    changes: int = 0
    manual: bool = False


class ShiftDecider:
    def __init__(
        self,
        *,
        min_heard_seconds: float = 12.0,
        commit_share: float = 0.55,
        commit_margin: float = 0.04,
        stable_seconds: float = 3.0,
        min_dwell_seconds: float = 25.0,
        decay_halflife: float = 8.0,
        cycle_seconds: float = 1.5,
        max_shift: int = 6,
        prefer_down: bool = True,
        silence_reset_seconds: float = 2.5,
        on_new_track: str = "hold",
        mode: str = "gain",
        min_gain: float = 0.05,
        gain_stable_cycles: int = 3,
    ):
        if mode not in ("gain", "votes"):
            raise ValueError("mode must be 'gain' or 'votes'")
        self.mode = mode
        self.min_gain = min_gain
        self.gain_stable_cycles = gain_stable_cycles
        self._applied_class = 0
        self._gain_candidate: int | None = None
        self._gain_streak = 0
        self.min_heard_seconds = min_heard_seconds
        self.commit_share = commit_share
        self.commit_margin = commit_margin
        self.stable_seconds = stable_seconds
        self.min_dwell_seconds = min_dwell_seconds
        self.max_shift = max_shift
        self.prefer_down = prefer_down
        self.silence_reset_seconds = silence_reset_seconds
        self.on_new_track = on_new_track
        self.cycle_seconds = cycle_seconds
        self._decay = 0.5 ** (cycle_seconds / max(decay_halflife, 1e-6))

        self._votes = np.zeros(12, dtype=np.float64)
        self._leader: int | None = None
        self._leader_since: float | None = None
        self._applied = 0
        self._applied_at: float | None = None
        self._locked = False
        self._changes = 0
        self._silent_for = 0.0
        self._manual: int | None = None
        self.state = DecisionState()

    # -- manual control -----------------------------------------------------
    def set_manual(self, semitones: int | None) -> None:
        """Pin the shift (or pass None to hand control back to the detector)."""
        self._manual = semitones
        if semitones is not None:
            self._applied = int(semitones)
            self._applied_class = int(semitones) % 12

    @property
    def manual(self) -> bool:
        return self._manual is not None

    def new_track(self) -> None:
        """Forget accumulated evidence; the music has changed."""
        self._votes[:] = 0.0
        self._leader = None
        self._leader_since = None
        self._locked = False
        self._gain_candidate = None
        self._gain_streak = 0
        if self.on_new_track == "passthrough":
            self._applied = 0
            self._applied_class = 0

    # -- main update --------------------------------------------------------
    def update(
        self,
        estimate: KeyEstimate | None,
        *,
        heard_seconds: float,
        silent: bool,
        now: float | None = None,
    ) -> int:
        """Fold in one estimate and return the shift that should be applied."""
        now = time.monotonic() if now is None else now

        if self._manual is not None:
            self.state = DecisionState(
                status="manual",
                applied_shift=self._applied,
                heard_seconds=heard_seconds,
                votes=self._votes.copy(),
                changes=self._changes,
                manual=True,
            )
            return self._applied

        if silent:
            self._silent_for += self.cycle_seconds
            if self._silent_for >= self.silence_reset_seconds:
                self.new_track()
            self.state = DecisionState(
                status="silent",
                applied_shift=self._applied,
                heard_seconds=heard_seconds,
                votes=self._votes.copy(),
                changes=self._changes,
            )
            return self._applied
        self._silent_for = 0.0

        if estimate is None:
            self.state = DecisionState(
                status="listening",
                applied_shift=self._applied,
                heard_seconds=heard_seconds,
                votes=self._votes.copy(),
                changes=self._changes,
            )
            return self._applied

        # A big chroma jump means the music changed underneath us; stale votes
        # would only slow down the new lock.
        if estimate.novelty > 0.45:
            self.new_track()

        if self.mode == "gain" and estimate.white_mass is not None:
            return self._update_gain(estimate, heard_seconds, now)

        # Vote with the whole probability vector rather than just the argmax, so
        # a genuinely ambiguous cycle contributes ambiguity instead of a hard
        # vote for whichever class edged ahead.
        probs = _softmax(estimate.shift_scores, temperature=0.06)
        self._votes *= self._decay
        self._votes += probs

        total = self._votes.sum()
        share = self._votes / total if total > 0 else self._votes
        order = np.argsort(share)[::-1]
        leader = int(order[0])
        leader_share = float(share[leader])

        if leader != self._leader:
            self._leader = leader
            self._leader_since = now
        held = now - (self._leader_since or now)

        candidate_shift = self._clamp(signed_shift(leader, self.prefer_down))

        ready = (
            heard_seconds >= self.min_heard_seconds
            and leader_share >= self.commit_share
            and estimate.margin >= self.commit_margin
            and held >= self.stable_seconds
        )
        dwell_ok = self._applied_at is None or (now - self._applied_at) >= self.min_dwell_seconds

        if ready and candidate_shift != self._applied and (dwell_ok or not self._locked):
            self._applied = candidate_shift
            self._applied_at = now
            self._changes += 1
            self._locked = True
        elif ready:
            self._locked = True

        maj, minr = key_indices_for_shift_class(leader)
        self.state = DecisionState(
            status="locked" if self._locked else "listening",
            applied_shift=self._applied,
            candidate_class=leader,
            candidate_shift=candidate_shift,
            leader_share=leader_share,
            key_name=estimate.key_name,
            key_confidence=estimate.key_confidence,
            margin=estimate.margin,
            heard_seconds=heard_seconds,
            votes=share.copy(),
            locked_at=self._applied_at,
            changes=self._changes,
        )
        return self._applied

    def _update_gain(self, estimate: KeyEstimate, heard_seconds: float, now: float) -> int:
        """Change only when the best shift beats the one in use by `min_gain`.

        Comparing the leader against the runner-up is a coin toss whenever two
        shifts are nearly as good -- which is the normal case, and what made the
        detector oscillate. Comparing it against what is already applied is
        decisive: moving off no-shift is worth tens of points, while swapping
        between two near-equivalent shifts is worth a fraction of one, so it
        simply does not happen.
        """
        white = np.asarray(estimate.white_mass, dtype=np.float64)
        best = int(np.argmax(white))
        current = white[self._applied_class]
        gain = float(white[best] - current)

        if best == self._gain_candidate:
            self._gain_streak += 1
        else:
            self._gain_candidate, self._gain_streak = best, 1

        dwell_ok = self._applied_at is None or (now - self._applied_at) >= self.min_dwell_seconds
        candidate_shift = self._clamp(signed_shift(best, self.prefer_down))

        white_before = float(white[self._applied_class])
        if (
            best != self._applied_class
            and gain >= self.min_gain
            and heard_seconds >= self.min_heard_seconds
            and self._gain_streak >= self.gain_stable_cycles
            and dwell_ok
        ):
            self._applied_class = best
            self._applied = candidate_shift
            self._applied_at = now
            self._changes += 1
            self._locked = True
        elif heard_seconds >= self.min_heard_seconds and self._gain_streak >= self.gain_stable_cycles:
            self._locked = True

        maj, minr = key_indices_for_shift_class(best)
        self.state = DecisionState(
            status="locked" if self._locked else "listening",
            applied_shift=self._applied,
            candidate_class=best,
            candidate_shift=candidate_shift,
            leader_share=float(white[best]),
            key_name=estimate.key_name,
            key_confidence=estimate.key_confidence,
            margin=gain,
            heard_seconds=heard_seconds,
            votes=white.copy(),
            locked_at=self._applied_at,
            changes=self._changes,
            # Coverage as it was before any switch this cycle, so a log line
            # reporting "x% -> y% (gain g)" shows the gain that justified it
            # rather than comparing the new value with itself.
            white_now=white_before,
            white_best=float(white[best]),
        )
        return self._applied

    def _clamp(self, shift: int) -> int:
        """Keep the shift inside the allowed range, staying in the same key."""
        if abs(shift) <= self.max_shift:
            return shift
        alt = shift - 12 if shift > 0 else shift + 12
        return alt if abs(alt) <= self.max_shift else shift


def _softmax(x: np.ndarray, temperature: float = 0.06) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp((x - x.max()) / max(temperature, 1e-9))
    return e / e.sum()


def shift_explanation(shift: int, key_name: str) -> str:
    direction = "down" if shift < 0 else "up"
    if shift == 0:
        return f"{key_name} is already on the white keys"
    return f"{key_name} -> shift {direction} {abs(shift)} semitone{'s' if abs(shift) != 1 else ''}"


__all__ = ["DecisionState", "ShiftDecider", "key_label", "shift_explanation"]
