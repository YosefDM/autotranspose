"""Synthesise short, deliberately band-like audio in a known key, for tests.

Not a replacement for real music, but it exercises the real failure modes:
harmonic-rich timbres, a bass line, inversions, percussion noise and a melody
that wanders outside the triad.
"""
from __future__ import annotations

import numpy as np

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
# I - V - vi - IV and i - VI - III - VII, as (scale degree, is_minor_triad)
MAJOR_PROG = [(0, False), (4, False), (5, True), (3, False)]
MINOR_PROG = [(0, True), (5, False), (2, False), (6, False)]


def _note_hz(semitones_above_c4: float) -> float:
    return 261.625565 * (2.0 ** (semitones_above_c4 / 12.0))


def _tone(freq: float, dur: float, sr: int, n_harmonics: int = 7, amp: float = 1.0) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        f = freq * h
        if f > sr * 0.45:
            break
        out += np.sin(2 * np.pi * f * t + h) / h
    env = np.exp(-t * 1.6) * (1.0 - np.exp(-t * 220.0))
    return (out * env * amp).astype(np.float64)


def _triad(root_semi: float, minor: bool) -> list[float]:
    third = 3 if minor else 4
    return [root_semi, root_semi + third, root_semi + 7]


def _add(out: np.ndarray, start: int, sig: np.ndarray) -> None:
    """Mix `sig` into `out` at `start`, clipping anything past the end."""
    if start >= out.size:
        return
    end = min(out.size, start + sig.size)
    out[start:end] += sig[: end - start]


def synth_song(
    tonic_pc: int,
    minor_key: bool = False,
    duration: float = 24.0,
    sr: int = 22050,
    bpm: float = 100.0,
    percussion: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """Render a chord progression plus bass, melody and percussion in one key."""
    rng = np.random.default_rng(seed)
    scale = MINOR_SCALE if minor_key else MAJOR_SCALE
    prog = MINOR_PROG if minor_key else MAJOR_PROG
    bar = 4 * 60.0 / bpm
    n = int(duration * sr)
    out = np.zeros(n + int(bar * sr) + sr, dtype=np.float64)

    bar_index = 0
    pos = 0.0
    while pos < duration:
        degree, is_minor_triad = prog[bar_index % len(prog)]
        root_pc = tonic_pc + scale[degree % 7]
        start = int(pos * sr)

        # Chord, voiced around C4, occasionally inverted.
        voices = _triad(root_pc, is_minor_triad)
        if bar_index % 3 == 2:
            voices = [voices[1], voices[2], voices[0] + 12]
        for v in voices:
            _add(out, start, _tone(_note_hz(v), bar, sr, amp=0.30))

        # Bass: root then fifth, two octaves down.
        for k, semi in enumerate((root_pc - 24, root_pc - 24 + 7)):
            _add(out, start + int(k * bar / 2 * sr),
                 _tone(_note_hz(semi), bar / 2, sr, n_harmonics=4, amp=0.55))

        # Melody: eighth notes drawn from the scale, an octave up.
        steps = 8
        for k in range(steps):
            deg = int(rng.integers(0, 7))
            semi = tonic_pc + scale[deg] + 12
            _add(out, start + int(k * bar / steps * sr),
                 _tone(_note_hz(semi), bar / steps * 1.4, sr, amp=0.18))

        # Percussion: broadband bursts on the beat. Pitch-neutral, but it smears
        # the chroma, which is exactly what we need to be robust to.
        if percussion > 0:
            for k in range(8):
                ln = int(0.05 * sr)
                burst = rng.standard_normal(ln) * np.exp(-np.arange(ln) / (0.006 * sr))
                _add(out, start + int(k * bar / 8 * sr),
                     burst * percussion * (0.6 if k % 2 else 1.0))

        pos += bar
        bar_index += 1

    out = out[:n]
    peak = float(np.abs(out).max())
    return (out / peak * 0.7).astype(np.float32) if peak > 0 else out.astype(np.float32)
