"""Pitch-class key profiles, and the key/shift algebra for auto-transposition.

The whole point of this app is a shift, not a key name. That matters because the
single most common key-detection error is confusing a key with its relative
major/minor (C major vs A minor) -- and those two keys need the *same* shift to
land on the white keys. So everything downstream votes on a shift class, not on
a key, which makes the detector far more stable than its raw key accuracy.
"""
from __future__ import annotations

import numpy as np

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# The naturals: the only notes the player can reach.
WHITE_PITCH_CLASSES = (0, 2, 4, 5, 7, 9, 11)
_WHITE_MASK = np.zeros(12, dtype=bool)
_WHITE_MASK[list(WHITE_PITCH_CLASSES)] = True


def white_mass_per_shift(chroma: np.ndarray) -> np.ndarray:
    """Share of pitch-class energy landing on white keys, for each shift class.

    This is the objective the app actually exists to maximise. Asking it directly
    is more robust than inferring a key and deriving a shift: it rests on the
    mass of seven pitch classes rather than on the one or two scale degrees that
    separate neighbouring keys, which is exactly where key detection fails.
    """
    chroma = np.asarray(chroma, dtype=np.float64).reshape(12)
    total = chroma.sum()
    if total <= 1e-12:
        return np.zeros(12)
    out = np.empty(12, dtype=np.float64)
    pcs = np.arange(12)
    for s in range(12):
        out[s] = chroma[_WHITE_MASK[(pcs + s) % 12]].sum() / total
    return out

# Published pitch-class profiles. Each is (major, minor), indexed from the tonic.
PROFILES: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    # Krumhansl & Kessler (1982) probe-tone ratings.
    "krumhansl": (
        (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
        (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    ),
    # Temperley (2001), from the Kostka-Payne corpus.
    "temperley": (
        (0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400),
        (0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330),
    ),
    # Albrecht & Shanahan (2013), fitted on a large score corpus.
    "albrecht": (
        (0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081),
        (0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052),
    ),
    # Sha'ath -- libKeyFinder's default, tuned on recorded popular music rather
    # than on scores or probe tones, which is the closest match to our input.
    "shaath": (
        (6.6, 2.0, 3.6, 2.1, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4),
        (6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2),
    ),
}

# Profiles averaged together after z-scoring; more robust than any single one.
BLEND_DEFAULT = ("shaath", "temperley", "krumhansl")

KEY_LABELS: tuple[str, ...] = tuple(
    [f"{n} major" for n in PITCH_CLASS_NAMES] + [f"{n} minor" for n in PITCH_CLASS_NAMES]
)


def _zscore(v: np.ndarray, axis: int = -1) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    mean = v.mean(axis=axis, keepdims=True)
    std = v.std(axis=axis, keepdims=True)
    return (v - mean) / np.maximum(std, 1e-12)


def template_matrix(profile: str | tuple[str, ...] = "blend") -> np.ndarray:
    """Return a (24, 12) z-scored template matrix.

    Rows 0-11 are major keys with tonic pitch class 0-11; rows 12-23 are the
    minor keys. Column index is absolute pitch class, so a chroma vector can be
    scored against every key with one matrix product.
    """
    names: tuple[str, ...]
    if profile == "blend":
        names = BLEND_DEFAULT
    elif isinstance(profile, str):
        names = (profile,)
    else:
        names = tuple(profile)

    stack = []
    for name in names:
        try:
            major, minor = PROFILES[name]
        except KeyError:
            raise ValueError(
                f"unknown key profile {name!r}; choose from {sorted(PROFILES)} or 'blend'"
            ) from None
        rows = np.empty((24, 12), dtype=np.float64)
        for tonic in range(12):
            rows[tonic] = np.roll(major, tonic)
            rows[tonic + 12] = np.roll(minor, tonic)
        stack.append(_zscore(rows, axis=1))
    return _zscore(np.mean(stack, axis=0), axis=1)


def key_scores(chroma: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Pearson correlation of a 12-bin chroma vector against all 24 keys."""
    obs = _zscore(np.asarray(chroma, dtype=np.float64).reshape(12))
    return (templates @ obs) / 12.0


def shift_class_for_key(key_index: int) -> int:
    """Shift class (0-11) that moves `key_index` onto C major / A minor."""
    pc = key_index % 12
    relative_major = pc if key_index < 12 else (pc + 3) % 12
    return (-relative_major) % 12


def key_indices_for_shift_class(shift_class: int) -> tuple[int, int]:
    """The (major, minor) key indices that both need this shift class."""
    major_pc = (-shift_class) % 12
    minor_pc = (9 - shift_class) % 12
    return major_pc, minor_pc + 12


def collapse_to_shift_classes(scores24: np.ndarray) -> np.ndarray:
    """Fold 24 key scores into 12 shift-class scores, keeping the better key.

    Relative major/minor pairs collapse onto the same entry, so the classic
    major/minor confusion costs us nothing.
    """
    scores24 = np.asarray(scores24, dtype=np.float64).reshape(24)
    out = np.empty(12, dtype=np.float64)
    for s in range(12):
        maj, minr = key_indices_for_shift_class(s)
        out[s] = max(scores24[maj], scores24[minr])
    return out


def signed_shift(shift_class: int, prefer_down: bool = True) -> int:
    """Smallest-magnitude signed semitone shift congruent to `shift_class`."""
    up = shift_class % 12
    down = up - 12
    if abs(up) < abs(down):
        return up
    if abs(down) < abs(up):
        return down
    return down if prefer_down else up


def key_label(key_index: int) -> str:
    return KEY_LABELS[key_index % 24]
