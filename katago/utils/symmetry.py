"""Dihedral (8-way) symmetry transforms.

AlphaGo Zero and KataGo both augment self-play data with the 8 symmetries of the
Go board ("random rotations or reflections of the position during search").
Pass is unaffected by these transforms, so it always keeps its own slot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

NUM_SYMMETRIES = 8


def apply_planes(x: np.ndarray, k: int) -> np.ndarray:
    """Apply symmetry ``k`` to the last two axes of ``x``."""
    if k >= 4:
        x = np.flip(x, axis=-1)
    return np.rot90(x, k % 4, axes=(-2, -1))


@lru_cache(maxsize=64)
def point_permutation(k: int, size: int) -> Tuple[int, ...]:
    """For symmetry ``k``, the permutation mapping old point indices to new ones.

    ``new_flat[perm] = old_flat``.
    """
    idx = np.arange(size * size).reshape(size, size)
    return tuple(apply_planes(idx, k).reshape(-1).tolist())


def apply_policy(policy: np.ndarray, k: int, size: int) -> np.ndarray:
    """Transform a policy vector of length ``size*size + 1``."""
    n = size * size
    out = np.empty_like(policy)
    perm = np.asarray(point_permutation(k, size))
    out[perm] = policy[:n]
    out[n:] = policy[n:]
    return out


def random_symmetry(rng: np.random.Generator) -> int:
    return int(rng.integers(0, NUM_SYMMETRIES))
