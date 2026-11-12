"""Replay buffer with a growing window, and symmetry-augmented batch sampling.

KataGo samples "uniformly from a growing moving window of the most recent data,
with window size beginning at 250,000 samples and increasing to about 22 million"
(Wu 2020, Section 2).  The window is expressed in samples but stored as whole
games, which keeps memory proportional to the number of positions actually kept.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from ..utils.symmetry import apply_planes, apply_policy
from .worker import GameRecord


class ReplayBuffer:
    def __init__(self, window_samples: int = 250_000, seed: int = 0) -> None:
        self.games: deque[GameRecord] = deque()
        self.window_samples = int(window_samples)
        self.rng = np.random.default_rng(seed)
        self.total_samples_seen = 0
        self._cached_samples: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ state
    def __len__(self) -> int:
        return len(self.games)

    @property
    def num_samples(self) -> int:
        return sum(g.num_samples for g in self.games)

    def add_game(self, record: GameRecord) -> None:
        self.games.append(record)
        self.total_samples_seen += record.num_samples
        self._evict()
        self._cached_samples = None

    def set_window(self, window_samples: int) -> None:
        self.window_samples = int(window_samples)
        self._evict()
        self._cached_samples = None

    def _evict(self) -> None:
        while len(self.games) > 1 and sum(g.num_samples for g in self.games) > self.window_samples:
            self.games.popleft()

    def _flatten(self) -> Dict[str, np.ndarray]:
        if self._cached_samples is not None:
            return self._cached_samples
        parts = [g.samples() for g in self.games]
        out: Dict[str, np.ndarray] = {}
        for key in ("spatial", "global_", "policy", "aux", "has_aux", "value", "ownership", "score"):
            out[key] = np.concatenate([p[key] for p in parts], axis=0)
        self._cached_samples = out
        return out

    # --------------------------------------------------------------- sampling
    def sample_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        augment: bool = True,
        size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Sample a training batch, applying a random dihedral symmetry per sample."""
        if not self.games:
            raise RuntimeError("replay buffer is empty")
        rng = rng or self.rng
        data = self._flatten()
        n = data["spatial"].shape[0]
        idx = rng.integers(0, n, size=batch_size)
        size = size or int(data["spatial"].shape[-1])

        spatial = np.empty((batch_size, *data["spatial"].shape[1:]), dtype=np.float32)
        own = np.empty((batch_size, data["ownership"].shape[1]), dtype=np.float32)
        policy = np.empty((batch_size, data["policy"].shape[1]), dtype=np.float32)
        aux = np.empty_like(policy)

        for j, i in enumerate(idx):
            k = int(rng.integers(0, 8)) if augment else 0
            spatial[j] = apply_planes(data["spatial"][i], k)
            # ownership is stored flat; apply the symmetry in 2-D
            own2 = apply_planes(data["ownership"][i].reshape(size, size), k)
            own[j] = own2.reshape(-1)
            policy[j] = apply_policy(data["policy"][i], k, size)
            aux[j] = apply_policy(data["aux"][i], k, size)

        has_aux = data["has_aux"][idx]
        # moves without a recorded successor get a uniform auxiliary target,
        # which contributes nothing once the loss weight cancels the constant
        aux[~has_aux] = 0.0
        tot = aux.sum(axis=1, keepdims=True)
        aux = np.divide(aux, np.maximum(tot, 1e-12))

        return {
            "spatial": spatial,
            "global_": data["global_"][idx].astype(np.float32),
            "policy": policy,
            "aux": aux,
            "value": data["value"][idx].astype(np.float32),
            "ownership": own,
            "score": data["score"][idx].astype(np.int64),
        }
