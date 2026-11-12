"""Backend-neutral network interface.

Both the PyTorch implementation (:mod:`katago.nn.model`) and the NumPy reference
implementation (:mod:`katago.nn.numpy_net`) expose this interface, so MCTS,
self-play, the trainer and the playing engine never branch on the backend.

All batches are plain dicts of ``float32``/``bool`` NumPy arrays:

============  ==================  ==============================================
key           shape               meaning
============  ==================  ==============================================
``spatial``   (B, 18, S, S)       18 binary input planes
``global_``   (B, 10)             10 real-valued global features
``legal``     (B, M)              boolean legal-move mask, M = S*S + 1
``policy``    (B, M)              policy target (already pruned), sums to 1
``aux``       (B, M)              opponent-reply target, sums to 1 (or absent)
``value``     (B, 2)              one-hot ``[loss, win]``
``ownership`` (B, S*S)           ownership target in ``[0, 1]`` for the mover
``score``     (B,)                int index into the score-belief bins
============  ==================  ==============================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import LossConfig, NetConfig


@dataclass
class EvalResult:
    """Network output for a batch of positions.

    ``policy`` has already been masked to legal moves and renormalised, so MCTS
    can use it directly as a prior.
    """

    policy: np.ndarray  # (B, M) probabilities over legal moves
    value: np.ndarray  # (B, 2) probabilities [loss, win]
    ownership: np.ndarray  # (B, S*S) in [0, 1], share owned by the player to move
    score: np.ndarray  # (B, K) score-belief distribution
    aux_policy: np.ndarray  # (B, M) opponent-reply distribution

    def __len__(self) -> int:
        return int(self.policy.shape[0])


def masked_softmax(logits: np.ndarray, legal: Optional[np.ndarray]) -> np.ndarray:
    """Softmax over the legal entries; the rest get exactly zero."""
    z = np.asarray(logits, dtype=np.float64)
    if legal is not None:
        z = np.where(legal, z, -1e30)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    s = e.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return (e / s).astype(np.float32)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


class BaseNet:
    """Common interface. Subclasses implement the four abstract members."""

    def __init__(self, config: NetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------- inference
    def predict(
        self,
        spatial: np.ndarray,
        global_: np.ndarray,
        legal: Optional[np.ndarray] = None,
    ) -> EvalResult:
        raise NotImplementedError

    # -------------------------------------------------------------- training
    def train_batch(
        self, batch: Dict[str, np.ndarray], loss_cfg: LossConfig, lr: float
    ) -> Dict[str, float]:
        """One SGD step (momentum already configured). Returns metrics."""
        raise NotImplementedError

    # ------------------------------------------------------------------ state
    def state_dict(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    @property
    def num_moves(self) -> int:
        return self.config.num_moves

    def num_parameters(self) -> int:
        return int(sum(v.size for v in self.state_dict().values()))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(blocks={self.config.block_count}, "
            f"channels={self.config.channels}, params={self.num_parameters():,})"
        )


def build_batch_arrays(
    spatial: List[np.ndarray],
    glob: List[np.ndarray],
    legal: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        "spatial": np.stack(spatial).astype(np.float32),
        "global_": np.stack(glob).astype(np.float32),
        "legal": np.stack(legal).astype(bool),
    }
