"""Neural network backends.

The primary implementation targets PyTorch (:mod:`katago.nn.model`).  When PyTorch
is unavailable, :mod:`katago.nn.numpy_net` provides a reference implementation
built on the small autodiff engine in :mod:`katago.nn.autograd`, so the rest of the
pipeline (MCTS, self-play, training, inference) runs unchanged.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseNet, EvalResult, masked_softmax, softmax_rows
from .config import LossConfig, MCTSConfig, NetConfig, PlayoutCapConfig, TrainConfig

__all__ = [
    "NetConfig",
    "MCTSConfig",
    "PlayoutCapConfig",
    "LossConfig",
    "TrainConfig",
    "BaseNet",
    "EvalResult",
    "masked_softmax",
    "softmax_rows",
    "torch_available",
    "resolve_torch_device",
    "build_net",
    "BackendUnavailable",
]


class BackendUnavailable(RuntimeError):
    """Raised when a requested backend is not importable in this environment."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def resolve_torch_device(requested: str = "auto") -> str:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` / ``cuda:1`` to a concrete device.

    ``auto`` picks CUDA when it is really usable, so a CPU-only PyTorch build
    silently stays on the CPU instead of failing at the first tensor transfer.
    """
    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch means no CUDA either
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 - broken driver / no CUDA runtime
        return "cpu"
    return "cpu"


def build_net(config: NetConfig, backend: str = "auto", **kwargs) -> BaseNet:
    """Construct a network for ``config`` using the best available backend.

    Parameters
    ----------
    backend:
        ``"torch"``, ``"numpy"`` or ``"auto"`` (prefer PyTorch).
    """
    if backend == "auto":
        backend = "torch" if torch_available() else "numpy"
    if backend == "torch":
        if not torch_available():
            raise BackendUnavailable(
                "PyTorch is not installed; pass backend='numpy' for the reference backend"
            )
        from .model import TorchNet

        return TorchNet(config, **kwargs)
    if backend == "numpy":
        from .numpy_net import NumpyNet

        return NumpyNet(config, **kwargs)
    raise ValueError(f"unknown backend {backend!r}")
