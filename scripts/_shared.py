"""Shared helpers for the command-line scripts: build a net from a checkpoint."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from katago.nn import NetConfig, build_net, torch_available
from katago.nn.checkpoint import load_checkpoint


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "torch" if torch_available() else "numpy"
    if backend == "torch" and not torch_available():
        raise SystemExit("error: PyTorch is not installed; use --backend numpy")
    return backend


def build_from_checkpoint(
    backend: str = "auto",
    checkpoint: Optional[str] = None,
    size: Optional[int] = None,
    blocks: Optional[int] = None,
    channels: Optional[int] = None,
    pool_blocks: Optional[int] = None,
    seed: int = 0,
) -> Tuple[object, NetConfig]:
    """Build a network, taking its shape from the checkpoint unless overridden.

    Explicit command-line sizes win, so a checkpoint can be inspected with a
    different architecture -- but a shape mismatch is reported, never silently
    swallowed.

    ``backend="auto"`` follows the backend recorded in the checkpoint: weights
    written by one backend cannot be read by the other, so preferring torch just
    because it is importable would break every NumPy-trained checkpoint.
    """
    requested = backend
    cfg_from_disk: Optional[NetConfig] = None
    state = None
    meta = {}
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise SystemExit(f"error: checkpoint not found: {checkpoint}")
        cfg_from_disk, state, meta = load_checkpoint(checkpoint)

    recorded = meta.get("backend") if meta else None
    if requested == "auto" and recorded in ("torch", "numpy"):
        backend = recorded
    else:
        backend = resolve_backend(requested)
        if recorded in ("torch", "numpy") and recorded != backend:
            raise SystemExit(
                f"error: {checkpoint} was written by the {recorded} backend; "
                f"weights are not interchangeable, use --backend {recorded}"
            )

    if cfg_from_disk is not None:
        cfg = cfg_from_disk
    else:
        cfg = NetConfig(
            board_size=size or 9,
            block_count=blocks or 3,
            channels=channels or 24,
            pool_channels=max(8, (channels or 24) // 2),
            pool_blocks=pool_blocks if pool_blocks is not None else 2,
        )
    # explicit overrides
    if size is not None:
        cfg = NetConfig(
            board_size=size,
            block_count=blocks or cfg.block_count,
            channels=channels or cfg.channels,
            pool_channels=max(8, (channels or cfg.channels) // 2),
            pool_blocks=pool_blocks if pool_blocks is not None else cfg.pool_blocks,
        )
    elif blocks is not None or channels is not None or pool_blocks is not None:
        cfg = NetConfig(
            board_size=cfg.board_size,
            block_count=blocks or cfg.block_count,
            channels=channels or cfg.channels,
            pool_channels=max(8, (channels or cfg.channels) // 2),
            pool_blocks=pool_blocks if pool_blocks is not None else cfg.pool_blocks,
        )

    net = build_net(cfg, backend=backend, seed=seed)
    if state:
        net.load_state_dict(state)
    return net, cfg
