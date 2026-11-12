"""Checkpoint format.

A checkpoint is a ``.npz`` holding the weights plus the network shape, so a model
can be loaded without being told its architecture by hand::

    net/<parameter name>   weights (float32)
    config/<field>         NetConfig fields (int scalars)
    meta/<key>             free-form json metadata (training progress, args, ...)

Not recording the shape is how mismatched models get loaded silently, which is
exactly the failure mode this module exists to prevent.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..nn.base import BaseNet
from .config import NetConfig

CONFIG_FIELDS = (
    "board_size",
    "block_count",
    "channels",
    "pool_channels",
    "pool_blocks",
    "policy_pool",
)


def save_checkpoint(path: str, net: BaseNet, meta: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    for k, v in net.state_dict().items():
        payload[f"net/{k}"] = v
    cfg = net.config
    for field in CONFIG_FIELDS:
        value = getattr(cfg, field)
        payload[f"config/{field}"] = np.asarray(int(value) if isinstance(value, bool) else value)
    if meta:
        payload["meta/json"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **payload)


def infer_net_config(state: Dict[str, np.ndarray]) -> Optional[NetConfig]:
    """Recover the architecture from the weights themselves.

    Used for checkpoints written before the shape was recorded, and as a
    cross-check when a stored config is present.
    """
    if "c0_w" not in state or "p_w" not in state:
        return None
    channels = int(state["c0_w"].shape[0])
    blocks = 0
    while f"b{blocks}_c1_w" in state:
        blocks += 1
    pool_channels = max(8, channels // 2)
    pool_blocks = 0
    for i in range(blocks):
        key = f"b{i}_gp_w"
        if key in state:
            pool_blocks += 1
            pool_channels = int(state[key].shape[1]) // 3
    num_moves = int(state["p_w"].shape[0])
    num_points = num_moves - 1
    board_size = int(round(num_points ** 0.5))
    if board_size * board_size != num_points or board_size < 2:
        return None
    return NetConfig(
        board_size=board_size,
        block_count=blocks,
        channels=channels,
        pool_channels=pool_channels,
        pool_blocks=max(1, pool_blocks),
        policy_pool="ph_gp_w" in state,
    )


def load_checkpoint(path: str) -> Tuple[Optional[NetConfig], Dict[str, np.ndarray], Dict[str, Any]]:
    """Return ``(config, weights, meta)``; ``config`` is ``None`` for old files."""
    with np.load(path, allow_pickle=False) as data:
        state = {k[len("net/"):]: data[k] for k in data.files if k.startswith("net/")}
        cfg_values = {
            k[len("config/"):]: data[k].item()
            for k in data.files
            if k.startswith("config/")
        }
        meta: Dict[str, Any] = {}
        if "meta/json" in data.files:
            try:
                meta = json.loads(str(data["meta/json"].item()))
            except (ValueError, TypeError):
                meta = {}
    config = None
    if "board_size" in cfg_values and "channels" in cfg_values:
        config = NetConfig(
            board_size=int(cfg_values["board_size"]),
            block_count=int(cfg_values["block_count"]),
            channels=int(cfg_values["channels"]),
            pool_channels=int(cfg_values["pool_channels"]),
            pool_blocks=int(cfg_values.get("pool_blocks", 2)),
            policy_pool=bool(cfg_values.get("policy_pool", True)),
        )
    else:
        config = infer_net_config(state)
    return config, state, meta
