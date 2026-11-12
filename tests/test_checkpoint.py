"""Checkpoint format: architecture round-trip, inference, and mismatch rejection."""

from __future__ import annotations

import os
import shutil

import numpy as np

from support import raises

from katago.nn.checkpoint import infer_net_config, load_checkpoint, save_checkpoint
from katago.nn.config import NetConfig
from katago.nn.numpy_net import NumpyNet


def _tmpdir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".tmp_ckpt")
    os.makedirs(path, exist_ok=True)
    return path


def test_roundtrip_preserves_config_and_weights():
    work = _tmpdir()
    try:
        cfg = NetConfig(board_size=9, block_count=2, channels=12, pool_channels=6, pool_blocks=2)
        net = NumpyNet(cfg, seed=0)
        path = os.path.join(work, "m.npz")
        save_checkpoint(path, net, meta={"hello": "world"})
        cfg2, state, meta = load_checkpoint(path)
        assert cfg2 is not None
        assert cfg2.board_size == 9
        assert cfg2.block_count == 2
        assert cfg2.channels == 12
        assert cfg2.pool_channels == 6
        assert meta["hello"] == "world"
        other = NumpyNet(cfg2, seed=1)
        other.load_state_dict(state)
        for k, v in net.state_dict().items():
            assert np.array_equal(v, other.state_dict()[k]), k
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_infer_config_from_weights_only():
    cfg = NetConfig(board_size=13, block_count=3, channels=20, pool_channels=10, pool_blocks=2)
    net = NumpyNet(cfg, seed=0)
    inferred = infer_net_config(net.state_dict())
    assert inferred is not None
    assert inferred.board_size == 13
    assert inferred.block_count == 3
    assert inferred.channels == 20
    assert inferred.pool_channels == 10
    assert inferred.pool_blocks == 2
    assert inferred.policy_pool is True


def test_load_rejects_a_mismatched_architecture():
    cfg = NetConfig(board_size=9, block_count=2, channels=12, pool_channels=6, pool_blocks=2)
    net = NumpyNet(cfg, seed=0)
    state = net.state_dict()
    other = NumpyNet(
        NetConfig(board_size=9, block_count=3, channels=24, pool_channels=12, pool_blocks=2),
        seed=1,
    )
    with raises(ValueError):
        other.load_state_dict(state)


def test_load_tolerates_extra_keys_when_not_strict():
    cfg = NetConfig(board_size=9, block_count=1, channels=8, pool_channels=4, pool_blocks=1)
    net = NumpyNet(cfg, seed=0)
    state = dict(net.state_dict())
    state["not_a_parameter"] = np.zeros(3, dtype=np.float32)
    net.load_state_dict(state, strict=False)
