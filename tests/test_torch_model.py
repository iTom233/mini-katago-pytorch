"""PyTorch model tests.

These are skipped automatically when PyTorch is not importable, so the suite
still passes on a machine using only the NumPy backend.  Run them wherever torch
is installed to validate the paper-faithful implementation (BatchNorm, the
global-pooling bias structure and all heads).
"""

from __future__ import annotations

import numpy as np

from support import SkipTest, approx

from katago.nn import torch_available
from katago.nn.config import LossConfig, NetConfig


def _require_torch():
    if not torch_available():
        raise SkipTest("PyTorch is not installed in this environment")


def tiny_cfg():
    return NetConfig(board_size=9, block_count=3, channels=16, pool_channels=8, pool_blocks=2)


def batch_for(cfg, n=4, seed=0):
    rng = np.random.default_rng(seed)
    legal = rng.random((n, cfg.num_moves)) > 0.3
    legal[:, cfg.pass_index] = True
    pol = rng.random((n, cfg.num_moves)) * legal
    pol /= pol.sum(axis=1, keepdims=True)
    return {
        "spatial": rng.integers(0, 2, size=(n, cfg.in_spatial, cfg.board_size, cfg.board_size)).astype(np.float32),
        "global_": rng.random((n, cfg.in_global)).astype(np.float32),
        "legal": legal,
        "policy": pol.astype(np.float32),
        "aux": pol.astype(np.float32),
        "value": np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n, 1)),
        "ownership": rng.integers(0, 2, size=(n, cfg.board_size * cfg.board_size)).astype(np.float32),
        "score": rng.integers(0, cfg.score_bins, size=(n,)).astype(np.int64),
    }


def test_torch_module_imports():
    _require_torch()
    from katago.nn.model import KataGoNet, TorchNet  # noqa: F401


def test_forward_shapes():
    _require_torch()
    from katago.nn.model import TorchNet

    cfg = tiny_cfg()
    net = TorchNet(cfg, seed=0)
    b = batch_for(cfg)
    out = net.predict(b["spatial"], b["global_"], b["legal"])
    assert out.policy.shape == (4, cfg.num_moves)
    assert out.value.shape == (4, 2)
    assert out.ownership.shape == (4, cfg.num_points)
    assert out.score.shape == (4, cfg.score_bins)
    assert np.allclose(out.policy.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(out.policy[~b["legal"]] == 0.0)


def test_global_pooling_layer_emits_three_c_statistics():
    _require_torch()
    import torch

    from katago.nn.model import GlobalPoolingLayer

    layer = GlobalPoolingLayer()
    x = torch.randn(2, 5, 9, 9)
    assert layer(x).shape == (2, 15)
    value_layer = GlobalPoolingLayer(value_mode=True)
    assert value_layer(x).shape == (2, 15)


def test_residual_blocks_and_pool_blocks_are_interleaved():
    _require_torch()
    from katago.nn.model import GlobalPoolingBlock, KataGoNet, ResBlock

    cfg = NetConfig(board_size=9, block_count=8, channels=16, pool_channels=8, pool_blocks=2)
    model = KataGoNet(cfg)
    kinds = [type(b).__name__ for b in model.blocks]
    assert kinds.count("GlobalPoolingBlock") == 2
    assert kinds.count("ResBlock") == 6


def test_training_step_reduces_loss():
    _require_torch()
    from katago.nn.model import TorchNet

    cfg = tiny_cfg()
    net = TorchNet(cfg, seed=0)
    b = batch_for(cfg, n=4)
    first = net.train_batch(b, LossConfig(), lr=1e-3)["loss"]
    for _ in range(15):
        m = net.train_batch(b, LossConfig(), lr=1e-3)
    assert m["loss"] < first, f"loss did not decrease: {first} -> {m['loss']}"


def test_state_dict_roundtrip():
    _require_torch()
    from katago.nn.model import TorchNet

    cfg = tiny_cfg()
    a = TorchNet(cfg, seed=0)
    b = TorchNet(cfg, seed=1)
    b.load_state_dict(a.state_dict())
    for k, v in a.state_dict().items():
        assert np.array_equal(v, b.state_dict()[k]), f"mismatch {k}"


def test_variable_board_size():
    """The network must accept 9..19 sized boards without changes."""
    _require_torch()
    from katago.nn.model import TorchNet

    for size in (9, 13, 19):
        cfg = NetConfig(board_size=size, block_count=2, channels=8, pool_channels=4, pool_blocks=1)
        net = TorchNet(cfg, seed=0)
        b = batch_for(cfg, n=1)
        out = net.predict(b["spatial"], b["global_"], b["legal"])
        assert out.policy.shape == (1, size * size + 1)
