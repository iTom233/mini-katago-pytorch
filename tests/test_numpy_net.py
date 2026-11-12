"""Architecture and training tests for the NumPy network backend."""

from __future__ import annotations

import numpy as np

from katago.game import BLACK, Position, Rules, encode_position
from katago.nn.autograd import Tensor
from katago.nn.config import LossConfig, NetConfig
from katago.nn.numpy_net import NumpyNet


def tiny_cfg():
    return NetConfig(board_size=9, block_count=3, channels=16, pool_channels=8, pool_blocks=2)


def random_batch(cfg, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    spatial = rng.integers(0, 2, size=(batch, cfg.in_spatial, cfg.board_size, cfg.board_size)).astype(np.float32)
    glob = rng.random((batch, cfg.in_global)).astype(np.float32)
    legal = rng.random((batch, cfg.num_moves)) > 0.3
    legal[:, cfg.pass_index] = True
    pol = rng.random((batch, cfg.num_moves)) * legal
    pol /= pol.sum(axis=1, keepdims=True)
    aux = pol.copy()
    value = np.zeros((batch, 2), dtype=np.float32)
    value[:, 1] = 1.0
    own = rng.integers(0, 2, size=(batch, cfg.num_points)).astype(np.float32)
    score = rng.integers(0, cfg.score_bins, size=(batch,)).astype(np.int64)
    return {
        "spatial": spatial,
        "global_": glob,
        "legal": legal,
        "policy": pol.astype(np.float32),
        "aux": aux.astype(np.float32),
        "value": value,
        "ownership": own,
        "score": score,
    }


def test_forward_shapes_and_probabilities():
    cfg = tiny_cfg()
    net = NumpyNet(cfg, seed=0)
    batch = random_batch(cfg)
    out = net.predict(batch["spatial"], batch["global_"], batch["legal"])
    assert out.policy.shape == (2, cfg.num_moves)
    assert out.value.shape == (2, 2)
    assert out.ownership.shape == (2, cfg.num_points)
    assert out.score.shape == (2, cfg.score_bins)
    assert out.aux_policy.shape == (2, cfg.num_moves)
    # probabilities
    assert np.allclose(out.policy.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(out.value.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(out.score.sum(axis=1), 1.0, atol=1e-5)
    # illegal moves get exactly zero prior
    assert np.all(out.policy[~batch["legal"]] == 0.0)
    assert np.all((out.ownership >= 0.0) & (out.ownership <= 1.0))


def test_pooling_block_indices_are_evenly_spaced():
    cfg = NetConfig(board_size=9, block_count=20, channels=32, pool_blocks=3)
    idx = cfg.pool_block_indices
    assert len(idx) == 3
    assert idx == sorted(idx)
    assert all(0 <= i < 20 for i in idx)


def test_state_dict_roundtrip():
    cfg = tiny_cfg()
    a = NumpyNet(cfg, seed=0)
    b = NumpyNet(cfg, seed=1)
    assert a.state_dict()["p_w"].shape == b.state_dict()["p_w"].shape
    b.load_state_dict(a.state_dict())
    for k, v in a.state_dict().items():
        assert np.array_equal(v, b.state_dict()[k]), f"mismatch in {k}"


def test_training_reduces_loss_on_a_fixed_batch():
    cfg = tiny_cfg()
    net = NumpyNet(cfg, seed=0)
    batch = random_batch(cfg, batch=4)
    loss_cfg = LossConfig()
    # per-sample lr; batch 4 means the mean-gradient multiplier is 4 * 0.003
    first = net.train_batch(batch, loss_cfg, lr=0.003)["loss"]
    for _ in range(20):
        metrics = net.train_batch(batch, loss_cfg, lr=0.003)
    last = metrics["loss"]
    assert last < first, f"loss did not drop: {first:.4f} -> {last:.4f}"
    assert last < first * 0.95, f"loss barely moved: {first:.4f} -> {last:.4f}"


def policy_only_loss_cfg():
    """Loss weights that isolate the policy term."""
    return LossConfig(c_game=0.0, w_aux_policy=0.0, w_score_pdf=0.0, w_score_cdf=0.0,
                      w_ownership_scale=0.0)


def test_training_improves_policy_accuracy_on_real_positions():
    """Fit the net to a deterministic policy on real 9x9 positions."""
    cfg = tiny_cfg()
    net = NumpyNet(cfg, seed=0)
    rules = Rules(size=9)
    rng = np.random.default_rng(3)
    positions = []
    pos = Position.empty(rules)
    for _ in range(16):
        legal = [m for m in pos.legal_moves() if m != pos.pass_index]
        move = int(rng.choice(legal))
        positions.append(pos)
        pos = pos.play(move)

    spatial, glob, legal, target = [], [], [], []
    for pos in positions:
        s, g = encode_position(pos)
        m = pos.legal_mask()
        best = int(np.flatnonzero(m)[0])
        t = np.zeros(cfg.num_moves, dtype=np.float32)
        t[best] = 1.0
        spatial.append(s)
        glob.append(g)
        legal.append(m)
        target.append(t)
    batch = {
        "spatial": np.stack(spatial),
        "global_": np.stack(glob),
        "legal": np.stack(legal),
        "policy": np.stack(target),
        "value": np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(positions), 1)),
    }

    def accuracy() -> float:
        out = net.predict(batch["spatial"], batch["global_"], batch["legal"])
        return float((out.policy.argmax(axis=1) == batch["policy"].argmax(axis=1)).mean())

    before = accuracy()
    for _ in range(40):
        net.train_batch(batch, policy_only_loss_cfg(), lr=1e-3)
    after = accuracy()
    assert after >= before, f"accuracy regressed: {before} -> {after}"
    assert after > 0.5, f"failed to fit the target policy (accuracy {after})"
