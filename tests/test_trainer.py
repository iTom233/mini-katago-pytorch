"""Training-loop tests: replay growth, SWA, gating and end-to-end learning."""

from __future__ import annotations

import numpy as np

from katago.game import Rules
from katago.nn.config import MCTSConfig, NetConfig, PlayoutCapConfig, TrainConfig
from katago.nn.numpy_net import NumpyNet
from katago.selfplay import ReplayBuffer, Trainer
from katago.selfplay.trainer import IterationStats


def tiny():
    return NumpyNet(
        NetConfig(board_size=9, block_count=2, channels=8, pool_channels=4, pool_blocks=2),
        seed=0,
    )


def fast_train_config() -> TrainConfig:
    return TrainConfig(
        batch_size=8,
        lr=5e-3,
        lr_warmup_samples=0,
        lr_warmup=5e-3,
        lr_decay_at_samples=-1,
        window_start=4000,
        window_max=20000,
        swa_snapshot_samples=10_000,  # effectively disabled in a short run
        swa_snapshots=4,
        gating_games=2,
        gating_wins=2,
    )


def fast_pipeline():
    net = tiny()
    rules = Rules(size=9, komi=7.5)
    trainer = Trainer(
        net,
        rules,
        train_config=fast_train_config(),
        mcts_config=MCTSConfig(resign_threshold=-2.0),
        cap_config=PlayoutCapConfig(enabled=True, full_fraction=0.5, full_cap=32, fast_cap=12,
                                    full_cap_late=None, fast_cap_late=None),
        backend="numpy",
        seed=0,
        log=lambda msg: None,
    )
    trainer.selfplay_config.batch_size = 8
    trainer.selfplay_config.temperature_moves = 8
    trainer.selfplay_config.resign_enabled = False
    return trainer


def test_trainer_runs_and_records_history():
    trainer = fast_pipeline()
    hist = trainer.train(iterations=2, games_per_iteration=1, steps_per_iteration=6)
    assert len(hist) == 2
    assert all(isinstance(h, IterationStats) for h in hist)
    assert trainer.total_samples_produced > 0
    assert trainer.total_samples_trained == 12 * 8, "12 steps of batch 8"
    assert len(trainer.replay) >= 2


def test_learning_rate_schedule():
    cfg = TrainConfig(lr=6e-5, lr_warmup=2e-5, lr_warmup_samples=1000,
                      lr_decay_at_samples=5000, lr_decay=6e-6)
    assert cfg.learning_rate(0) == 2e-5
    assert cfg.learning_rate(999) == 2e-5
    assert cfg.learning_rate(1000) == 6e-5
    assert cfg.learning_rate(5000) == 6e-6


def test_replay_window_grows_then_caps():
    cfg = TrainConfig(window_start=1000, window_max=10_000)
    assert cfg.window_size(0) == 1000
    assert cfg.window_size(500) == 1000
    assert cfg.window_size(5000) == 5000
    assert cfg.window_size(10**9) == 10_000


def test_swa_averages_snapshots():
    trainer = fast_pipeline()
    net = trainer.train_net
    base = net.state_dict()
    key = sorted(base)[0]
    # feed four synthetic snapshots so the EMA is well defined
    for i in range(4):
        snap = {k: v.copy() for k, v in base.items()}
        snap[key] = base[key] + float(i + 1)
        trainer.snapshots.append(snap)
    cand = trainer._swa_candidate()
    assert cand is not None
    # seeded with the oldest snapshot; weights sum to 1 and the newest dominates
    d = trainer.train_config.swa_decay
    weights = [d ** 3, (1 - d) * d ** 2, (1 - d) * d, (1 - d)]
    expected = sum(w * (i + 1) for i, w in enumerate(weights))
    assert abs(sum(weights) - 1.0) < 1e-12
    got = float(cand[key].mean() - base[key].mean())
    assert abs(got - expected) < 1e-4, f"EMA shift {got} != expected {expected}"


def test_trainer_save_load_roundtrip():
    import os
    import shutil

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work = os.path.join(root, ".tmp_ckpt_test")
    os.makedirs(work, exist_ok=True)
    try:
        trainer = fast_pipeline()
        trainer.train(iterations=1, games_per_iteration=1, steps_per_iteration=4)
        path = os.path.join(work, "ckpt.npz")
        trainer.save(path)
        other = fast_pipeline()
        other.load(path)
        a = trainer.selfplay_net.state_dict()
        b = other.selfplay_net.state_dict()
        for k in a:
            assert np.array_equal(a[k], b[k]), f"weights differ after save/load: {k}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_training_reduces_policy_loss_over_iterations():
    """End-to-end signal: the training loss must come down as the loop runs."""
    trainer = fast_pipeline()
    hist = trainer.train(iterations=5, games_per_iteration=1, steps_per_iteration=25)
    losses = [h.loss for h in hist]
    assert np.all(np.isfinite(losses)), f"non-finite loss: {losses}"
    assert hist[-1].loss < hist[0].loss, f"loss rose: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert trainer.total_samples_trained == 5 * 25 * 8
