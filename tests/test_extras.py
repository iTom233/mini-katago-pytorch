"""Saved models, SGF export, the stone click and the widening schedule."""

from __future__ import annotations

import array
import io
import os
import shutil
import sys
import wave

import numpy as np

from katago.game import PASS_MOVE, Position, Rules
from katago.nn.config import MCTSConfig, NetConfig, PlayoutCapConfig, TrainConfig
from katago.nn.numpy_net import NumpyNet
from katago.selfplay.trainer import Trainer
from katago.utils.sgf import to_sgf
from katago.utils.sound import (
    ASSET_PATH,
    CLICK_SAMPLE_RATE,
    CLICK_SECONDS,
    prepare_click,
    render_click,
)


def quick_caps() -> PlayoutCapConfig:
    return PlayoutCapConfig(enabled=True, full_fraction=0.5, full_cap=16, fast_cap=6,
                            full_cap_late=None, fast_cap_late=None)


def quick_mcts() -> MCTSConfig:
    return MCTSConfig(resign_threshold=-2.0)


def test_sgf_roundtrip_shape():
    size = 9
    moves = [0, 1, 2, PASS_MOVE, 4 * 9 + 4]
    sgf = to_sgf(moves, size, komi=7.5)
    assert sgf.startswith("(;GM[1]FF[4]")
    assert "SZ[9]" in sgf
    assert "KM[7.5]" in sgf
    assert sgf.count(";B[") == 3, "black plays moves 0, 2, 4"
    assert sgf.count(";W[") == 2
    assert ";W[]" in sgf, "pass is written as an empty vertex (White's move here)"
    assert sgf.rstrip().endswith(")")


def test_sgf_coordinates():
    # (0,0) is the top-left: column a, row a
    assert ";B[aa]" in to_sgf([0], 9)
    # (8,8) is bottom-right
    assert ";B[ii]" in to_sgf([8 * 9 + 8], 9)


def test_sgf_result_and_comment():
    sgf = to_sgf([0], 9, result="B+7.5", comment="self-play")
    assert "RE[B+7.5]" in sgf
    assert "C[self-play]" in sgf


def test_bundled_stone_click_is_ready_to_play():
    # winsound needs uncompressed PCM; a wrapped mp3 would fail at play time
    assert os.path.exists(ASSET_PATH), f"missing recorded click: {ASSET_PATH}"
    with wave.open(ASSET_PATH) as fh:
        assert fh.getcomptype() == "NONE", "the click must be uncompressed PCM"
        assert fh.getsampwidth() == 2, "expected 16-bit samples"
        seconds = fh.getnframes() / fh.getframerate()
    assert 0.1 < seconds < 1.5, f"recorded click is {seconds:.2f}s"
    assert prepare_click() == ASSET_PATH, "the recording should win over the fallback"


def test_fallback_click_is_audible_and_decays():
    # only used when katago/assets/stone_click.wav is missing
    data = render_click()
    with wave.open(io.BytesIO(data)) as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getframerate() == CLICK_SAMPLE_RATE
        count = fh.getnframes()
        raw = fh.readframes(count)
    assert count == int(CLICK_SAMPLE_RATE * CLICK_SECONDS)
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(abs(s) for s in samples)
    assert 8000 < peak < 32767, f"click peak {peak} should be loud but not clipped"
    ten_ms = int(CLICK_SAMPLE_RATE * 0.01)
    head = sum(abs(s) for s in samples[:ten_ms]) / ten_ms
    tail = sum(abs(s) for s in samples[-2 * ten_ms:]) / (2 * ten_ms)
    assert head > tail * 20, f"no decay: head {head:.0f} vs tail {tail:.0f}"
    # the "tock" must still be ringing 40-100 ms in; a tiny tick would be gone
    mid_lo = int(CLICK_SAMPLE_RATE * 0.04)
    mid_hi = int(CLICK_SAMPLE_RATE * 0.10)
    mid = sum(abs(s) for s in samples[mid_lo:mid_hi]) / (mid_hi - mid_lo)
    assert mid > head * 0.3, f"click dies too fast: head {head:.0f} mid {mid:.0f}"


def test_progressive_widening_promotes_a_larger_net():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work = os.path.join(root, ".tmp_widen_test")
    os.makedirs(work, exist_ok=True)
    try:
        net = NumpyNet(
            NetConfig(board_size=9, block_count=1, channels=8, pool_channels=4, pool_blocks=1),
            seed=0,
        )
        cfg = TrainConfig(
            batch_size=4, lr=5e-3, lr_warmup_samples=0, lr_warmup=5e-3,
            lr_decay_at_samples=-1, window_start=4000, window_max=20000,
            swa_snapshot_samples=10**9, gating_enabled=False,
            widening=[(2, 12)],
        )
        trainer = Trainer(net, Rules(size=9), train_config=cfg, mcts_config=quick_mcts(),
                          cap_config=quick_caps(), backend="numpy", seed=0, log=lambda m: None)
        trainer.selfplay_config.batch_size = 4
        trainer.selfplay_config.resign_enabled = False
        assert trainer.train_net.config.block_count == 1
        trainer.train(iterations=3, games_per_iteration=1, steps_per_iteration=8)
        # the candidate was started on iteration 1 and should have been promoted
        assert trainer.train_net.config.block_count == 2, "widening did not promote"
        assert trainer.train_net.config.channels == 12
        assert trainer.selfplay_net.config.block_count == 2
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_widening_schedule_with_equal_size_is_skipped():
    net = NumpyNet(
        NetConfig(board_size=9, block_count=2, channels=8, pool_channels=4, pool_blocks=1),
        seed=0,
    )
    cfg = TrainConfig(batch_size=4, lr=5e-3, lr_warmup_samples=0, lr_warmup=5e-3,
                      swa_snapshot_samples=10**9, gating_enabled=False,
                      widening=[(2, 8)])
    trainer = Trainer(net, Rules(size=9), train_config=cfg, mcts_config=quick_mcts(),
                      cap_config=quick_caps(), backend="numpy", seed=0, log=lambda m: None)
    trainer.selfplay_config.batch_size = 4
    trainer.selfplay_config.resign_enabled = False
    trainer.train(iterations=1, games_per_iteration=1, steps_per_iteration=4)
    assert trainer._next_net is None, "no candidate should be created for an identical size"
    assert trainer._widen_schedule == [], "the schedule entry should be consumed"
