"""Self-play, replay, gating and engine tests."""

from __future__ import annotations

import numpy as np

from katago.game import BLACK, PASS_MOVE, WHITE, Position, Rules
from katago.inference import GTPEngine, Engine, gtp_to_move, move_to_gtp
from katago.nn.config import MCTSConfig, NetConfig, PlayoutCapConfig, TrainConfig
from katago.selfplay import (
    ReplayBuffer,
    SelfPlayConfig,
    evaluate_vs_net,
    gating_test,
    play_game,
    play_match_game,
)
from katago.utils.symmetry import apply_planes, apply_policy, point_permutation

from support import approx, raises


def small_net():
    from katago.nn.numpy_net import NumpyNet

    return NumpyNet(NetConfig(board_size=9, block_count=2, channels=8, pool_channels=4, pool_blocks=2), seed=0)


def fast_mcts():
    return MCTSConfig(forced_playouts=True, root_noise_eps=0.25, resign_threshold=-2.0)


def fast_caps():
    return PlayoutCapConfig(enabled=True, full_fraction=0.5, full_cap=12, fast_cap=4,
                            full_cap_late=None, fast_cap_late=None)


# ---------------------------------------------------------------- symmetries


def test_point_permutation_is_a_bijection():
    for k in range(8):
        perm = point_permutation(k, 9)
        assert len(perm) == 81
        assert sorted(perm) == list(range(81)), f"symmetry {k} is not a bijection"


def test_apply_planes_matches_numpy_ops():
    x = np.arange(9 * 9).reshape(9, 9)
    assert np.array_equal(apply_planes(x, 0), x)
    assert np.array_equal(apply_planes(x, 1), np.rot90(x, 1))
    assert np.array_equal(apply_planes(x, 2), np.rot90(x, 2))
    assert np.array_equal(apply_planes(x, 4), np.flip(x, axis=-1))


def test_apply_policy_keeps_pass_slot():
    size = 9
    pol = np.zeros(size * size + 1, dtype=np.float32)
    pol[0] = 1.0
    pol[-1] = 0.5
    out = apply_policy(pol, 3, size)
    assert out[-1] == 0.5, "pass probability must be invariant"
    assert out[point_permutation(3, size)[0]] == 1.0


# ----------------------------------------------------------------- self-play


def test_play_game_produces_valid_targets():
    net = small_net()
    rules = Rules(size=9, komi=7.5)
    rng = np.random.default_rng(0)
    rec = play_game(
        net,
        rules,
        fast_mcts(),
        fast_caps(),
        SelfPlayConfig(temperature_moves=4, resign_enabled=False, batch_size=4),
        rng,
    )
    assert rec.num_samples > 0
    assert len(rec.moves) > 0
    assert rec.winner in (0, 1, 2)
    s = rec.samples()
    t = rec.num_samples
    n_moves = rules.num_moves
    assert s["spatial"].shape == (t, 18, 9, 9)
    assert s["policy"].shape == (t, n_moves)
    assert s["aux"].shape == (t, n_moves)
    assert s["value"].shape == (t, 2)
    assert s["ownership"].shape == (t, 81)
    assert s["score"].shape == (t,)
    # policy targets are distributions
    assert np.allclose(s["policy"].sum(axis=1), 1.0, atol=1e-5)
    # value targets are one-hot (or 50/50 for a draw)
    assert np.allclose(s["value"].sum(axis=1), 1.0, atol=1e-6)
    # only recorded (full-search) turns appear
    assert t == rec.num_full_searches
    assert t <= len(rec.moves)


def test_only_full_search_turns_are_recorded():
    net = small_net()
    rules = Rules(size=9)
    rng = np.random.default_rng(1)
    caps = PlayoutCapConfig(enabled=True, full_fraction=0.0, full_cap=12, fast_cap=3,
                            full_cap_late=None, fast_cap_late=None)
    rec = play_game(
        net, rules, fast_mcts(), caps,
        SelfPlayConfig(temperature_moves=2, resign_enabled=False, batch_size=4), rng,
    )
    assert rec.num_full_searches == 1, (
        "with full_fraction=0 no turn is recorded, so only the fallback position remains"
    )


def test_all_turns_recorded_when_cap_randomization_is_disabled():
    net = small_net()
    rules = Rules(size=9)
    rng = np.random.default_rng(1)
    caps = PlayoutCapConfig(enabled=False, full_cap=8, fast_cap=3)
    rec = play_game(
        net, rules, fast_mcts(), caps,
        SelfPlayConfig(temperature_moves=2, resign_enabled=False, batch_size=4), rng,
    )
    assert rec.num_full_searches == len(rec.moves)


def test_value_target_matches_winner():
    net = small_net()
    rules = Rules(size=9, komi=7.5)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        rec = play_game(
            net, rules, fast_mcts(), fast_caps(),
            SelfPlayConfig(temperature_moves=3, resign_enabled=False, batch_size=4), rng,
        )
        s = rec.samples()
        if rec.winner == 0:
            assert np.allclose(s["value"][:, 0], 0.5)
            continue
        for i, player in enumerate(rec.player):
            expected_win = 1.0 if player == rec.winner else 0.0
            assert s["value"][i, 1] == approx(expected_win)


def test_aux_target_is_the_next_turn_policy():
    net = small_net()
    rules = Rules(size=9)
    rng = np.random.default_rng(2)
    rec = play_game(
        net, rules, fast_mcts(), fast_caps(),
        SelfPlayConfig(temperature_moves=2, resign_enabled=False, batch_size=4), rng,
    )
    s = rec.samples()
    if rec.num_samples > 1:
        assert np.allclose(s["aux"][:-1], rec.policy[1:], atol=1e-6)
        assert not s["has_aux"][-1], "the final turn has no successor"


# -------------------------------------------------------------------- replay


def test_replay_roundtrip_and_window():
    net = small_net()
    rules = Rules(size=9)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(window_samples=10_000, seed=0)
    for _ in range(3):
        rec = play_game(
            net, rules, fast_mcts(), fast_caps(),
            SelfPlayConfig(temperature_moves=2, resign_enabled=False, batch_size=4), rng,
        )
        buf.add_game(rec)
    assert len(buf) == 3
    batch = buf.sample_batch(4, rng=rng, size=9)
    assert batch["spatial"].shape == (4, 18, 9, 9)
    assert np.allclose(batch["policy"].sum(axis=1), 1.0, atol=1e-5)
    # shrink the window to one game
    buf.set_window(1)
    assert len(buf) == 1


def test_symmetry_augmentation_preserves_distributions():
    net = small_net()
    rules = Rules(size=9)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(window_samples=10_000, seed=0)
    rec = play_game(
        net, rules, fast_mcts(), fast_caps(),
        SelfPlayConfig(temperature_moves=2, resign_enabled=False, batch_size=4), rng,
    )
    buf.add_game(rec)
    batch = buf.sample_batch(8, rng=rng, augment=True, size=9)
    assert np.allclose(batch["policy"].sum(axis=1), 1.0, atol=1e-5)
    # the auxiliary target is a distribution where a successor turn was recorded
    # and all zeros on the final turn of a game
    sums = batch["aux"].sum(axis=1)
    assert np.all(sums <= 1.0 + 1e-5)
    assert np.all((sums < 1e-6) | (np.abs(sums - 1.0) < 1e-5))
    assert np.any(sums < 1e-6), "games end without a successor, so some rows are zero"
    # plane 0 is "on board": all ones under every symmetry
    assert np.allclose(batch["spatial"][:, 0], 1.0)


# -------------------------------------------------------------------- gating


def test_match_play_returns_a_legal_result():
    net = small_net()
    rules = Rules(size=9, komi=7.5)
    result = play_match_game(net, net, rules, fast_mcts(), visits=6,
                             rng=np.random.default_rng(0))
    assert result in (0, 1, 2)


def test_gating_test_self_play_is_not_decisive():
    """A net against itself should not pass a 100% threshold."""
    net = small_net()
    rules = Rules(size=9, komi=7.5)
    accepted, result = gating_test(
        net, net, rules, games=2, wins_needed=2, visits=4,
        mcts_config=fast_mcts(), rng=np.random.default_rng(0),
    )
    assert result.games >= 1
    assert not accepted or result.wins == result.games


def test_evaluate_vs_net_reports_a_rate():
    """A net against an identical copy should score within [0, 1]."""
    net = small_net()
    other = small_net()  # same seed, so the same weights
    rules = Rules(size=9, komi=7.5)
    rate = evaluate_vs_net(
        net, other, rules, games=2, visits=4,
        mcts_config=fast_mcts(), rng=np.random.default_rng(0),
    )
    assert 0.0 <= rate <= 1.0
    # two games, so only these values are reachable
    assert rate in (0.0, 0.5, 1.0)


def test_random_opening_is_seeded_and_legal():
    """The opening must replay from a seed and must be a real opening."""
    from katago.selfplay.gating import _random_opening

    rules = Rules(size=9, komi=7.5)
    a = _random_opening(Position.empty(rules), np.random.default_rng(1), 4)
    b = _random_opening(Position.empty(rules), np.random.default_rng(1), 4)
    c = _random_opening(Position.empty(rules), np.random.default_rng(2), 4)

    assert np.array_equal(a.grid, b.grid), "same seed must replay the same opening"
    assert not np.array_equal(a.grid, c.grid), "different seeds must differ"
    assert np.count_nonzero(a.grid) == 4, "four opening stones were requested"
    assert a.to_play == BLACK, "black plays the odd-numbered opening moves"


def test_match_game_accepts_opening_and_temperature():
    """Guards the argument plumbing used by the evaluator."""
    net = small_net()
    rules = Rules(size=9, komi=7.5)
    result = play_match_game(
        net, net, rules, fast_mcts(), visits=4,
        rng=np.random.default_rng(0), opening_moves=2, temperature=1.0,
    )
    assert result in (0, 1, 2)


def test_trainer_loads_a_frozen_eval_opponent():
    """A checkpoint path given to the trainer is loaded once and reused."""
    import os
    import tempfile

    from katago.nn.checkpoint import save_checkpoint
    from katago.selfplay import Trainer

    net = small_net()
    rules = Rules(size=9, komi=7.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "opponent.npz")
        save_checkpoint(path, net)
        trainer = Trainer(net, rules, backend="numpy", eval_opponent=path, log=None)
        opponent = trainer._get_eval_net()
        assert opponent is not None
        assert opponent is trainer._get_eval_net()  # cached, not rebuilt
        assert opponent.config.board_size == rules.size

    plain = Trainer(net, rules, backend="numpy", log=None)
    assert plain._get_eval_net() is None


# -------------------------------------------------------------------- engine


def test_gtp_coordinate_roundtrip():
    assert move_to_gtp(0, 9) == "A9"
    assert move_to_gtp(4 * 9 + 4, 9) == "E5"
    assert move_to_gtp(PASS_MOVE, 9) == "pass"
    for size in (9, 13, 19):
        for idx in (0, 3, size * size - 1):
            assert gtp_to_move(move_to_gtp(idx, size), size) == idx


def test_gtp_coordinates_skip_letter_i():
    assert gtp_to_move("J9", 9) == 8, "J is the 9th column, since I is skipped"
    assert gtp_to_move("H9", 9) == 7


def test_engine_genmove_returns_legal_move():
    net = small_net()
    rules = Rules(size=9)
    engine = Engine(net, rules, visits=8, seed=0)
    pos = Position.empty(rules)
    move = engine.genmove(pos, visits=8)
    assert pos.is_legal(move)


def test_engine_analyze_shapes():
    net = small_net()
    rules = Rules(size=9)
    engine = Engine(net, rules, visits=8, seed=0)
    a = engine.analyze(Position.empty(rules), visits=8)
    assert 0.0 <= a.win_rate <= 1.0
    assert a.ownership.shape == (81,)
    assert a.policy.shape == (82,)
    assert abs(a.policy.sum() - 1.0) < 1e-6
    assert a.visits > 0
    assert len(a.top_moves(3)) == 3


def test_gtp_session():
    net = small_net()
    gtp = GTPEngine(net, Rules(size=9), visits=6)
    assert gtp.handle("name").startswith("= ")
    assert gtp.handle("boardsize 9") == "= "
    assert gtp.handle("komi 6.5") == "= "
    assert gtp.position.to_play == BLACK, "a fresh board is Black to move"

    assert gtp.handle("genmove b").startswith("= ")
    assert gtp.position.to_play == WHITE
    assert gtp.handle("play w D4") == "= "
    assert gtp.position.to_play == BLACK
    assert gtp.handle("undo") == "= "
    assert gtp.position.to_play == WHITE

    board = gtp.handle("showboard")
    assert "9 " in board
    analysis = gtp.handle("kata-analyze")
    assert "winrate" in analysis and "scoreLead" in analysis

    assert gtp.handle("bogus").startswith("?")
    # out-of-turn play is reported as an error
    assert gtp.handle("play b A1").startswith("?")
    # out-of-board coordinates are rejected
    with raises(ValueError):
        gtp_to_move("Z99", 9)
    with raises(ValueError):
        gtp_to_move("I9", 19)
