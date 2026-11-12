"""MCTS tests, including the KataGo-specific tricks.

A deterministic stub network keeps these tests fast and independent of any
training: it returns a fixed prior shape and a fixed value.
"""

from __future__ import annotations

import numpy as np

from katago.game import BLACK, PASS_MOVE, WHITE, Position, Rules
from katago.mcts import MCTS, Node, terminal_value_of
from katago.nn.base import BaseNet, EvalResult
from katago.nn.config import MCTSConfig, NetConfig


class StubNet(BaseNet):
    """Uniform priors over legal moves; constant value."""

    def __init__(self, board_size=9, value=0.0, prefer=None):
        super().__init__(NetConfig(board_size=board_size))
        self.value = value
        self.prefer = prefer  # optional (prob vector builder)

    def predict(self, spatial, global_, legal=None):
        b = spatial.shape[0]
        moves = self.config.num_moves
        pol = np.ones((b, moves), dtype=np.float32)
        if legal is not None:
            pol = pol * legal.astype(np.float32)
        pol /= np.maximum(pol.sum(axis=1, keepdims=True), 1e-12)
        v = np.zeros((b, 2), dtype=np.float32)
        v[:, 1] = 0.5 + 0.5 * self.value
        v[:, 0] = 0.5 - 0.5 * self.value
        own = np.full((b, self.config.num_points), 0.5, dtype=np.float32)
        score = np.full((b, self.config.score_bins), 1.0 / self.config.score_bins, dtype=np.float32)
        return EvalResult(policy=pol, value=v, ownership=own, score=score, aux_policy=pol)

    def train_batch(self, batch, loss_cfg, lr):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def empty_pos(size=9, **kwargs):
    return Position.empty(Rules(size=size, **kwargs))


# --------------------------------------------------------------------- basics


def test_root_is_expanded_and_noised():
    net = StubNet()
    mcts = MCTS(net, MCTSConfig(), rng=np.random.default_rng(0))
    pos = empty_pos()
    mcts.set_root(pos, add_noise=True)
    moves, counts = mcts.visit_counts()
    assert len(moves) > 0, "root should have children"
    assert counts.sum() == 0, "no playouts yet"
    priors = np.array([mcts.root.children[m].prior for m in moves])
    assert abs(priors.sum() - 1.0) < 1e-6, "priors must sum to 1 after noise"


def test_dirichlet_noise_changes_priors_but_stays_a_distribution():
    cfg = MCTSConfig()
    a = MCTS(StubNet(), cfg, rng=np.random.default_rng(1))
    b = MCTS(StubNet(), cfg, rng=np.random.default_rng(2))
    pos = empty_pos()
    a.set_root(pos, add_noise=False)
    b.set_root(pos, add_noise=True)
    pa = np.array([a.root.children[m].prior for m in sorted(a.root.children)])
    pb = np.array([b.root.children[m].prior for m in sorted(b.root.children)])
    assert pb.sum() == pytest_approx(1.0, 1e-6)
    assert not np.allclose(pa, pb), "noise should move the priors"
    assert np.all(pb >= 0.0)


def pytest_approx(v, tol):
    class _A:
        def __eq__(self, other):
            return abs(float(other) - v) <= tol

    return _A()


# --------------------------------------------------------------- value sign


def test_backup_flips_sign_correctly():
    net = StubNet(value=1.0)  # every leaf believes "player to move wins"
    mcts = MCTS(net, MCTSConfig(forced_playouts=False), rng=np.random.default_rng(0))
    pos = empty_pos()
    mcts.set_root(pos, add_noise=False)
    root_before = mcts.root.value_sum
    assert mcts.run(1) == 1
    # exactly one child got the playout, with value +1 from its own perspective
    moves, counts = mcts.visit_counts()
    visited = [m for m, c in zip(moves, counts) if c > 0]
    assert len(visited) == 1
    child = mcts.root.children[visited[0]]
    assert child.value_sum == 1.0
    assert abs(child.mean_value - 1.0) < 1e-9
    # the root sees the negation, plus its own initial evaluation
    assert abs(mcts.root.value_sum - (root_before - 1.0)) < 1e-9


def test_terminal_value_from_player_to_move():
    rules = Rules(size=9, komi=0.0)
    grid = np.zeros((9, 9), dtype=np.int8)
    grid[0, 0] = BLACK  # Black has exactly one point, komi 0
    pos = Position(rules=rules, grid=grid, to_play=WHITE)
    pos = pos.play(PASS_MOVE).play(PASS_MOVE)
    assert pos.is_terminal
    assert terminal_value_of(pos) == -1.0, "White to move and lost"
    pos_black = Position(rules=rules, grid=grid, to_play=BLACK)
    pos_black = pos_black.play(PASS_MOVE).play(PASS_MOVE)
    assert terminal_value_of(pos_black) == 1.0


def test_search_stops_at_terminal_root():
    pos = empty_pos().play(PASS_MOVE).play(PASS_MOVE)
    mcts = MCTS(StubNet(), MCTSConfig(), rng=np.random.default_rng(0))
    mcts.set_root(pos, add_noise=False)
    assert mcts.run(10) == 0


# ------------------------------------------------------------ forced playouts


def test_forced_playouts_reach_their_target():
    cfg = MCTSConfig(forced_playouts=True, forced_k=2.0, prune_policy_target=False)
    mcts = MCTS(StubNet(), cfg, rng=np.random.default_rng(0))
    mcts.set_root(empty_pos(), add_noise=False)
    n = 200
    mcts.run(n)
    moves, counts = mcts.visit_counts()
    total = counts.sum()
    targets = mcts._forced_targets
    explored = [m for m, c in zip(moves, counts) if c > 0]
    assert len(explored) >= 8, f"expected broad exploration, got {len(explored)} children"
    short = []
    for m in explored:
        if counts[moves.index(m)] + 1e-9 < min(targets[m], 1.0):
            short.append((m, counts[moves.index(m)], targets[m]))
    assert not short, f"children under their forced-playout target: {short[:3]}"
    assert total == n


class ValueByLastMoveNet(StubNet):
    """Uniform priors, but values every position by *where the last move was*.

    This gives the search a real preference, which is what makes target pruning
    observable: visits that MCTS only spent on bad moves must not leak into the
    training target.
    """

    def predict(self, spatial, global_, legal=None):
        out = super().predict(spatial, global_, legal)
        b = spatial.shape[0]
        # plane 7 is the most recent move (Wu 2020, Table 3)
        top_row = spatial[:, 7, 0, :].sum(axis=1)
        good = (top_row > 0.5).astype(np.float32)
        v = np.zeros((b, 2), dtype=np.float32)
        v[:, 1] = 0.5 + 0.5 * (2.0 * good - 1.0)
        v[:, 0] = 1.0 - v[:, 1]
        return EvalResult(out.policy, v, out.ownership, out.score, out.aux_policy)


def _handmade_tree():
    """Root with three children whose priors, visit counts and values are known."""
    mcts = MCTS(StubNet(), MCTSConfig(forced_playouts=True), rng=np.random.default_rng(0))
    mcts.set_root(empty_pos(), add_noise=False)
    root = mcts.root
    root.children = {}
    spec = [
        # move, prior, visits, mean value from the *child's* perspective
        (0, 0.60, 100, 0.10),
        (1, 0.30, 10, 0.80),
        (2, 0.10, 3, 0.90),
    ]
    for move, prior, visits, mean in spec:
        node = Node(prior=prior, to_play=3 - root.to_play, move=move)
        node.visits = visits
        node.value_sum = mean * visits
        node.is_expanded = True
        root.children[move] = node
    root.visits = sum(s[2] for s in spec)
    mcts._forced_targets = {0: 0.0, 1: 5.0, 2: 2.0}
    return mcts


def test_prune_counts_is_deterministic_and_budget_limited():
    mcts = _handmade_tree()
    moves = sorted(mcts.root.children)
    counts = np.array([mcts.root.children[m].visits for m in moves], dtype=np.float64)
    pruned = mcts._prune_counts(moves, counts.copy())

    # the best move keeps every visit
    assert pruned[0] == 100, "the most-visited child must be untouched"
    # the clearly-bad move loses its whole forced budget (3 -> 1, budget 2)
    assert counts[2] - pruned[2] == 2
    # the middling move is trimmed to where it would still have been selected
    assert 0 < counts[1] - pruned[1] <= 5
    # never below one visit, never negative
    assert np.all(pruned >= 1)
    assert np.all(pruned <= counts)


def test_pruning_is_disabled_by_config():
    cfg = MCTSConfig(forced_playouts=True, prune_policy_target=False)
    mcts = MCTS(StubNet(), cfg, rng=np.random.default_rng(0))
    mcts.set_root(empty_pos(), add_noise=False)
    mcts.run(80)
    assert mcts.policy_target() is not None
    # with prune=False explicitly the raw distribution is returned
    raw = mcts.policy_target(prune=False)
    assert abs(raw.sum() - 1.0) < 1e-9


def test_target_pruning_moves_mass_towards_the_best_move():
    net = ValueByLastMoveNet(9)
    cfg = MCTSConfig(forced_playouts=True, prune_policy_target=True)
    mcts = MCTS(net, cfg, rng=np.random.default_rng(0))
    mcts.set_root(empty_pos(), add_noise=False)
    mcts.run(200)
    raw = mcts.policy_target(prune=False)
    pruned = mcts.policy_target(prune=True)
    assert abs(pruned.sum() - 1.0) < 1e-9
    assert np.argmax(pruned) == np.argmax(raw), "the best move stays the best move"
    # both distributions are normalised, so pruning (which removes counts) can
    # only raise the share of the surviving moves
    assert pruned.max() >= raw.max() - 1e-9
    # and it must never add mass to a move that had zero
    assert np.all((pruned > 0) <= (raw > 0))


def test_single_visit_children_are_pruned_from_the_target():
    mcts = MCTS(StubNet(), MCTSConfig(forced_playouts=False), rng=np.random.default_rng(0))
    mcts.set_root(empty_pos(), add_noise=False)
    mcts.run(30)
    moves, counts = mcts.visit_counts()
    target = mcts.policy_target(prune=True)
    for move, n in zip(moves, counts):
        if n <= 1:
            assert target[move] == 0.0, "a move seen once carries no information"
        else:
            assert target[move] > 0.0


# ------------------------------------------------------------------ behaviour


class CaptureNet(StubNet):
    """Puts almost all prior mass on a chosen move."""

    def __init__(self, board_size, best_move):
        super().__init__(board_size=board_size)
        self.best_move = best_move

    def predict(self, spatial, global_, legal=None):
        out = super().predict(spatial, global_, legal)
        pol = out.policy.copy()
        pol[:] = 0.0
        pol[:, self.best_move] = 1.0
        pol = pol * (legal.astype(np.float32) if legal is not None else 1.0)
        pol /= np.maximum(pol.sum(axis=1, keepdims=True), 1e-12)
        return EvalResult(pol, out.value, out.ownership, out.score, pol)


def test_search_follows_a_sharp_prior():
    best = 4 * 9 + 4
    net = CaptureNet(9, best)
    mcts = MCTS(net, MCTSConfig(forced_playouts=False), rng=np.random.default_rng(0))
    mcts.set_root(empty_pos(), add_noise=False)
    mcts.run(32)
    assert mcts.best_move() == best


def test_policy_target_is_zero_on_illegal_moves():
    mcts = MCTS(StubNet(), MCTSConfig(), rng=np.random.default_rng(0))
    pos = empty_pos()
    pos = pos.play(4 * 9 + 4)  # occupied
    mcts.set_root(pos, add_noise=False)
    mcts.run(40)
    target = mcts.policy_target()
    assert target[4 * 9 + 4] == 0.0, "occupied point must have zero target"
    assert target.sum() == pytest_approx(1.0, 1e-9)


def test_temperature_sampling_eventually_picks_all_moves():
    mcts = MCTS(StubNet(), MCTSConfig(), rng=np.random.default_rng(0))
    mcts.set_root(empty_pos(), add_noise=False)
    mcts.run(60)
    picks = {mcts.sample_move(temperature=1.0) for _ in range(200)}
    assert len(picks) > 1, "temperature sampling should explore"
    greedy = {mcts.best_move(temperature=0.0) for _ in range(20)}
    assert len(greedy) == 1, "greedy read-out must be deterministic"


def test_batched_search_matches_sequential_visits():
    cfg = MCTSConfig(forced_playouts=False)
    a = MCTS(StubNet(), cfg, batch_size=1, rng=np.random.default_rng(0))
    b = MCTS(StubNet(), cfg, batch_size=8, rng=np.random.default_rng(0))
    pos = empty_pos()
    a.set_root(pos, add_noise=False)
    b.set_root(pos, add_noise=False)
    a.run(64)
    b.run(64)
    assert a.root.visits == b.root.visits, "same number of playouts should be recorded"


def test_principal_variation_is_a_legal_sequence():
    mcts = MCTS(StubNet(), MCTSConfig(), rng=np.random.default_rng(0))
    pos = empty_pos()
    mcts.set_root(pos, add_noise=False)
    mcts.run(80)
    pv = mcts.principal_variation(4)
    assert len(pv) >= 1
    cur = pos
    for move in pv:
        assert cur.is_legal(move), f"PV move {move} is illegal"
        cur = cur.play(move)
