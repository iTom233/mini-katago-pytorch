"""Incremental analysis (``AnalysisSession``).

The analysis GUI shows a live report, so the tree must survive between calls
and the read-out must convert the negamax backup back to the root player's
point of view.
"""

from __future__ import annotations

import numpy as np

from katago.game import PASS_MOVE, Position, Rules
from katago.inference import AnalysisSession
from katago.nn.base import BaseNet, EvalResult
from katago.nn.config import NetConfig


class ScoreNet(BaseNet):
    """Uniform priors; every position looks ``+4`` and 70% won for the mover."""

    def __init__(self, board_size=9):
        super().__init__(NetConfig(board_size=board_size))

    def predict(self, spatial, global_, legal=None):
        b = spatial.shape[0]
        pol = np.ones((b, self.config.num_moves), dtype=np.float32)
        if legal is not None:
            pol = pol * legal.astype(np.float32)
        pol /= np.maximum(pol.sum(axis=1, keepdims=True), 1e-12)
        v = np.zeros((b, 2), dtype=np.float32)
        v[:, 1] = 0.7
        v[:, 0] = 0.3
        own = np.full((b, self.config.num_points), 0.5, dtype=np.float32)
        score = np.zeros((b, self.config.score_bins), dtype=np.float32)
        score[:, self.config.score_offset + 4] = 1.0
        return EvalResult(pol, v, own, score, pol)

    def train_batch(self, batch, loss_cfg, lr):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None


def _rules(size=9):
    return Rules(size=size, komi=7.5)


def test_session_accumulates_visits_across_steps():
    rules = _rules()
    session = AnalysisSession(ScoreNet(), rules, seed=0)
    session.set_position(Position.empty(rules))
    total = 0
    for _ in range(3):
        report, ran = session.step(24)
        total += ran
        assert report.visits == total, "the tree must be kept between steps"
    assert total == 72


def test_report_before_any_playout_lists_the_priors():
    rules = _rules()
    session = AnalysisSession(ScoreNet(), rules, seed=0)
    session.set_position(Position.empty(rules))
    report = session.report(10)
    assert report.visits == 0
    assert len(report.moves) == 10
    assert all(m.visits == 0 for m in report.moves)
    assert all(m.prior > 0.0 for m in report.moves)


def test_report_is_ranked_by_visits_and_bounded():
    rules = _rules()
    session = AnalysisSession(ScoreNet(), rules, seed=0)
    session.set_position(Position.empty(rules))
    session.step(150)
    report = session.report(10)
    assert 0 < len(report.moves) <= 10
    visits = [m.visits for m in report.moves]
    assert visits == sorted(visits, reverse=True)
    # the table only lists the leaders, the root total covers every child
    assert sum(visits) <= report.visits


def test_report_values_are_from_the_root_player_view():
    """Children are stored from the opponent's view; the read-out flips them.

    With a stub that always says "+4, 70% won for the side to move", a child
    node reads +4/+0.7 from its own side, i.e. -4/30% for the player who just
    moved at the root.
    """
    rules = _rules()
    session = AnalysisSession(ScoreNet(), rules, seed=0)
    session.set_position(Position.empty(rules))
    report, ran = session.step(1)
    assert ran == 1
    visited = [m for m in report.moves if m.visits == 1]
    assert len(visited) == 1, "exactly one child received the single playout"
    move = visited[0]
    assert abs(move.score_mean + 4.0) < 1e-6
    assert abs(move.win_rate - 0.3) < 1e-6
    # the root mixes its own +4/+0.7 evaluation with the negated child value,
    # which cancels exactly after one playout
    assert abs(report.score_mean) < 1e-6
    assert abs(report.win_rate - 0.5) < 1e-6


def test_terminal_position_yields_no_candidates():
    rules = _rules()
    pos = Position.empty(rules).play(PASS_MOVE).play(PASS_MOVE)
    session = AnalysisSession(ScoreNet(), rules, seed=0)
    session.set_position(pos)
    report, ran = session.step(10)
    assert ran == 0
    assert report.visits == 0
    assert report.moves == []
