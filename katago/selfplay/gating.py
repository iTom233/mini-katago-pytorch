"""Gating tests and match play between two networks.

KataGo replaces the self-play network only when the candidate passes a gating
test: "candidate nets must pass a gating test by winning at least 100 out of 200
test games against the current net" (Wu 2020, Section 2).  This is stricter than
AlphaGo Zero's "55% of 400 games" and keeps the self-play data distribution from
regressing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..game import PASS_MOVE, Position, Rules
from ..mcts import MCTS
from ..nn.base import BaseNet
from ..nn.config import MCTSConfig


@dataclass
class MatchResult:
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.games == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.wins}-{self.losses}-{self.draws} ({self.win_rate:.1%})"


def _quiet_config(base: MCTSConfig) -> MCTSConfig:
    """Match settings: no root noise, no resignation, deterministic read-out."""
    return MCTSConfig(
        c_puct=base.c_puct,
        c_fpu=base.c_fpu,
        forced_playouts=base.forced_playouts,
        forced_k=base.forced_k,
        prune_policy_target=False,
        root_noise_eps=0.0,
        resign_threshold=-2.0,
        resign_min_visits=10**9,
    )


def _random_opening(
    position: Position, rng: np.random.Generator, moves: int
) -> Position:
    """Play up to ``moves`` uniformly random legal moves from ``position``."""
    for _ in range(max(0, moves)):
        if position.is_terminal:
            break
        legal = position.legal_moves()
        if not legal:
            break
        position = position.play(int(rng.choice(legal)))
    return position


def play_match_game(
    black: BaseNet,
    white: BaseNet,
    rules: Rules,
    mcts_config: Optional[MCTSConfig] = None,
    visits: int = 32,
    max_moves: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    opening_moves: int = 0,
    temperature: float = 0.0,
) -> int:
    """Play one game; returns 1 if Black won, 2 if White won, 0 for a draw.

    Both nets search deterministically (``add_noise=False`` and a greedy
    read-out), so without extra randomness every game between the same two
    weight sets is a byte-for-byte replay and a large match carries almost no
    statistical signal.

    ``opening_moves`` plays that many uniformly random legal moves first, which
    is the cheapest way to decorrelate the games: four random stones on 9x9
    already give tens of millions of distinct starting positions, and both
    sides get the same treatment so the comparison stays fair.

    ``temperature > 0`` additionally samples every move from the visit counts
    instead of taking the argmax -- more variety, slightly weaker play.
    """
    cfg = _quiet_config(mcts_config or MCTSConfig())
    rng = rng or np.random.default_rng()
    position = _random_opening(Position.empty(rules), rng, opening_moves)
    limit = max_moves or rules.max_moves
    moves = min(opening_moves, rules.max_moves)
    while not position.is_terminal and moves < limit:
        net = black if position.to_play == 1 else white
        mcts = MCTS(net, cfg, batch_size=8, rng=rng)
        mcts.set_root(position, add_noise=False)
        mcts.run(visits)
        move = mcts.sample_move(temperature)
        position = position.play(move)
        moves += 1
    if position.is_draw:
        return 0
    score = position.score()
    if abs(score) < 1e-9:
        return 0
    return 1 if score > 0 else 2


def gating_test(
    candidate: BaseNet,
    current: BaseNet,
    rules: Rules,
    games: int = 20,
    wins_needed: Optional[int] = None,
    visits: int = 32,
    mcts_config: Optional[MCTSConfig] = None,
    rng: Optional[np.random.Generator] = None,
    progress=None,
    opening_moves: int = 0,
    temperature: float = 0.0,
) -> Tuple[bool, MatchResult]:
    """Play ``games`` games alternating colours; accept if ``wins_needed`` reached.

    Defaults to the strictest interpretation of Wu 2020 (50% + ties broken by the
    incumbent, i.e. the candidate must not lose).  ``opening_moves`` and
    ``temperature`` are forwarded to :func:`play_match_game`.
    """
    rng = rng or np.random.default_rng()
    if wins_needed is None:
        wins_needed = (games + 1) // 2
    wins = losses = draws = 0
    for g in range(games):
        if g % 2 == 0:
            result = play_match_game(
                candidate, current, rules, mcts_config, visits, rng=rng,
                opening_moves=opening_moves, temperature=temperature,
            )
            candidate_won = result == 1
            candidate_lost = result == 2
        else:
            result = play_match_game(
                current, candidate, rules, mcts_config, visits, rng=rng,
                opening_moves=opening_moves, temperature=temperature,
            )
            candidate_won = result == 2
            candidate_lost = result == 1
        if result == 0:
            draws += 1
        elif candidate_won:
            wins += 1
        elif candidate_lost:
            losses += 1
        if progress is not None:
            progress(g + 1, wins, losses, draws)
        # early stop once the outcome is decided
        remaining = games - (g + 1)
        if wins >= wins_needed:
            return True, MatchResult(wins, losses, draws)
        if wins + remaining + draws < wins_needed:
            return False, MatchResult(wins, losses, draws)
    return wins >= wins_needed, MatchResult(wins, losses, draws)


def evaluate_vs_net(
    net: BaseNet,
    opponent: BaseNet,
    rules: Rules,
    games: int = 20,
    visits: int = 32,
    mcts_config: Optional[MCTSConfig] = None,
    rng: Optional[np.random.Generator] = None,
    opening_moves: int = 0,
    temperature: float = 0.0,
) -> float:
    """Win rate of ``net`` against another network, alternating colours.

    Both sides search with the same ``visits`` budget, so the score compares the
    two weight sets rather than their search settings.  Draws count as half a
    win, matching :func:`evaluate_vs_random`.

    ``opening_moves`` matters here: deterministic search would make all
    ``games`` a replay of the same two positions (one per colour), so the win
    rate could only ever come out as 0%, 50% or 100%.
    """
    rng = rng or np.random.default_rng()
    wins = 0.0
    for g in range(games):
        net_is_black = g % 2 == 0
        if net_is_black:
            result = play_match_game(
                net, opponent, rules, mcts_config, visits, rng=rng,
                opening_moves=opening_moves, temperature=temperature,
            )
            wins += 1.0 if result == 1 else (0.5 if result == 0 else 0.0)
        else:
            result = play_match_game(
                opponent, net, rules, mcts_config, visits, rng=rng,
                opening_moves=opening_moves, temperature=temperature,
            )
            wins += 1.0 if result == 2 else (0.5 if result == 0 else 0.0)
    return wins / games


def evaluate_vs_random(
    net: BaseNet,
    rules: Rules,
    games: int = 20,
    visits: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Win rate of ``net`` against a uniformly random legal-move player."""
    rng = rng or np.random.default_rng()

    class _Random:
        def best_move(self, position):
            return int(rng.choice(position.legal_moves()))

    cfg = _quiet_config(MCTSConfig())
    wins = 0
    for g in range(games):
        net_is_black = g % 2 == 0
        position = Position.empty(rules)
        moves = 0
        while not position.is_terminal and moves < rules.max_moves:
            net_turn = (position.to_play == 1) == net_is_black
            if net_turn:
                mcts = MCTS(net, cfg, batch_size=8, rng=rng)
                mcts.set_root(position, add_noise=False)
                mcts.run(visits)
                move = mcts.best_move(0.0)
            else:
                move = int(rng.choice(position.legal_moves()))
            position = position.play(move)
            moves += 1
        if position.is_draw:
            wins += 0.5
            continue
        score = position.score()
        if abs(score) < 1e-9:
            wins += 0.5
            continue
        black_won = score > 0
        if black_won == net_is_black:
            wins += 1
    return wins / games
