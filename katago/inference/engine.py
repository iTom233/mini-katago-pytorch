"""Playing engine: move generation and position analysis.

This is the inference side of the project.  ``Engine.analyze`` exposes the same
quantities KataGo's analysis mode reports -- policy, win rate, score belief and
ownership -- and ``Engine.genmove`` is what the GTP front-end calls.
``AnalysisSession`` keeps its tree between calls so the GUI can show a live
report that refines as visits accumulate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..game import PASS_MOVE, Position, Rules
from ..game.features import encode_position
from ..mcts import MCTS
from ..nn.base import BaseNet
from ..nn.config import MCTSConfig


@dataclass
class Analysis:
    """Result of analysing one position."""

    visits: int
    win_rate: float  # for the player to move, in [0, 1]
    score_mean: float  # expected score for the player to move
    ownership: np.ndarray  # (S*S,) share owned by the player to move
    policy: np.ndarray  # (M,) visit distribution
    move: int
    principal_variation: List[int]

    def top_moves(self, k: int = 5) -> List[Tuple[int, float]]:
        order = np.argsort(self.policy)[::-1]
        out = []
        for m in order:
            if self.policy[m] <= 0:
                continue
            out.append((int(m), float(self.policy[m])))
            if len(out) >= k:
                break
        return out


class Engine:
    """Wraps a network and an MCTS configuration into a playable engine."""

    def __init__(
        self,
        net: BaseNet,
        rules: Optional[Rules] = None,
        mcts_config: Optional[MCTSConfig] = None,
        visits: int = 200,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        self.net = net
        self.rules = rules or Rules(size=net.config.board_size)
        self.mcts_config = mcts_config or MCTSConfig()
        self.visits = visits
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ analysis
    def analyze(
        self,
        position: Position,
        visits: Optional[int] = None,
        add_noise: bool = False,
    ) -> Analysis:
        mcts = MCTS(self.net, self.mcts_config, batch_size=self.batch_size, rng=self.rng)
        mcts.set_root(position, add_noise=add_noise)
        mcts.run(visits if visits is not None else self.visits)

        # a fresh network evaluation supplies ownership and the score belief
        spatial, glob = encode_position(position)
        legal = position.legal_mask()
        out = self.net.predict(spatial[None], glob[None], legal[None])
        win_rate = float(out.value[0, 1])
        score_probs = out.score[0]
        bins = np.arange(len(score_probs)) - self.net.config.score_offset
        score_mean = float((score_probs * bins).sum())
        ownership = out.ownership[0]

        policy = mcts.policy_target(prune=False)
        move = mcts.best_move(temperature=0.0)
        return Analysis(
            visits=mcts.root.visits,
            win_rate=win_rate,
            score_mean=score_mean,
            ownership=ownership,
            policy=policy,
            move=move,
            principal_variation=mcts.principal_variation(8),
        )

    # ------------------------------------------------------------- play-out
    def genmove(
        self,
        position: Position,
        visits: Optional[int] = None,
        temperature: float = 0.0,
        resign_threshold: Optional[float] = None,
    ) -> int:
        """Return a move index, or ``PASS_MOVE``/``resign`` sentinel handling."""
        mcts = MCTS(self.net, self.mcts_config, batch_size=self.batch_size, rng=self.rng)
        mcts.set_root(position, add_noise=False)
        mcts.run(visits if visits is not None else self.visits)
        if resign_threshold is not None and mcts.root_q() < resign_threshold:
            return RESIGN
        return mcts.sample_move(temperature)


@dataclass
class MoveAnalysis:
    """One candidate move at the root of a search."""

    move: int
    visits: int
    win_rate: float  # for the player to move at the root, in [0, 1]
    score_mean: float  # in points, for the player to move at the root
    prior: float


@dataclass
class PositionReport:
    """A snapshot of an ongoing search, as the analysis GUI displays it."""

    visits: int
    win_rate: float  # for the player to move, in [0, 1]
    score_mean: float  # in points, for the player to move
    moves: List[MoveAnalysis]


class AnalysisSession:
    """Incremental analysis: run a few playouts, read the tree, repeat.

    Unlike :meth:`Engine.analyze`, the tree survives between calls, so the
    reported numbers converge as visits accumulate -- which is what the GUI
    shows live.  Root noise and forced playouts are off, so the same position
    searched again yields the same report.
    """

    def __init__(
        self,
        net: BaseNet,
        rules: Optional[Rules] = None,
        mcts_config: Optional[MCTSConfig] = None,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        self.net = net
        self.rules = rules or Rules(size=net.config.board_size)
        config = mcts_config or MCTSConfig(forced_playouts=False)
        self.mcts = MCTS(
            net, config, batch_size=batch_size, rng=np.random.default_rng(seed)
        )

    def set_position(self, position: Position) -> None:
        """Root the tree at ``position`` and evaluate it with the network."""
        self.mcts.set_root(position, add_noise=False)

    def step(self, playouts: int, top_k: int = 10) -> Tuple[PositionReport, int]:
        """Run up to ``playouts`` more simulations and snapshot the tree."""
        performed = self.mcts.run(playouts)
        return self.report(top_k), performed

    def report(self, top_k: int = 10) -> PositionReport:
        """Rank the root children by visits (then prior) and read them out."""
        root = self.mcts.root
        if root is None:
            raise RuntimeError("set_position() must be called before report()")
        children = root.children
        ordered = sorted(
            children.values(), key=lambda c: (c.visits, c.prior), reverse=True
        )[:top_k]
        moves: List[MoveAnalysis] = []
        for child in ordered:
            # child values are from the *opponent*'s point of view
            value = -child.mean_value if child.visits > 0 else 0.0
            score = -child.mean_score if child.visits > 0 else 0.0
            moves.append(
                MoveAnalysis(
                    move=child.move,
                    visits=child.visits,
                    win_rate=0.5 * (1.0 + value),
                    score_mean=score,
                    prior=child.prior,
                )
            )
        return PositionReport(
            visits=sum(c.visits for c in children.values()),
            win_rate=0.5 * (1.0 + root.mean_value),
            score_mean=root.mean_score,
            moves=moves,
        )


#: Sentinel returned by :meth:`Engine.genmove` when the engine resigns.
RESIGN = -2

#: GTP coordinate letters: 'I' is skipped, as in the Go Text Protocol.
GTP_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def move_to_gtp(move: int, size: int) -> str:
    if move == PASS_MOVE:
        return "pass"
    if move == RESIGN:
        return "resign"
    x, y = move % size, move // size
    return f"{GTP_LETTERS[x]}{size - y}"


def gtp_to_move(coord: str, size: int) -> int:
    c = coord.strip().lower()
    if c in ("pass", "resign"):
        return PASS_MOVE
    letter = c[0].upper()
    if letter not in GTP_LETTERS:
        raise ValueError(f"bad column {coord!r}")
    x = GTP_LETTERS.index(letter)
    y = size - int(c[1:])
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"coordinate {coord!r} outside a {size}x{size} board")
    return y * size + x
