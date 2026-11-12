"""Self-play data generation.

Implements the KataGo self-play loop of Wu 2020:

* playout cap randomization (Section 3.1): a fraction ``p`` of turns get a full
  search, the rest get a cheap one; **only full-search turns are recorded for
  training**, and fast searches disable root noise to maximise strength;
* policy target pruning is applied when the turn is recorded (Section 3.2);
* the auxiliary opponent-policy target for turn ``t`` is the recorded policy of
  turn ``t + 1`` (Section 3.4);
* the value, ownership and score targets come from the finished game
  (Sections 2 and 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..game import PASS_MOVE, Position, Rules
from ..game.features import encode_position
from ..mcts import MCTS
from ..nn.base import BaseNet
from ..nn.config import MCTSConfig, NetConfig, PlayoutCapConfig


@dataclass
class SelfPlayConfig:
    #: play with temperature 1 for the first ``temperature_moves`` turns
    temperature_moves: int = 12
    #: resign once the root value falls below the MCTS threshold
    resign_enabled: bool = True
    #: never resign before this many moves (avoids silly early resignations)
    resign_min_move: int = 10
    #: batch size for leaf evaluation
    batch_size: int = 8
    #: cap on the number of moves per game
    max_moves: Optional[int] = None
    #: forbid passing before this move number in self-play only.
    #:
    #: Small-scale runs collapse into a degenerate "pass immediately" equilibrium:
    #: Tromp-Taylor area scoring on a nearly empty board awards the whole board to
    #: whichever colour has a single stone, so ending the game at move 2 *looks*
    #: like a huge win to the value target.  At full scale the search discovers
    #: that the opponent simply keeps playing, but a CPU demo cannot get there, so
    #: Self-play masks the pass move early on.  Inference is unaffected.
    min_pass_move: int = 0


@dataclass
class GameRecord:
    """One finished self-play game, in the form the trainer consumes."""

    spatial: np.ndarray  # (T, 18, S, S) uint8
    glob: np.ndarray  # (T, 10) float32
    policy: np.ndarray  # (T, M) float32, pruned
    player: np.ndarray  # (T,) int8
    ownership: np.ndarray  # (S*S,) float32, Black's share at the end
    final_score: float  # Black's perspective, komi included
    winner: int  # 1 black, 2 white, 0 draw
    moves: List[int] = field(default_factory=list)
    resigned: bool = False
    num_full_searches: int = 0

    @property
    def num_samples(self) -> int:
        return int(self.spatial.shape[0])

    @property
    def rules_size(self) -> int:
        return int(self.spatial.shape[-1])

    def samples(self) -> Dict[str, np.ndarray]:
        """Per-turn training targets, all from the *mover's* perspective."""
        t = self.num_samples
        size = self.rules_size
        n_pts = size * size
        score_offset = n_pts + 1
        score_bins = 2 * (n_pts + 1) + 1

        # value: +1 win, -1 loss, 0 draw
        value = np.zeros((t, 2), dtype=np.float32)
        if self.winner == 0:
            value[:, 0] = 0.5
            value[:, 1] = 0.5
        else:
            won = (self.player == self.winner).astype(np.float32)
            value[:, 1] = won
            value[:, 0] = 1.0 - won

        # ownership from the mover's perspective
        black_own = self.ownership.reshape(-1)
        own = np.where(self.player[:, None] == 1, black_own[None, :], 1.0 - black_own[None, :])

        # score from the mover's perspective
        signed = np.where(self.player == 1, self.final_score, -self.final_score)
        bins = np.clip(np.round(signed).astype(np.int64) + score_offset, 0, score_bins - 1)

        # auxiliary target: the *next* turn's recorded policy
        aux = np.zeros_like(self.policy)
        has_aux = np.zeros((t,), dtype=bool)
        if t > 1:
            aux[:-1] = self.policy[1:]
            norm = aux[:-1].sum(axis=1, keepdims=True)
            has_aux[:-1] = norm[:, 0] > 0
            np.divide(aux[:-1], np.maximum(norm, 1e-12), out=aux[:-1])
            aux[~has_aux] = 0.0

        return {
            "spatial": self.spatial,
            "global_": self.glob,
            "policy": self.policy,
            "aux": aux,
            "has_aux": has_aux,
            "value": value,
            "ownership": own.astype(np.float32),
            "score": bins,
        }


def _mask_pass(target: np.ndarray) -> np.ndarray:
    """Zero the pass entry of a policy target and renormalise."""
    out = target.copy()
    out[-1] = 0.0
    total = out.sum()
    if total <= 0:
        return target
    return (out / total).astype(np.float32)


def play_game(
    net: BaseNet,
    rules: Rules,
    mcts_config: MCTSConfig,
    cap_config: PlayoutCapConfig,
    selfplay_config: SelfPlayConfig,
    rng: np.random.Generator,
    samples_seen: int = 0,
) -> GameRecord:
    """Play one game of self-play and return its training record."""
    position = Position.empty(rules)
    spatial: List[np.ndarray] = []
    glob: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    players: List[int] = []
    moves: List[int] = []
    resigned = False

    max_moves = selfplay_config.max_moves or rules.max_moves

    while not position.is_terminal and len(moves) < max_moves:
        mcts = MCTS(net, mcts_config, batch_size=selfplay_config.batch_size, rng=rng)
        # playout cap randomization: a fraction of turns get a full search;
        # only those turns are recorded for training (Wu 2020, Section 3.1).
        full = (not cap_config.enabled) or (rng.random() < cap_config.full_fraction)
        mcts.set_root(position, add_noise=full)
        full_cap, fast_cap = cap_config.caps(samples_seen)
        cap = full_cap if full else fast_cap
        mcts.run(cap)

        if full:
            sp, gl = encode_position(position)
            spatial.append(sp.astype(np.uint8))
            glob.append(gl)
            target = mcts.policy_target(prune=True).astype(np.float32)
            if len(moves) < selfplay_config.min_pass_move:
                target = _mask_pass(target)
            policies.append(target)
            players.append(position.to_play)

        # resignation: the mover is losing badly
        if (
            selfplay_config.resign_enabled
            and len(moves) >= selfplay_config.resign_min_move
            and mcts.root.visits >= mcts_config.resign_min_visits
            and mcts.root_q() < mcts_config.resign_threshold
        ):
            resigned = True
            winner = 3 - position.to_play
            break

        temperature = 1.0 if len(moves) < selfplay_config.temperature_moves else 0.0
        exclude = (
            [rules.pass_index] if len(moves) < selfplay_config.min_pass_move else None
        )
        move = mcts.sample_move(temperature, exclude=exclude)
        moves.append(move)
        position = position.play(move)

    if not resigned:
        if position.is_draw:
            winner = 0
        else:
            score = position.score()
            winner = 0 if abs(score) < 1e-9 else (1 if score > 0 else 2)

    final_score = position.score()
    ownership = position.area_ownership().reshape(-1).astype(np.float32)

    if not spatial:
        # pathological: every turn was a fast search. Record one position anyway
        sp, gl = encode_position(position)
        spatial.append(sp.astype(np.uint8))
        glob.append(gl)
        legal = position.legal_mask().astype(np.float32)
        policies.append((legal / legal.sum()).astype(np.float32))
        players.append(position.to_play)

    return GameRecord(
        spatial=np.stack(spatial),
        glob=np.stack(glob).astype(np.float32),
        policy=np.stack(policies).astype(np.float32),
        player=np.asarray(players, dtype=np.int8),
        ownership=ownership,
        final_score=float(final_score),
        winner=int(winner),
        moves=moves,
        resigned=resigned,
        num_full_searches=len(policies),
    )
