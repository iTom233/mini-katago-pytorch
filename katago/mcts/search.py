"""KataGo-flavoured Monte-Carlo tree search.

Implements, with the paper's section numbers:

* **PUCT selection** with ``c_puct = 1.1`` (Section 2) and the first-play-urgency
  rule for unvisited children (Section 2, footnote 3):
  ``V(c) = V(parent) - c_fpu * sqrt(P_explored)``, ``c_fpu = 0.2``.
* **Root Dirichlet noise** ``P = 0.75 P_raw + 0.25 Dir(alpha)`` with
  ``alpha = 0.03 * 19**2 / N_legal`` (Section 2).
* **Root softmax temperature 1.03** on the priors, borrowed from SAI (Section 2).
* **Forced playouts** (Section 3.2): every root child that has received any
  playout is guaranteed at least ``n_forced(c) = (k P(c) sum N)^(1/2)`` of them.
* **Policy target pruning** (Section 3.2): those forced playouts are subtracted
  back out of the training target, decoupling it from MCTS exploration.
* **Playout cap randomization** (Section 3.1) is driven by the caller
  (see :mod:`katago.selfplay.worker`), which picks the per-turn cap.

Every node also averages the score belief of the evaluations it collects, so the
analysis GUI can show a score estimate per candidate move next to the win rate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..game import BLACK, PASS_MOVE, Position
from ..game.features import encode_position
from ..nn.config import MCTSConfig
from ..nn.base import BaseNet


class Node:
    """One node of the search tree. ``value`` is from ``to_play``'s perspective."""

    __slots__ = (
        "prior",
        "visits",
        "value_sum",
        "score_sum",
        "children",
        "is_expanded",
        "is_terminal",
        "terminal_value",
        "position",
        "to_play",
        "pending",
        "move",
    )

    def __init__(
        self,
        prior: float = 1.0,
        to_play: int = 1,
        move: int = -1,
        position: Optional[Position] = None,
    ) -> None:
        self.prior = float(prior)
        self.visits = 0
        self.value_sum = 0.0
        self.score_sum = 0.0
        self.children: Dict[int, "Node"] = {}
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value: float = 0.0
        self.position = position
        self.to_play = to_play
        self.pending = False
        self.move = move

    @property
    def mean_value(self) -> float:
        """Mean value from ``to_play``'s perspective."""
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def mean_score(self) -> float:
        """Mean score estimate, in points, from ``to_play``'s perspective."""
        if self.visits <= 0:
            return 0.0
        return self.score_sum / self.visits

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Node move={self.move} N={self.visits} Q={self.mean_value:+.3f}>"


def terminal_value_of(position: Position) -> float:
    """Value in ``[-1, 1]`` for the player to move at a finished game."""
    if position.is_draw:
        return 0.0
    score = position.score()  # Black's perspective
    black_won = score > 0
    if abs(score) < 1e-9:
        return 0.0
    winner = 1 if black_won else 2
    return 1.0 if winner == position.to_play else -1.0


def terminal_score_of(position: Position, to_play: int) -> float:
    """Area score in points for ``to_play`` at a finished game."""
    score = position.score()  # Black's perspective
    return score if to_play == BLACK else -score


class MCTS:
    """A single search tree rooted at one position."""

    def __init__(
        self,
        net: BaseNet,
        config: Optional[MCTSConfig] = None,
        batch_size: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.net = net
        self.config = config or MCTSConfig()
        self.batch_size = max(1, batch_size)
        self.rng = rng or np.random.default_rng()
        self.root: Optional[Node] = None
        self.root_value: float = 0.0
        self.root_score: float = 0.0
        self._forced_targets: Dict[int, float] = {}
        self._score_bins = (
            np.arange(net.config.score_bins, dtype=np.float64) - net.config.score_offset
        )

    # ------------------------------------------------------------------ setup
    def set_root(self, position: Position, add_noise: bool = True) -> None:
        """Build the root node, evaluating the position with the network."""
        spatial, glob = encode_position(position)
        legal = position.legal_mask()
        out = self.net.predict(
            spatial[None], glob[None], legal[None]
        )
        probs = out.policy[0].astype(np.float64)

        # root softmax temperature (SAI trick, Wu 2020 Section 2)
        temp = self.config.root_policy_temp
        if temp != 1.0 and probs.sum() > 0:
            with np.errstate(divide="ignore"):
                logits = np.log(np.maximum(probs, 1e-30))
            logits = np.where(legal, logits, -np.inf)
            probs = _softmax(logits * temp)

        if add_noise:
            n_legal = int(legal.sum())
            alpha = self.config.noise_alpha(n_legal)
            noise = self.rng.dirichlet(np.full(n_legal, alpha))
            noised = np.zeros_like(probs)
            noised[legal] = noise
            eps = self.config.root_noise_eps
            probs = (1.0 - eps) * probs + eps * noised

        self.root = Node(prior=1.0, to_play=position.to_play, position=position)
        self._expand(self.root, probs, is_root=True)
        self.root.visits = 1
        self.root.value_sum = float(out.value[0, 1] - out.value[0, 0])
        self.root.score_sum = float(np.dot(out.score[0], self._score_bins))
        self.root_value = self.root.value_sum
        self.root_score = self.root.score_sum
        self._compute_forced_targets()

    def _compute_forced_targets(self) -> None:
        """``n_forced(c) = (k P(c) sum_a N(a)) ** 0.5`` (Wu 2020, Eq. in 3.2)."""
        self._forced_targets = {}
        if not self.config.forced_playouts or self.root is None:
            return
        total = float(sum(c.visits for c in self.root.children.values())) or 1.0
        k = self.config.forced_k
        for move, child in self.root.children.items():
            self._forced_targets[move] = float(np.sqrt(k * child.prior * total))

    # -------------------------------------------------------------- expansion
    def _expand(self, node: Node, policy: np.ndarray, is_root: bool = False) -> None:
        position = node.position
        assert position is not None
        if position.is_terminal:
            node.is_terminal = True
            node.terminal_value = terminal_value_of(position)
            node.is_expanded = True
            return
        legal = position.legal_mask()
        num_moves = position.rules.num_moves
        for move in np.flatnonzero(legal):
            prior = float(policy[move]) if move < len(policy) else 0.0
            if prior <= 0.0:
                continue
            node.children[int(move)] = Node(
                prior=prior,
                to_play=3 - position.to_play,
                move=int(move),
                position=None,  # resolved lazily on first visit
            )
        node.is_expanded = True
        if not node.children:  # only pass is legal and it has zero prior
            node.is_terminal = True
            node.terminal_value = terminal_value_of(position)

    # --------------------------------------------------------------- selection
    def _puct_scores(self, node: Node) -> Dict[int, float]:
        """PUCT scores from ``node``'s perspective (negamax on the child values)."""
        children = node.children
        total = float(sum(c.visits for c in children.values()))
        sqrt_total = np.sqrt(total)
        parent_q = node.mean_value if node.visits > 0 else 0.0
        p_explored = sum(c.prior for c in children.values() if c.visits > 0)
        fpu = parent_q - self.config.c_fpu * np.sqrt(max(p_explored, 0.0))
        scores: Dict[int, float] = {}
        for move, child in children.items():
            if child.visits == 0:
                q = fpu
            else:
                q = -child.mean_value  # flip to the parent's perspective
            u = self.config.c_puct * child.prior * sqrt_total / (1.0 + child.visits)
            scores[move] = q + u
        return scores

    def _select_child(self, node: Node) -> Optional[Node]:
        candidates = {m: c for m, c in node.children.items() if not c.pending}
        if not candidates:
            return None
        # Forced playouts: at the root, a child that has been explored at all but
        # has fewer than n_forced playouts gets infinite urgency.
        if self.config.forced_playouts and node is self.root:
            best_move, best_deficit = None, 0.0
            for move, child in candidates.items():
                target = self._forced_targets.get(move, 0.0)
                deficit = target - child.visits
                if child.visits > 0 and deficit > best_deficit:
                    best_deficit, best_move = deficit, move
            if best_move is not None:
                return candidates[best_move]
        scores = self._puct_scores(node)
        best_move = max(candidates, key=lambda m: scores[m])
        return candidates[best_move]

    def _descend(self) -> Tuple[List[Tuple[Node, Node]], Optional[Node]]:
        """Walk from the root to a leaf, returning ``(path, leaf)``."""
        assert self.root is not None
        node = self.root
        path: List[Tuple[Node, Node]] = []
        while True:
            if node.is_terminal:
                return path, node
            if not node.is_expanded:
                return path, node
            child = self._select_child(node)
            if child is None:
                return path, None
            if child.position is None:
                assert node.position is not None
                child.position = node.position.play(child.move)
            path.append((node, child))
            node = child

    # ----------------------------------------------------------------- backup
    def _backup(
        self,
        path: List[Tuple[Node, Node]],
        leaf: Node,
        value: float,
        score: float = 0.0,
    ) -> None:
        """Propagate ``value``/``score`` (from ``leaf.to_play``'s perspective) up the path."""
        leaf.visits += 1
        leaf.value_sum += value
        leaf.score_sum += score
        v = value
        s = score
        for parent, child in reversed(path):
            v = -v  # flip to the parent's perspective
            s = -s
            parent.visits += 1
            parent.value_sum += v
            parent.score_sum += s

    # ------------------------------------------------------------------ search
    def run(self, playouts: int, max_time: Optional[float] = None) -> int:
        """Run up to ``playouts`` simulations; returns how many were performed."""
        import time

        if self.root is None or self.root.is_terminal:
            return 0
        done = 0
        start = time.perf_counter()
        while done < playouts:
            batch: List[Tuple[List[Tuple[Node, Node]], Node]] = []
            terminal_leaves = 0
            for _ in range(min(self.batch_size, playouts - done)):
                path, leaf = self._descend()
                if leaf is None:
                    break
                if leaf.is_terminal:
                    self._backup(
                        path,
                        leaf,
                        leaf.terminal_value,
                        terminal_score_of(leaf.position, leaf.to_play),
                    )
                    terminal_leaves += 1
                    continue
                leaf.pending = True
                leaf.visits += 1
                leaf.value_sum -= self.config.virtual_loss
                batch.append((path, leaf))
            if not batch:
                done += terminal_leaves
                if terminal_leaves == 0:
                    break
            else:
                feats = [encode_position(leaf.position) for _, leaf in batch]
                sp = np.stack([f[0] for f in feats])
                gl = np.stack([f[1] for f in feats])
                legal = np.stack([leaf.position.legal_mask() for _, leaf in batch])
                out = self.net.predict(sp, gl, legal)
                for i, (path, leaf) in enumerate(batch):
                    leaf.pending = False
                    leaf.visits -= 1
                    leaf.value_sum += self.config.virtual_loss
                    policy = out.policy[i].astype(np.float64)
                    self._expand(leaf, policy)
                    value = float(out.value[i, 1] - out.value[i, 0])
                    score = float(np.dot(out.score[i], self._score_bins))
                    self._backup(path, leaf, value, score)
                done += len(batch) + terminal_leaves
            if max_time is not None and time.perf_counter() - start > max_time:
                break
        if self.root is not None:
            self.root_value = self.root.mean_value
            self.root_score = self.root.mean_score
        return done

    # --------------------------------------------------------------- read-out
    def visit_counts(self) -> Tuple[List[int], np.ndarray]:
        """``(moves, visits)`` for every expanded root child."""
        assert self.root is not None
        moves = sorted(self.root.children)
        counts = np.array([self.root.children[m].visits for m in moves], dtype=np.float64)
        return moves, counts

    def policy_target(self, prune: Optional[bool] = None) -> np.ndarray:
        """The training policy target over all ``num_moves`` moves.

        With ``prune`` enabled this is the *pruned* visit distribution of
        Wu 2020, Section 3.2: forced playouts are subtracted back out, so the
        target is not contaminated by MCTS exploration, and children that end up
        with a single playout are dropped.
        """
        assert self.root is not None
        position = self.root.position
        assert position is not None
        num_moves = position.rules.num_moves
        if prune is None:
            prune = self.config.prune_policy_target
        moves, raw = self.visit_counts()

        candidates: List[np.ndarray] = []
        if prune and self.config.forced_playouts and len(moves) > 0:
            pruned = self._prune_counts(moves, raw.copy())
            # drop children reduced to a single playout
            pruned = np.where(pruned <= 1.0, 0.0, pruned)
            candidates.append(pruned)
        # fall back to the unpruned distribution if pruning removed everything;
        # this only happens with very small playout budgets
        candidates.append(raw)

        chosen = None
        for counts in candidates:
            if counts.sum() > 0:
                chosen = counts
                break

        out = np.zeros(num_moves, dtype=np.float64)
        if chosen is None:
            # last resort: the network priors
            for m in moves:
                out[m] = self.root.children[m].prior
            total = out.sum()
            return out / total if total > 0 else np.full(num_moves, 1.0 / num_moves)
        out[moves] = chosen
        return out / out.sum()

    def _prune_counts(self, moves: List[int], counts: np.ndarray) -> np.ndarray:
        """Subtract up to ``n_forced`` visits while PUCT stays below the best child."""
        children = self.root.children
        total = float(counts.sum())
        if total <= 0:
            return counts
        best = int(np.argmax(counts))
        best_move = moves[best]
        best_child = children[best_move]
        sqrt_total = np.sqrt(total)
        best_q = -best_child.mean_value if best_child.visits > 0 else 0.0
        best_score = best_q + self.config.c_puct * best_child.prior * sqrt_total / (
            1.0 + best_child.visits
        )
        pruned = counts.copy()
        for idx, move in enumerate(moves):
            if idx == best:
                continue
            child = children[move]
            budget = int(np.floor(self._forced_targets.get(move, 0.0)))
            n = pruned[idx]
            q = -child.mean_value if child.visits > 0 else 0.0
            removed = 0
            while removed < budget and n - 1 >= 1:
                trial = q + self.config.c_puct * child.prior * sqrt_total / (1.0 + (n - 1))
                if trial >= best_score:
                    break
                n -= 1
                removed += 1
            pruned[idx] = n
        return pruned

    def root_q(self) -> float:
        return self.root_value

    def best_move(self, temperature: float = 0.0, exclude: Optional[List[int]] = None) -> int:
        """Argmax of the (unpruned) visit counts, or a temperature sample."""
        moves, counts = self.visit_counts()
        if exclude:
            blocked = set(exclude)
            keep = [i for i, m in enumerate(moves) if m not in blocked]
            if not keep:
                return PASS_MOVE
            moves = [moves[i] for i in keep]
            counts = counts[keep]
        if len(moves) == 0:
            return PASS_MOVE
        if temperature <= 1e-3:
            return int(moves[int(np.argmax(counts))])
        weights = np.power(np.maximum(counts, 0.0), 1.0 / temperature)
        if weights.sum() <= 0:
            return int(moves[int(np.argmax(counts))])
        return int(self.rng.choice(moves, p=weights / weights.sum()))

    def sample_move(self, temperature: float, exclude: Optional[List[int]] = None) -> int:
        """Sample from visit counts raised to ``1/temperature``."""
        moves, counts = self.visit_counts()
        if exclude:
            blocked = set(exclude)
            keep = [i for i, m in enumerate(moves) if m not in blocked]
            if not keep:
                return PASS_MOVE
            moves = [moves[i] for i in keep]
            counts = counts[keep]
        if len(moves) == 0:
            return PASS_MOVE
        if temperature <= 1e-3:
            return int(moves[int(np.argmax(counts))])
        weights = np.power(np.maximum(counts, 1e-9), 1.0 / temperature)
        weights = weights / weights.sum()
        return int(self.rng.choice(moves, p=weights))

    # ------------------------------------------------------------ diagnostics
    def principal_variation(self, length: int = 8) -> List[int]:
        out: List[int] = []
        node = self.root
        for _ in range(length):
            if node is None or not node.children:
                break
            move = max(node.children, key=lambda m: node.children[m].visits)
            out.append(move)
            node = node.children[move]
        return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.nanmax(z)
    e = np.exp(z)
    return e / e.sum()
