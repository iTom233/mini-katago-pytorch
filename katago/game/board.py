"""Go board and position state.

A :class:`Position` is immutable: :meth:`Position.play` returns a new position.
This keeps search state simple to reason about (no undo bugs) at the cost of a
copy per move, which is irrelevant next to a neural network evaluation.

Indexing is flat: ``idx = y * size + x``.  The move value ``size * size`` is the
pass move.

Rule coverage
-------------
* captures, suicide (optional), simple / positional / situational ko, superko
* Tromp-Taylor area scoring
* Benson's algorithm for unconditionally alive ("pass-alive") areas, used by the
  pass-alive input feature (Wu 2020, Appendix A.1)
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .rules import BLACK, EMPTY, PASS_MOVE, WHITE, Rules

_NEIGHBOR_CACHE: Dict[int, Tuple[Tuple[int, ...], ...]] = {}


def neighbors_table(size: int) -> Tuple[Tuple[int, ...], ...]:
    """Neighbour lists for every intersection of a ``size`` x ``size`` board."""
    table = _NEIGHBOR_CACHE.get(size)
    if table is not None:
        return table
    out: List[Tuple[int, ...]] = []
    for idx in range(size * size):
        x, y = idx % size, idx // size
        ns: List[int] = []
        if y > 0:
            ns.append(idx - size)
        if y < size - 1:
            ns.append(idx + size)
        if x > 0:
            ns.append(idx - 1)
        if x < size - 1:
            ns.append(idx + 1)
        out.append(tuple(ns))
    _NEIGHBOR_CACHE[size] = tuple(out)
    return tuple(out)


def _zobrist_tables(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic Zobrist keys so hashes are stable across processes."""
    n = size * size
    seed = 0x9E3779B97F4A7C15 ^ (size * 0x100000001B3)
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, np.iinfo(np.uint64).max, size=(3, n), dtype=np.uint64)
    black_to_play = np.uint64(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    return keys[BLACK], keys[WHITE], black_to_play


_ZOBRIST: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def zobrist(size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size not in _ZOBRIST:
        _ZOBRIST[size] = _zobrist_tables(size)
    return _ZOBRIST[size]


class IllegalMove(Exception):
    """Raised when :meth:`Position.play` is given an illegal move."""


def group_and_liberties(
    grid: Sequence[int], idx: int, nbrs: Sequence[Tuple[int, ...]]
) -> Tuple[List[int], Set[int]]:
    """Flood-fill the chain containing ``idx``.

    Returns ``(stones, liberties)``; ``liberties`` is a set of empty points.
    """
    color = grid[idx]
    if color == EMPTY:
        return [], set()
    stack = [idx]
    seen = {idx}
    libs: Set[int] = set()
    while stack:
        p = stack.pop()
        for q in nbrs[p]:
            v = grid[q]
            if v == EMPTY:
                libs.add(q)
            elif v == color and q not in seen:
                seen.add(q)
                stack.append(q)
    return list(seen), libs


def benson_pass_alive(grid: Sequence[int], color: int, size: int) -> np.ndarray:
    """Benson's algorithm: the unconditionally-alive (pass-alive) area of ``color``.

    A *region* is a maximal connected set of points not occupied by ``color``.
    A region is *vital* to a chain when every empty point of the region is a
    liberty of that chain -- i.e. the region is an eye for it.  A chain is
    unconditionally alive when it is vital to at least two regions while each of
    those regions is vital to it alone; iterate to a fixed point.

    Note the difference from the naive "adjacent to at most one chain" reading:
    without the *vital* condition a lone stone on an empty board would count as
    alive, because the whole board is one region adjacent only to it.

    KataGo uses this (Wu 2020, footnote 4) to build the pass-alive input planes.
    Returns a boolean array of shape ``(size*size,)`` covering both the stones
    and the eye space they enclose.
    """
    n = size * size
    nbrs = neighbors_table(size)
    flat = list(grid)
    alive = np.zeros(n, dtype=bool)

    # --- chains of `color`, with their liberties
    chain_of = [-1] * n
    chains: List[List[int]] = []
    chain_libs: List[Set[int]] = []
    for idx in range(n):
        if flat[idx] == color and chain_of[idx] < 0:
            stones, libs = group_and_liberties(flat, idx, nbrs)
            cid = len(chains)
            chains.append(stones)
            chain_libs.append(libs)
            for s in stones:
                chain_of[s] = cid
    if not chains:
        return alive

    # --- regions: maximal connected sets of points not occupied by `color`
    region_of = [-1] * n
    regions: List[List[int]] = []
    for idx in range(n):
        if flat[idx] == color or region_of[idx] >= 0:
            continue
        rid = len(regions)
        stack = [idx]
        region_of[idx] = rid
        cells: List[int] = []
        while stack:
            p = stack.pop()
            cells.append(p)
            for q in nbrs[p]:
                if flat[q] != color and region_of[q] < 0:
                    region_of[q] = rid
                    stack.append(q)
        regions.append(cells)
    region_empty = [[p for p in cells if flat[p] == EMPTY] for cells in regions]

    live_chains = set(range(len(chains)))
    live_regions = set(range(len(regions)))
    changed = True
    while changed:
        changed = False
        # A surviving region must be vital to exactly one surviving chain.
        vitality: Dict[int, int] = {}
        for rid in list(live_regions):
            empties = region_empty[rid]
            if not empties:
                live_regions.discard(rid)
                changed = True
                continue
            adjacent: Set[int] = set()
            for p in empties:
                for q in nbrs[p]:
                    if flat[q] == color:
                        adjacent.add(chain_of[q])
            vital = [
                c
                for c in adjacent
                if c in live_chains and all(p in chain_libs[c] for p in empties)
            ]
            if len(vital) != 1:
                live_regions.discard(rid)
                changed = True
                continue
            vitality[rid] = vital[0]
        # A surviving chain needs at least two surviving eyes.
        eye_count: Dict[int, int] = {}
        for rid in live_regions:
            c = vitality[rid]
            if c in live_chains:
                eye_count[c] = eye_count.get(c, 0) + 1
        for cid in list(live_chains):
            if eye_count.get(cid, 0) < 2:
                live_chains.discard(cid)
                changed = True

    for cid in live_chains:
        for p in chains[cid]:
            alive[p] = True
    for rid in live_regions:
        if vitality[rid] in live_chains:
            for p in regions[rid]:
                alive[p] = True
    return alive


class Position:
    """An immutable Go position.

    Attributes
    ----------
    rules: rule set in force.
    grid: ``int8`` array of shape ``(size, size)`` with 0/1/2.
    to_play: ``BLACK`` or ``WHITE``.
    ko_point: flat index forbidden by the simple-ko rule, or ``-1``.
    move_history: flat indices of all moves played so far (``PASS_MOVE`` for pass).
    """

    __slots__ = (
        "rules",
        "grid",
        "to_play",
        "ko_point",
        "move_history",
        "consecutive_passes",
        "move_number",
        "_hash",
        "_hash_stack",
        "_hash_stack_board",
        "_legal_mask",
        "_pass_alive",
    )

    def __init__(
        self,
        rules: Rules,
        grid: np.ndarray,
        to_play: int,
        ko_point: int = -1,
        move_history: Tuple[int, ...] = (),
        consecutive_passes: int = 0,
        hash_stack: Tuple[int, ...] = (),
        hash_stack_board: Tuple[int, ...] = (),
    ) -> None:
        self.rules = rules
        self.grid = grid
        self.to_play = to_play
        self.ko_point = ko_point
        self.move_history = move_history
        self.consecutive_passes = consecutive_passes
        self.move_number = len(move_history)
        # One board hash feeds both stacks; ``board_hash`` recomputed it a second
        # time, which doubled the cost of creating every search-tree node.
        board = self._board_hash(self.grid.reshape(-1))
        self._hash = board ^ (int(zobrist(self.rules.size)[2]) if to_play == BLACK else 0)
        self._hash_stack = hash_stack + (self._hash,)
        self._hash_stack_board = hash_stack_board + (board,)
        self._legal_mask: Optional[np.ndarray] = None
        self._pass_alive: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ---------------------------------------------------------------- basics

    @staticmethod
    def empty(rules: Rules, to_play: int = BLACK) -> "Position":
        return Position(
            rules=rules,
            grid=np.zeros((rules.size, rules.size), dtype=np.int8),
            to_play=to_play,
        )

    @property
    def size(self) -> int:
        return self.rules.size

    @property
    def num_points(self) -> int:
        return self.rules.num_points

    @property
    def pass_index(self) -> int:
        return self.rules.pass_index

    @property
    def opponent(self) -> int:
        return BLACK if self.to_play == WHITE else WHITE

    def flat(self) -> List[int]:
        return self.grid.reshape(-1).tolist()

    def _compute_hash(self) -> int:
        """Situational hash: board arrangement *and* player to move."""
        return self._board_hash(self.grid.reshape(-1)) ^ (
            int(zobrist(self.size)[2]) if self.to_play == BLACK else 0
        )

    def _board_hash(self, flat: Sequence[int]) -> int:
        black_key, white_key, _ = zobrist(self.size)
        arr = np.asarray(flat, dtype=np.int64)
        h = np.uint64(0)
        h ^= black_key[arr == BLACK].sum(dtype=np.uint64)
        h ^= white_key[arr == WHITE].sum(dtype=np.uint64)
        return int(h)

    @property
    def hash(self) -> int:
        return self._hash

    @property
    def board_hash(self) -> int:
        """Hash of the stone arrangement only (ignores the player to move)."""
        return self._board_hash(self.grid.reshape(-1))

    # -------------------------------------------------------------- legality

    def _simulate(
        self, move: int
    ) -> Tuple[Optional[List[int]], Optional[List[int]], int]:
        """Apply ``move`` on a copy of the flat grid.

        Returns ``(grid, captured, ko_point)``; ``grid`` is ``None`` when the
        move is illegal.
        """
        n = self.num_points
        if move < 0 or move >= n:
            return None, None, -1
        flat = self.flat()
        if flat[move] != EMPTY:
            return None, None, -1
        nbrs = neighbors_table(self.size)
        color = self.to_play
        opp = 3 - color
        flat[move] = color
        captured: List[int] = []
        for q in nbrs[move]:
            if flat[q] == opp:
                stones, libs = group_and_liberties(flat, q, nbrs)
                if not libs:
                    for s in stones:
                        flat[s] = EMPTY
                    captured.extend(stones)
        own_stones, own_libs = group_and_liberties(flat, move, nbrs)
        if not own_libs:
            if not self.rules.suicide_allowed:
                return None, None, -1
            for s in own_stones:
                flat[s] = EMPTY
            captured.extend(own_stones)
        ko_point = -1
        if len(captured) == 1 and len(own_stones) == 1 and len(own_libs) == 1:
            ko_point = captured[0]
        return flat, captured, ko_point

    def _flat_hash(self, flat: Sequence[int], to_play: int) -> int:
        h = self._board_hash(flat)
        if to_play == BLACK:
            h ^= int(zobrist(self.size)[2])
        return h

    def _violates_superko(self, flat: Sequence[int], to_play: int) -> bool:
        """Superko check.

        ``positional`` forbids recreating any earlier *arrangement*, ignoring
        whose turn it is; ``situational`` also distinguishes the player to move.
        ``simple`` is handled entirely by ``ko_point``.
        """
        rule = self.rules.ko
        if rule == "simple":
            return False
        board_h = self._board_hash(flat)
        if rule == "positional":
            return board_h in self._hash_stack_board
        return self._flat_hash(flat, to_play) in self._hash_stack

    def is_legal(self, move: int) -> bool:
        if move == PASS_MOVE:
            return True
        if move < 0 or move >= self.num_points:
            return False
        if self.grid.reshape(-1)[move] != EMPTY:
            return False
        if move == self.ko_point:
            return False
        flat, captured, _ = self._simulate(move)
        if flat is None:
            return False
        # A move that captures nothing adds a stone, so the new board cannot
        # reproduce an earlier position and the superko lookup is unnecessary.
        # This is the hot path: ``legal_mask`` runs it for every empty point of
        # every newly created leaf position, and the hash is the expensive part.
        if not captured:
            return True
        return not self._violates_superko(flat, 3 - self.to_play)

    def legal_mask(self) -> np.ndarray:
        """Boolean vector of length ``num_moves``; the pass entry is always True."""
        if self._legal_mask is None:
            flat = self.grid.reshape(-1)
            mask = np.zeros(self.rules.num_moves, dtype=bool)
            for idx in range(self.num_points):
                if flat[idx] == EMPTY and idx != self.ko_point:
                    mask[idx] = self.is_legal(idx)
            mask[self.pass_index] = True
            self._legal_mask = mask
        return self._legal_mask

    def legal_moves(self) -> List[int]:
        return np.flatnonzero(self.legal_mask()).tolist()

    # ------------------------------------------------------------------ play

    def play(self, move: int) -> "Position":
        if move == PASS_MOVE or move == self.pass_index:
            return Position(
                rules=self.rules,
                grid=self.grid,
                to_play=self.opponent,
                ko_point=-1,
                move_history=self.move_history + (PASS_MOVE,),
                consecutive_passes=self.consecutive_passes + 1,
                hash_stack=self._hash_stack,
                hash_stack_board=self._hash_stack_board,
            )
        if not self.is_legal(move):
            raise IllegalMove(f"illegal move {move} for player {self.to_play}")
        flat, _, ko_point = self._simulate(move)
        assert flat is not None
        grid = np.asarray(flat, dtype=np.int8).reshape(self.size, self.size)
        return Position(
            rules=self.rules,
            grid=grid,
            to_play=self.opponent,
            ko_point=ko_point,
            move_history=self.move_history + (move,),
            consecutive_passes=0,
            hash_stack=self._hash_stack,
            hash_stack_board=self._hash_stack_board,
        )

    # -------------------------------------------------------------- terminal

    @property
    def is_terminal(self) -> bool:
        if self.consecutive_passes >= 2:
            return True
        return self.move_number >= self.rules.max_moves

    @property
    def is_draw(self) -> bool:
        return self.move_number >= self.rules.max_moves and self.consecutive_passes < 2

    # -------------------------------------------------------------- scoring

    def area_score(self) -> Tuple[int, int]:
        """Tromp-Taylor area score ``(black, white)`` in points, komi excluded."""
        size = self.size
        grid = self.grid.reshape(-1).tolist()
        nbrs = neighbors_table(size)
        owner = [-1] * (size * size)
        for idx in range(size * size):
            if grid[idx] != EMPTY:
                continue
            if owner[idx] >= 0:
                continue
            stack = [idx]
            region = []
            borders: Set[int] = set()
            owner[idx] = -2
            while stack:
                p = stack.pop()
                region.append(p)
                for q in nbrs[p]:
                    if grid[q] == EMPTY:
                        if owner[q] == -1:
                            owner[q] = -2
                            stack.append(q)
                    else:
                        borders.add(grid[q])
            if borders == {BLACK}:
                for p in region:
                    owner[p] = BLACK
            elif borders == {WHITE}:
                for p in region:
                    owner[p] = WHITE
            else:
                for p in region:
                    owner[p] = EMPTY
        black = sum(1 for idx in range(size * size) if grid[idx] == BLACK or owner[idx] == BLACK)
        white = sum(1 for idx in range(size * size) if grid[idx] == WHITE or owner[idx] == WHITE)
        return black, white

    def score(self, komi: Optional[float] = None) -> float:
        """Score from **Black's** perspective (``black - white - komi``)."""
        black, white = self.area_score()
        k = self.rules.komi if komi is None else komi
        return float(black) - float(white) - float(k)

    def area_ownership(self) -> np.ndarray:
        """Per-point ownership as Black's share in ``{0.0, 0.5, 1.0}``.

        This is the auxiliary ownership target of Wu 2020 Section 4.1:
        ``o(l, p) in {0, 0.5, 1}`` where ``o(l, black) + o(l, white) == 1``.
        """
        size = self.size
        grid = self.grid.reshape(-1).tolist()
        nbrs = neighbors_table(size)
        own = np.full(size * size, 0.5, dtype=np.float32)
        seen = [False] * (size * size)
        for idx in range(size * size):
            if grid[idx] == BLACK:
                own[idx] = 1.0
                seen[idx] = True
            elif grid[idx] == WHITE:
                own[idx] = 0.0
                seen[idx] = True
        for idx in range(size * size):
            if seen[idx]:
                continue
            stack = [idx]
            region = []
            seen[idx] = True
            borders: Set[int] = set()
            while stack:
                p = stack.pop()
                region.append(p)
                for q in nbrs[p]:
                    if grid[q] == EMPTY:
                        if not seen[q]:
                            seen[q] = True
                            stack.append(q)
                    else:
                        borders.add(grid[q])
            if borders == {BLACK}:
                for p in region:
                    own[p] = 1.0
            elif borders == {WHITE}:
                for p in region:
                    own[p] = 0.0
            else:
                for p in region:
                    own[p] = 0.5
        return own.reshape(size, size)

    # ---------------------------------------------------------- pass-alive

    def pass_alive(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(self_alive, opponent_alive)`` boolean arrays of shape ``(size, size)``."""
        if self._pass_alive is None:
            flat = self.flat()
            mine = benson_pass_alive(flat, self.to_play, self.size).reshape(self.size, self.size)
            theirs = benson_pass_alive(flat, self.opponent, self.size).reshape(self.size, self.size)
            self._pass_alive = (mine, theirs)
        return self._pass_alive

    # --------------------------------------------------------------- misc

    def stones_of(self, color: int) -> np.ndarray:
        return self.grid == color

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Position size={self.size} to_play={self.to_play} moves={self.move_number}>"

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        rows = []
        letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        header = "   " + " ".join(letters[: self.size])
        rows.append(header)
        for y in range(self.size):
            cells = []
            for x in range(self.size):
                v = self.grid[y, x]
                cells.append("X" if v == BLACK else ("O" if v == WHITE else "."))
            rows.append(f"{self.size - y:2d} " + " ".join(cells) + f" {self.size - y}")
        rows.append(header)
        return "\n".join(rows)
