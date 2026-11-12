"""KataGo input feature encoding.

Implements exactly the feature set described in Wu 2020, Appendix A.1:

    spatial:  b x b x 18 binary planes
    global:   10 real values

Plane layout (Table 3)::

    0        location is on board
    1        location has own stone
    2        location has opponent stone
    3..5     location has a stone with 1 / 2 / 3 liberties
    6        moving here is illegal due to ko / superko
    7..11    the last five move locations, most recent first (one-hot)
    12..14   ladderable stones 0 / 1 / 2 turns ago
    15       moving here catches an opponent stone in a ladder
    16..17   pass-alive area for self / opponent

Global vector (Table 4)::

    0..4     which of the previous five moves were passes
    5        komi / 15
    6..7     ko rule, one-hot over {simple, positional, situational}
    8        suicide allowed
    9        komi + board size parity
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .board import EMPTY, Position, group_and_liberties, neighbors_table
from .ladder import ladder_captures
from .rules import PASS_MOVE

NUM_SPATIAL_PLANES = 18
NUM_GLOBAL_FEATURES = 10


def _chain_liberty_counts(flat: List[int], size: int) -> np.ndarray:
    """Liberty count of the chain each stone belongs to (0 for empty points)."""
    nbrs = neighbors_table(size)
    counts = np.zeros(size * size, dtype=np.int8)
    seen = [False] * (size * size)
    for idx in range(size * size):
        if flat[idx] == EMPTY or seen[idx]:
            continue
        # iterative flood fill so 19x19 boards do not hit the recursion limit
        stack = [idx]
        seen[idx] = True
        stones = []
        libs = set()
        while stack:
            p = stack.pop()
            stones.append(p)
            for q in nbrs[p]:
                if flat[q] == EMPTY:
                    libs.add(q)
                elif flat[q] == flat[idx] and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        counts[stones] = min(len(libs), 127)
    return counts


def _ladder_planes(pos: Position, flat: List[int], size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ladder planes 12..14 (stones played 0/1/2 moves ago) and plane 15."""
    nbrs = neighbors_table(size)
    recent = np.zeros((3, size, size), dtype=np.float32)
    hist = pos.move_history
    for k in range(3):
        if len(hist) > k:
            mv = hist[-1 - k]
            if mv == PASS_MOVE or flat[mv] == EMPTY:
                continue
            # Is the stone that was played then still capturable by a ladder?
            if ladder_captures(flat, size, mv):
                recent[k, mv // size, mv % size] = 1.0

    catches = np.zeros((size, size), dtype=np.float32)
    opp = pos.opponent
    seen = [False] * (size * size)
    for idx in range(size * size):
        if flat[idx] != opp or seen[idx]:
            continue
        stones, libs = group_and_liberties(flat, idx, nbrs)
        for s in stones:
            seen[s] = True
        if len(libs) != 2:
            continue
        for lib in libs:
            if flat[lib] != EMPTY:
                continue
            probe = list(flat)
            probe[lib] = pos.to_play
            for q in nbrs[lib]:
                if probe[q] == opp:
                    grp, gl = group_and_liberties(probe, q, nbrs)
                    if not gl:
                        for s in grp:
                            probe[s] = EMPTY
            if probe[lib] != pos.to_play:
                continue
            # The chain we would put in atari must be the same one.
            new_stones, new_libs = group_and_liberties(probe, lib, nbrs)
            if len(new_libs) != 1:
                continue
            if ladder_captures(probe, size, idx):
                catches[lib // size, lib % size] = 1.0
    return recent, catches


def encode_position(pos: Position) -> Tuple[np.ndarray, np.ndarray]:
    """Encode ``pos`` from the perspective of the player to move.

    Returns ``(spatial, global_)`` with shapes ``(18, S, S)`` and ``(10,)``.
    """
    size = pos.size
    flat = pos.flat()
    nbrs = neighbors_table(size)

    spatial = np.zeros((NUM_SPATIAL_PLANES, size, size), dtype=np.float32)
    spatial[0] = 1.0
    grid = pos.grid
    spatial[1] = (grid == pos.to_play).astype(np.float32)
    spatial[2] = (grid == pos.opponent).astype(np.float32)

    counts = _chain_liberty_counts(flat, size).reshape(size, size)
    for k, libs in enumerate((1, 2, 3)):
        spatial[3 + k] = ((counts == libs) & (grid != EMPTY)).astype(np.float32)

    # Illegal-by-ko plane: empty points that are not legal moves.
    legal = pos.legal_mask()
    legal_grid = legal[: pos.num_points].reshape(size, size)
    spatial[6] = ((grid == EMPTY) & ~legal_grid).astype(np.float32)

    hist = pos.move_history
    for k in range(5):
        if len(hist) > k:
            mv = hist[-1 - k]
            if mv != PASS_MOVE and 0 <= mv < pos.num_points:
                spatial[7 + k, mv // size, mv % size] = 1.0

    recent, catches = _ladder_planes(pos, flat, size)
    spatial[12:15] = recent
    spatial[15] = catches

    mine_alive, theirs_alive = pos.pass_alive()
    spatial[16] = mine_alive.astype(np.float32)
    spatial[17] = theirs_alive.astype(np.float32)

    glob = np.zeros(NUM_GLOBAL_FEATURES, dtype=np.float32)
    for k in range(5):
        if len(hist) > k:
            glob[k] = 1.0 if hist[-1 - k] == PASS_MOVE else 0.0
    glob[5] = float(pos.rules.komi) / 15.0
    ko_index = {"simple": 0, "positional": 1, "situational": 2}[pos.rules.ko]
    if ko_index == 0:
        glob[6], glob[7] = 0.0, 0.0
    elif ko_index == 1:
        glob[6], glob[7] = 1.0, 0.0
    else:
        glob[6], glob[7] = 0.0, 1.0
    glob[8] = 1.0 if pos.rules.suicide_allowed else 0.0
    glob[9] = (float(pos.rules.komi) + pos.size % 2) / 15.0

    return spatial, glob
