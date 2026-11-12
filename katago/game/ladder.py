"""Bounded ladder (shicho) reader.

KataGo's input features include "ladderable stones" and "moving here catches an
opponent in a ladder" (Wu 2020, Appendix A.1, Table 3).  A full ladder solver is
a substantial piece of computer-Go engineering; this module implements the core
reading loop -- continuous atari with escape detection -- which handles the
standard ladder and its common breaker, and bails out beyond a depth bound.

It is deliberately conservative: when unsure it reports "not a ladder", so the
feature can only under-report, never fabricate.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .board import EMPTY, group_and_liberties, neighbors_table


def _liberties(flat: List[int], stones: Sequence[int], nbrs) -> List[int]:
    libs = set()
    for s in stones:
        for q in nbrs[s]:
            if flat[q] == EMPTY:
                libs.add(q)
    return sorted(libs)


def ladder_captures(
    flat: Sequence[int],
    grid_size: int,
    target: int,
    max_depth: Optional[int] = None,
) -> bool:
    """Can the player *not* owning the stone at ``target`` capture it by a ladder?

    ``flat`` is the board as a flat list; ``target`` an index of a stone.  The
    function returns ``True`` only when it can prove a continuous-atari capture,
    i.e. the defender is always forced and always ends in atari.
    """
    nbrs = neighbors_table(grid_size)
    grid = list(flat)
    if grid[target] == EMPTY:
        return False
    defender = grid[target]
    attacker = 3 - defender
    if max_depth is None:
        max_depth = 2 * grid_size + 4

    stones, libs = group_and_liberties(grid, target, nbrs)
    libs = list(libs)
    if len(libs) > 2:
        return False

    for _ in range(max_depth):
        lib_set = _liberties(grid, stones, nbrs)
        if not lib_set:
            return True  # captured
        if len(lib_set) > 2:
            return False  # defender has room; not a ladder
        if len(lib_set) == 1:
            return True  # attacker simply captures next move

        # Two liberties: the attacker must play one, forcing the defender to run
        # out of the other one.
        progressed = False
        for atari_point in lib_set:
            probe = list(grid)
            probe[atari_point] = attacker
            captured_own = False
            for q in nbrs[atari_point]:
                if probe[q] == defender:
                    grp, gl = group_and_liberties(probe, q, nbrs)
                    if not gl:
                        for s in grp:
                            probe[s] = EMPTY
            # Attacker's own stone must survive (no self-capture).
            if probe[atari_point] != attacker:
                continue
            agrp, alib = group_and_liberties(probe, atari_point, nbrs)
            if not alib:
                continue

            remaining = [p for p in lib_set if p != atari_point]
            if len(remaining) != 1:
                continue
            escape = remaining[0]
            if probe[escape] != EMPTY:
                continue

            # Defender extends.
            after = list(probe)
            after[escape] = defender
            # If the extension captures any attacker stone the ladder broke.
            for q in nbrs[escape]:
                if after[q] == attacker:
                    grp, gl = group_and_liberties(after, q, nbrs)
                    if not gl:
                        captured_own = True
                        break
            if captured_own:
                continue

            new_stones, _ = group_and_liberties(after, escape, nbrs)
            new_libs = _liberties(after, new_stones, nbrs)
            if len(new_libs) > 2:
                continue

            grid = after
            stones = new_stones
            progressed = True
            break

        if not progressed:
            return False
    return False


def ladder_escape(
    flat: Sequence[int], grid_size: int, target: int, max_depth: Optional[int] = None
) -> bool:
    """Inverse of :func:`ladder_captures` for the owner of ``target``."""
    nbrs = neighbors_table(grid_size)
    grid = list(flat)
    if grid[target] == EMPTY:
        return False
    defender = grid[target]
    attacker = 3 - defender
    stones, libs = group_and_liberties(grid, target, nbrs)
    if len(libs) > 2:
        return True  # already free
    return not ladder_captures(flat, grid_size, target, max_depth)
