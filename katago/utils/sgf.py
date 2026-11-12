"""Minimal SGF writer, so self-play games can be reviewed in any Go GUI."""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..game import PASS_MOVE, Rules
from ..game.rules import BLACK, WHITE

SGF_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _sgf_point(move: int, size: int) -> str:
    if move == PASS_MOVE:
        return ""
    x, y = move % size, move // size
    return f"{SGF_LETTERS[x]}{SGF_LETTERS[y]}"


def to_sgf(
    moves: Iterable[int],
    size: int,
    komi: float = 7.5,
    black_name: str = "Black",
    white_name: str = "White",
    result: Optional[str] = None,
    comment: Optional[str] = None,
) -> str:
    """Serialise a move list as SGF."""
    props = [
        "(;GM[1]FF[4]",
        f"SZ[{size}]",
        f"KM[{komi}]",
        f"PB[{black_name}]",
        f"PW[{white_name}]",
    ]
    if result:
        props.append(f"RE[{result}]")
    if comment:
        props.append(f"C[{comment}]")
    body: List[str] = []
    color = BLACK
    for move in moves:
        tag = "B" if color == BLACK else "W"
        body.append(f";{tag}[{_sgf_point(move, size)}]")
        color = WHITE if color == BLACK else BLACK
    return "".join(props) + "".join(body) + ")\n"
