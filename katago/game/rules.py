"""Go rules configuration.

Implements the rule set modelled by KataGo's self-play randomisation
(Wu 2020, Appendix D): board size, komi, ko rule, suicide, scoring.

The scoring used throughout is Tromp-Taylor area scoring
(https://tromp.github.io/go.html), which is what KataGo's self-play uses
after the modification described in footnote 4 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

PASS_MOVE: int = -1

#: Ko rule identifiers. Mirrors KataGo's ``simple``/``positional``/``situational``.
KO_SIMPLE = "simple"
KO_POSITIONAL = "positional"
KO_SITUATIONAL = "situational"

KO_RULES: Tuple[str, ...] = (KO_SIMPLE, KO_POSITIONAL, KO_SITUATIONAL)

BLACK = 1
WHITE = 2
EMPTY = 0


@dataclass(frozen=True)
class Rules:
    """A concrete rule set.

    Attributes
    ----------
    size:
        Board width; KataGo supports 9..19 (Wu 2020, Appendix A.1).
    komi:
        Compensation points for White.
    ko:
        One of ``simple``, ``positional``, ``situational``.
    suicide_allowed:
        Whether a suicidal move is legal. KataGo randomises this.
    max_moves:
        Hard cap on game length, after which the game is a draw.
    """

    size: int = 9
    komi: float = 7.5
    ko: str = KO_SITUATIONAL
    suicide_allowed: bool = False
    max_moves: int = 0  # 0 -> derived from board size

    def __post_init__(self) -> None:
        if not 2 <= self.size <= 25:
            raise ValueError(f"board size must be in 2..25, got {self.size}")
        if self.ko not in KO_RULES:
            raise ValueError(f"unknown ko rule {self.ko!r}")
        if self.max_moves == 0:
            object.__setattr__(self, "max_moves", self.size * self.size * 2)

    @property
    def num_points(self) -> int:
        return self.size * self.size

    @property
    def num_moves(self) -> int:
        """Size of the policy vector: every intersection plus pass."""
        return self.num_points + 1

    @property
    def pass_index(self) -> int:
        return self.num_points

    def successor(self, **kwargs) -> "Rules":
        """Return a copy with some fields replaced (handy for randomisation)."""
        data = dict(
            size=self.size,
            komi=self.komi,
            ko=self.ko,
            suicide_allowed=self.suicide_allowed,
            max_moves=self.max_moves,
        )
        data.update(kwargs)
        return Rules(**data)
