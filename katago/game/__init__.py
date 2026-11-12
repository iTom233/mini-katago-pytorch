"""Go rules, board representation and input features."""

from __future__ import annotations

from .board import (
    IllegalMove,
    Position,
    benson_pass_alive,
    group_and_liberties,
    neighbors_table,
)
from .features import (
    NUM_GLOBAL_FEATURES,
    NUM_SPATIAL_PLANES,
    encode_position,
)
from .ladder import ladder_captures, ladder_escape
from .rules import (
    BLACK,
    EMPTY,
    KO_POSITIONAL,
    KO_RULES,
    KO_SIMPLE,
    KO_SITUATIONAL,
    PASS_MOVE,
    WHITE,
    Rules,
)

__all__ = [
    "IllegalMove",
    "Position",
    "benson_pass_alive",
    "group_and_liberties",
    "neighbors_table",
    "NUM_GLOBAL_FEATURES",
    "NUM_SPATIAL_PLANES",
    "encode_position",
    "ladder_captures",
    "ladder_escape",
    "BLACK",
    "WHITE",
    "EMPTY",
    "PASS_MOVE",
    "Rules",
    "KO_SIMPLE",
    "KO_POSITIONAL",
    "KO_SITUATIONAL",
    "KO_RULES",
]
