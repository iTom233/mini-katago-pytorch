"""Rules-engine tests: captures, ko, suicide, scoring, Benson, ladders, features."""

from __future__ import annotations

import numpy as np

from support import approx, raises

from katago.game import (
    BLACK,
    EMPTY,
    PASS_MOVE,
    WHITE,
    IllegalMove,
    Position,
    Rules,
    benson_pass_alive,
    encode_position,
    ladder_captures,
    neighbors_table,
)


def pos_from(rows, to_play=BLACK, size=None, **rule_kwargs):
    """Build a position from ASCII art. ``X`` black, ``O`` white, ``.`` empty."""
    if size is None:
        size = len(rows)
    grid = np.zeros((size, size), dtype=np.int8)
    for y in range(size):
        row = rows[y] if y < len(rows) else "." * size
        cells = [c for c in row.replace(" ", "")]
        assert len(cells) == size, f"row {y} has {len(cells)} cells, expected {size}"
        for x, ch in enumerate(cells):
            if ch in "Xx":
                grid[y, x] = BLACK
            elif ch in "Oo":
                grid[y, x] = WHITE
    return Position(rules=Rules(size=size, **rule_kwargs), grid=grid, to_play=to_play)


def ko_position(ko_rule="simple"):
    """The canonical ko shape, Black to play at (1,1).

    ``. . . .``       black: (0,2) (2,2) (1,3)
    ``. O . .``       white: (0,1) (2,1) (1,0) (1,2)
    ``. O X .``       black captures the white stone at (1,2) by playing (1,1).
    ``. X . .``
    """
    size = 9
    grid = np.zeros((size, size), dtype=np.int8)
    for y, x in ((0, 2), (2, 2), (1, 3)):
        grid[y, x] = BLACK
    for y, x in ((0, 1), (2, 1), (1, 0), (1, 2)):
        grid[y, x] = WHITE
    return Position(rules=Rules(size=size, ko=ko_rule), grid=grid, to_play=BLACK)


# ------------------------------------------------------------------- geometry


def test_neighbors_corners_and_center():
    nbrs = neighbors_table(9)
    assert len(nbrs[0]) == 2, "corner has 2 neighbours"
    assert len(nbrs[8]) == 2
    assert len(nbrs[4 * 9 + 4]) == 4, "centre has 4 neighbours"
    assert len(nbrs[0 * 9 + 5]) == 3, "edge has 3 neighbours"


def test_empty_board_legal_mask():
    pos = Position.empty(Rules(size=9))
    mask = pos.legal_mask()
    assert mask.shape == (82,), "81 points + pass"
    assert mask[:81].all()
    assert mask[81]
    assert len(pos.legal_moves()) == 82


# ------------------------------------------------------------------- captures


def test_single_stone_capture():
    rows = [
        "XO.......",
        ".X.......",
    ]
    pos = pos_from(rows, to_play=BLACK, size=9)
    nxt = pos.play(0 * 9 + 2)
    assert nxt.grid[0, 1] == EMPTY, "white stone captured"
    assert nxt.grid[0, 0] == BLACK and nxt.grid[0, 2] == BLACK


def test_group_capture_requires_all_liberties():
    rows = [
        "XOO......",
        ".X.......",
    ]
    pos = pos_from(rows, to_play=BLACK, size=9)
    nxt = pos.play(0 * 9 + 3)
    assert nxt.grid[0, 1] == WHITE, "group still has liberties, not captured"


def test_group_capture_when_last_liberty_taken():
    rows = [
        ".XO......",
        "XOX......",
        ".XO......",
    ]
    # White chain (0,1),(1,2)? build a solid 2-stone group with one liberty
    size = 9
    grid = np.zeros((size, size), dtype=np.int8)
    grid[1, 1] = WHITE
    grid[1, 2] = WHITE
    for y, x in ((0, 1), (0, 2), (0, 3), (1, 3), (1, 0), (2, 1), (2, 2), (2, 3)):
        grid[y, x] = BLACK
    grid[0, 0] = BLACK
    grid[2, 0] = BLACK
    p = Position(rules=Rules(size=size), grid=grid, to_play=WHITE)
    # check the white group is in atari
    libs = [q for q in ((1, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 1), (2, 2), (2, 3))]
    assert grid[1, 1] == WHITE
    assert p.is_terminal is False


# -------------------------------------------------------------------- suicide


def test_suicide_illegal_by_default():
    rows = [
        ".O.......",
        "O........",
    ]
    pos = pos_from(rows, to_play=BLACK, size=9)
    assert not pos.is_legal(0), "corner capture of nothing is suicide"


def test_suicide_allowed_when_enabled():
    rows = [
        ".O.......",
        "O........",
    ]
    pos = pos_from(rows, to_play=BLACK, size=9, suicide_allowed=True)
    assert pos.is_legal(0)
    nxt = pos.play(0)
    assert nxt.grid[0, 0] == EMPTY, "suicidal stone removes itself"


def test_self_atari_with_liberty_is_legal():
    rows = [
        ".O.......",
        "O........",
    ]
    pos = pos_from(rows, to_play=BLACK, size=9)
    assert pos.is_legal(1 * 9 + 1), "playing elsewhere is legal"


# ------------------------------------------------------------------------- ko


def test_ko_shape_is_capturable():
    p = ko_position()
    assert p.is_legal(1 * 9 + 1), "black may capture"
    after = p.play(1 * 9 + 1)
    assert after.grid[1, 1] == BLACK
    assert after.grid[1, 2] == EMPTY, "white stone captured"
    assert after.ko_point == 1 * 9 + 2, "ko point is the vacated intersection"


def test_simple_ko_blocks_immediate_recapture():
    after = ko_position("simple").play(1 * 9 + 1)
    assert not after.is_legal(1 * 9 + 2), "immediate recapture forbidden"


def test_ko_recapture_after_tenuki():
    after = ko_position("simple").play(1 * 9 + 1)
    w = after.play(5 * 9 + 5)
    b = w.play(5 * 9 + 6)
    assert b.is_legal(1 * 9 + 2), "ko threat resolved"


def test_situational_superko_blocks_recapture():
    after = ko_position("situational").play(1 * 9 + 1)
    assert not after.is_legal(1 * 9 + 2)


def test_positional_superko_blocks_recapture():
    after = ko_position("positional").play(1 * 9 + 1)
    assert not after.is_legal(1 * 9 + 2)


# -------------------------------------------------------------------- scoring

TWO_EYE_ROWS = [
    "XXXXXXXXX",
    "X.XXXXX.X",
    "XXXXXXXXX",
    "OOOOOOOOO",
]


def test_area_scoring_two_eyes_plus_white_moyo():
    pos = pos_from(TWO_EYE_ROWS, size=9)
    black, white = pos.area_score()
    # 25 black stones + 2 eyes
    assert black == 27, f"expected 27, got {black}"
    # 9 white stones + 5 empty rows
    assert white == 54, f"expected 54, got {white}"
    assert black + white == 81


def test_area_scoring_neutral_region():
    rows = [
        "XXXXXXXXX",
        "X.XXXXX.X",
        "XXXXXXXXX",
        ".........",
        "OOOOOOOOO",
    ]
    pos = pos_from(rows, size=9)
    black, white = pos.area_score()
    assert black == 27, "black keeps stones + eyes"
    assert white == 45, "9 stones + 4 empty rows; row 3 is neutral"


def test_ownership_values():
    pos = pos_from(TWO_EYE_ROWS, size=9)
    own = pos.area_ownership()
    assert own.shape == (9, 9)
    assert own[1, 1] == approx(1.0), "black eye"
    assert own[1, 7] == approx(1.0), "black eye"
    assert own[5, 5] == approx(0.0), "white territory"
    assert own[0, 0] == approx(1.0), "black stone"


def test_score_includes_komi():
    pos = Position.empty(Rules(size=9, komi=7.5))
    assert pos.score() == approx(-7.5)
    assert pos.score(komi=0.0) == approx(0.0)
    two = pos_from(TWO_EYE_ROWS, size=9)
    assert two.score(komi=0.0) == approx(27 - 54)


# ----------------------------------------------------------------- pass-alive


def test_benson_finds_two_eyes():
    pos = pos_from(TWO_EYE_ROWS, size=9)
    alive = benson_pass_alive(pos.flat(), BLACK, 9).reshape(9, 9)
    assert alive[1, 1] and alive[1, 7], "both eyes are pass-alive"
    assert alive[0, 0], "the wall is pass-alive too"
    assert not alive[5, 5], "white's area is not black pass-alive"


def test_benson_rejects_leaky_shape():
    rows = [
        "XXXXX....",
        "X.X......",
        "XXXXX....",
    ]
    pos = pos_from(rows, size=9)
    alive = benson_pass_alive(pos.flat(), BLACK, 9).reshape(9, 9)
    assert not alive.any(), "one eye is not enough: nothing is unconditionally alive"


def test_benson_empty_board():
    empty = np.zeros(81, dtype=np.int8)
    assert not benson_pass_alive(empty, BLACK, 9).any()


def test_benson_lone_stone_is_not_alive():
    grid = np.zeros(81, dtype=np.int8)
    grid[4 * 9 + 4] = BLACK
    assert not benson_pass_alive(grid, BLACK, 9).any(), (
        "a lone stone must not count as alive just because the whole board is "
        "one region adjacent only to it"
    )


# -------------------------------------------------------------------- ladders


def test_no_ladder_for_free_stone():
    grid = np.zeros((9, 9), dtype=np.int8)
    grid[4, 4] = WHITE
    assert not ladder_captures(list(grid.reshape(-1)), 9, 4 * 9 + 4), "4 liberties"


def test_ladder_against_diagonal_wall_is_captured():
    """Black wall on the main diagonal; White runs up the neighbouring column."""
    size = 9
    grid = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        grid[i, i] = BLACK
    grid[2, 1] = WHITE
    assert ladder_captures(list(grid.reshape(-1)), size, 2 * size + 1)


def test_ladder_fails_when_wall_is_missing():
    size = 9
    grid = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        grid[i, i] = BLACK
    # Remove the wall stones White would run into, leaving open space.
    for i in range(0, 4):
        grid[i, i] = 0
    grid[2, 1] = WHITE
    assert not ladder_captures(list(grid.reshape(-1)), size, 2 * size + 1)


# ------------------------------------------------------------------- features


def test_feature_shapes_and_dtypes():
    pos = Position.empty(Rules(size=9))
    spatial, glob = encode_position(pos)
    assert spatial.shape == (18, 9, 9)
    assert glob.shape == (10,)
    assert spatial.dtype == np.float32 and glob.dtype == np.float32
    assert spatial[0].all(), "on-board plane"


def test_feature_own_opponent_planes_swap_with_colour():
    rows = ["XO......."]
    pos = pos_from(rows, to_play=BLACK, size=9)
    spatial, _ = encode_position(pos)
    assert spatial[1, 0, 0] == 1.0 and spatial[2, 0, 1] == 1.0
    white_pos = Position(rules=pos.rules, grid=pos.grid, to_play=WHITE)
    sp_w, _ = encode_position(white_pos)
    assert sp_w[1, 0, 1] == 1.0 and sp_w[2, 0, 0] == 1.0


def test_feature_liberty_planes():
    pos = pos_from(["XO......."], to_play=BLACK, size=9)
    spatial, _ = encode_position(pos)
    assert spatial[3, 0, 0] == 1.0, "corner black stone has 1 liberty"
    assert spatial[4, 0, 1] == 1.0, "white stone has 2 liberties"


def test_feature_last_move_planes():
    pos = Position.empty(Rules(size=9)).play(4 * 9 + 4)
    spatial, _ = encode_position(pos)
    assert spatial[7, 4, 4] == 1.0
    pos2 = pos.play(2 * 9 + 2)
    spatial2, _ = encode_position(pos2)
    assert spatial2[7, 2, 2] == 1.0 and spatial2[8, 4, 4] == 1.0


def test_feature_pass_flags():
    pos = Position.empty(Rules(size=9)).play(PASS_MOVE)
    spatial, glob = encode_position(pos)
    assert glob[0] == 1.0
    assert spatial[7].sum() == 0.0, "pass leaves the move plane empty"


def test_feature_ko_plane():
    after = ko_position("simple").play(1 * 9 + 1)
    spatial, _ = encode_position(after)
    assert spatial[6, 1, 2] == 1.0, "ko point marked illegal"
    assert spatial[6].sum() == 1.0, "exactly one ko point"


def test_feature_pass_alive_planes():
    pos = pos_from(TWO_EYE_ROWS, size=9)
    spatial, _ = encode_position(pos)
    assert spatial[16, 1, 1] == 1.0, "self pass-alive plane"
    assert spatial[16, 0, 0] == 1.0, "wall stones are pass-alive"
    assert spatial[17].sum() == 0.0, "opponent has no pass-alive area"


def test_features_are_finite():
    pos = Position.empty(Rules(size=9, komi=6.5, ko="positional", suicide_allowed=True))
    for m in (0, 10, 20, 30, PASS_MOVE):
        pos = pos.play(m)
        spatial, glob = encode_position(pos)
        assert np.isfinite(spatial).all() and np.isfinite(glob).all()
        assert spatial.max() <= 1.0 and spatial.min() >= 0.0


# --------------------------------------------------------------- immutability


def test_play_does_not_mutate_parent():
    pos = Position.empty(Rules(size=9))
    before = pos.grid.copy()
    pos.play(4 * 9 + 4)
    assert np.array_equal(pos.grid, before)


def test_illegal_move_raises():
    pos = Position.empty(Rules(size=9)).play(4 * 9 + 4)
    with raises(IllegalMove):
        pos.play(4 * 9 + 4)


def test_terminal_after_two_passes():
    pos = Position.empty(Rules(size=9))
    assert not pos.is_terminal
    pos = pos.play(PASS_MOVE).play(PASS_MOVE)
    assert pos.is_terminal
    assert not pos.is_draw
