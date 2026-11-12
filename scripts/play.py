#!/usr/bin/env python
"""Play against (or watch) a trained engine in the terminal.

    python scripts/play.py --checkpoint runs/demo/model.npz --size 9 --visits 200
    python scripts/play.py --checkpoint runs/demo/model.npz --self
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.game import BLACK, PASS_MOVE, WHITE, Position, Rules  # noqa: E402
from katago.inference import Engine, move_to_gtp  # noqa: E402
from katago.utils.sgf import to_sgf  # noqa: E402
from scripts._shared import build_from_checkpoint  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    p.add_argument("--size", type=int, default=None, help="override the board size")
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--pool-blocks", type=int, default=None)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--visits", type=int, default=200)
    p.add_argument("--self", action="store_true", help="engine plays itself")
    p.add_argument("--engine-plays", default="white", choices=["black", "white", "both"])
    p.add_argument("--save-sgf", default=None)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = always the most-visited move; >0 samples among the candidates "
        "with weight N**(1/T).  1.0 samples in proportion to visit counts, "
        "smaller values stay closer to the leader",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="only consider the K most-visited moves (0 = no limit)",
    )
    p.add_argument(
        "--min-visit-ratio",
        type=float,
        default=0.0,
        help="drop candidates whose visits are below this fraction of the top "
        "move's, e.g. 0.3 keeps moves with at least 30%% of the leader's visits",
    )
    p.add_argument(
        "--opening-moves",
        type=int,
        default=0,
        help="play this many uniformly random legal moves before the engine "
        "starts, so repeated runs produce different games",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for temperature/opening randomness (default: a fresh "
        "seed each run, so games differ; pass a value to reproduce one)",
    )
    return p.parse_args(argv)


def board_with_coords(pos: Position) -> str:
    size = pos.size
    letters = "ABCDEFGHJKLMNOPQRST"
    rows = ["   " + " ".join(letters[:size])]
    for y in range(size):
        cells = []
        for x in range(size):
            v = pos.grid[y, x]
            cells.append("X" if v == 1 else ("O" if v == 2 else "."))
        rows.append(f"{size - y:2d} " + " ".join(cells) + f" {size - y}")
    rows.append("   " + " ".join(letters[:size]))
    return "\n".join(rows)


def choose_move(
    analysis,
    temperature: float,
    top_k: int,
    min_visit_ratio: float,
    rng: np.random.Generator,
) -> int:
    """Pick a move from the visit distribution, with an optional quality floor.

    ``temperature == 0`` returns the most-visited move (the engine's first
    choice).  With a positive temperature the candidates are sampled with weight
    ``N ** (1 / T)``, optionally restricted to the ``top_k`` most-visited moves
    and to those holding at least ``min_visit_ratio`` of the leader's visits.
    """
    weights = np.asarray(analysis.policy, dtype=np.float64)
    order = [int(m) for m in np.argsort(weights)[::-1]]
    if top_k > 0:
        order = order[:top_k]
    leader = weights[order[0]] if order else 0.0
    if min_visit_ratio > 0.0 and leader > 0.0:
        order = [m for m in order if weights[m] >= leader * min_visit_ratio]
    if not order:
        return PASS_MOVE
    if temperature <= 1e-3 or len(order) == 1:
        return int(order[0])
    w = np.array([weights[m] for m in order], dtype=np.float64)
    if w.sum() <= 0.0:
        return int(order[0])
    w = np.power(w, 1.0 / temperature)
    w /= w.sum()
    return int(rng.choice(order, p=w))


def main(argv=None) -> int:
    args = parse_args(argv)
    net, cfg = build_from_checkpoint(
        backend=args.backend,
        checkpoint=args.checkpoint,
        size=args.size,
        blocks=args.blocks,
        channels=args.channels,
        pool_blocks=args.pool_blocks,
    )
    print(f"loaded {args.checkpoint}  ({net!r})")

    rules = Rules(size=cfg.board_size, komi=args.komi)
    board_size = rules.size
    engine = Engine(net, rules, visits=args.visits, seed=0)
    rng = np.random.default_rng(args.seed)
    pos = Position.empty(rules)
    moves: list[int] = []

    # Random opening: lets repeated runs explore different games.  Both sides
    # get the same treatment, so it is fair.
    opening: list[str] = []
    for _ in range(args.opening_moves):
        if pos.is_terminal:
            break
        legal = pos.legal_moves()
        if not legal:
            break
        move = int(rng.choice(legal))
        opening.append(move_to_gtp(move, board_size))
        moves.append(move)
        pos = pos.play(move)
    if opening:
        print(f"random opening: {' '.join(opening)}")

    print(board_with_coords(pos))
    while not pos.is_terminal:
        engine_turn = args.self or args.engine_plays == "both" or (
            (pos.to_play == BLACK and args.engine_plays == "black")
            or (pos.to_play == WHITE and args.engine_plays == "white")
        )
        if engine_turn:
            analysis = engine.analyze(pos, visits=args.visits)
            move = choose_move(
                analysis,
                args.temperature,
                args.top_k,
                args.min_visit_ratio,
                rng,
            )
            top = " ".join(
                f"{move_to_gtp(m, board_size)}:{p:.2f}" for m, p in analysis.top_moves(3)
            )
            note = "" if args.temperature <= 1e-3 else " (sampled)"
            print(
                f"\nengine ({'black' if pos.to_play == BLACK else 'white'}): "
                f"{move_to_gtp(move, board_size)}{note}  "
                f"winrate {analysis.win_rate:.1%}  scoreLead {analysis.score_mean:+.1f}  [{top}]"
            )
        else:
            human_color = "black" if pos.to_play == BLACK else "white"
            raw = input(f"\nyour move ({human_color}, e.g. E5, or 'pass'): ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                return 0
            if raw.lower() in ("pass", ""):
                move = PASS_MOVE
            else:
                from katago.inference.engine import gtp_to_move

                try:
                    move = gtp_to_move(raw, board_size)
                except ValueError as exc:
                    print(f"  {exc}")
                    continue
        moves.append(move)
        pos = pos.play(move)
        print()
        print(board_with_coords(pos))

    score = pos.score()
    print()
    if abs(score) < 1e-9:
        print(f"draw (komi {args.komi})")
    else:
        print(f"{'Black' if score > 0 else 'White'} wins by {abs(score):.1f}")

    if args.save_sgf:
        with open(args.save_sgf, "w", encoding="utf-8") as f:
            f.write(to_sgf(moves, board_size, args.komi))
        print(f"game saved to {args.save_sgf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
