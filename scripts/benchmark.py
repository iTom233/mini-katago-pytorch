#!/usr/bin/env python
"""Micro-benchmarks: feature encoding, forward pass, and search throughput.

    python scripts/benchmark.py --backend numpy --size 9 --blocks 3 --channels 24
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.game import Position, Rules, encode_position  # noqa: E402
from katago.mcts import MCTS  # noqa: E402
from katago.nn import NetConfig, build_net  # noqa: E402
from katago.nn.config import MCTSConfig  # noqa: E402


def timeit(fn, n=5):
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--channels", type=int, default=24)
    p.add_argument("--pool-blocks", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--playouts", type=int, default=64)
    args = p.parse_args(argv)

    rules = Rules(size=args.size)
    pos = Position.empty(rules)
    for m in [18, 22, 30, 40, 50, 60, 12, 20]:
        if pos.is_legal(m):
            pos = pos.play(m)

    t = timeit(lambda: encode_position(pos), 3)
    print(f"encode_position          {t * 1e3:7.2f} ms")

    cfg = NetConfig(
        board_size=args.size,
        block_count=args.blocks,
        channels=args.channels,
        pool_channels=max(8, args.channels // 2),
        pool_blocks=args.pool_blocks,
    )
    net = build_net(cfg, backend=args.backend, seed=0)
    print(f"network                  {net!r}")

    sp, gl = encode_position(pos)
    legal = pos.legal_mask()
    sp_b = np.repeat(sp[None], args.batch, axis=0)
    gl_b = np.repeat(gl[None], args.batch, axis=0)
    lg_b = np.repeat(legal[None], args.batch, axis=0)

    t1 = timeit(lambda: net.predict(sp[None], gl[None], legal[None]), 3)
    print(f"predict batch=1          {t1 * 1e3:7.2f} ms  ({1 / t1:6.0f} pos/s)")

    tb = timeit(lambda: net.predict(sp_b, gl_b, lg_b), 3)
    print(
        f"predict batch={args.batch:<2d}         {tb * 1e3:7.2f} ms  "
        f"({args.batch / tb:6.0f} pos/s)"
    )

    def search():
        mcts = MCTS(net, MCTSConfig(), batch_size=args.batch, rng=np.random.default_rng(0))
        mcts.set_root(pos, add_noise=True)
        mcts.run(args.playouts)

    ts = timeit(search, 2)
    evals = args.playouts + 1
    print(
        f"search {args.playouts} playouts   {ts * 1e3:7.1f} ms  "
        f"({ts / evals * 1e3:5.2f} ms/position, {1 / (ts / evals):6.0f} pos/s)"
    )
    print()
    print(
        f"estimated cost of one 9x9 self-play game "
        f"(~80 moves at {args.playouts} playouts): {ts * 80 / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
