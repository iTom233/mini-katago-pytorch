#!/usr/bin/env python
"""Run the engine as a GTP bot, so any Go GUI can play against it.

    python scripts/gtp.py --checkpoint runs/demo9x9/model.npz --visits 400

The network shape is read from the checkpoint, so nothing else is needed.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.game import Rules  # noqa: E402
from katago.inference import GTPEngine  # noqa: E402
from scripts._shared import build_from_checkpoint  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    p.add_argument("--size", type=int, default=None, help="override the board size")
    p.add_argument("--blocks", type=int, default=None, help="override residual blocks")
    p.add_argument("--channels", type=int, default=None, help="override trunk channels")
    p.add_argument("--pool-blocks", type=int, default=None)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--visits", type=int, default=400)
    p.add_argument("--resign-threshold", type=float, default=None,
                   help="e.g. -0.95 to let the engine resign")
    return p.parse_args(argv)


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
    engine = GTPEngine(
        net,
        Rules(size=cfg.board_size, komi=args.komi),
        visits=args.visits,
        resign_threshold=args.resign_threshold,
    )
    engine.loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
