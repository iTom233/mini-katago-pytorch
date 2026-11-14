#!/usr/bin/env python
"""Train a KataGo-style Go engine by self-play.

Examples
--------
Quick CPU demo (NumPy backend, a few minutes)::

    python scripts/train.py --backend numpy --size 9 --blocks 3 --channels 24 \
        --iterations 40 --games 2 --steps 40 --full-cap 80 --fast-cap 20 \
        --eval-every 5 --out runs/demo

The paper-faithful configuration on a GPU::

    python scripts/train.py --backend torch --size 19 --blocks 20 --channels 256 \
        --iterations 100000 --games 32 --steps 500 --full-cap 600 --fast-cap 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.game import KO_POSITIONAL, KO_SIMPLE, KO_SITUATIONAL, Rules  # noqa: E402
from katago.nn import (
    BackendUnavailable,
    NetConfig,
    build_net,
    resolve_torch_device,
    torch_available,
)  # noqa: E402
from katago.nn.config import (  # noqa: E402
    LossConfig,
    MCTSConfig,
    PlayoutCapConfig,
    TrainConfig,
)
from katago.selfplay import SelfPlayConfig, Trainer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    p.add_argument("--device", default="auto",
                   help="torch device: auto (cuda when usable) | cpu | cuda | cuda:1")
    p.add_argument("--size", type=int, default=9, help="board size (9..19)")
    p.add_argument("--blocks", type=int, default=3, help="residual blocks")
    p.add_argument("--channels", type=int, default=24, help="trunk channels")
    p.add_argument("--pool-blocks", type=int, default=2, help="blocks using global pooling")
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--ko", default="situational",
                   choices=[KO_SIMPLE, KO_POSITIONAL, KO_SITUATIONAL])
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--games", type=int, default=2, help="self-play games per iteration")
    p.add_argument("--steps", type=int, default=40, help="gradient steps per iteration")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-3,
                   help="step size on the batch-mean gradient (paper: 6e-5 per "
                        "sample at batch 256, i.e. 0.0154 here)")
    p.add_argument("--lr-warmup-samples", type=int, default=20_000)
    p.add_argument("--full-cap", type=int, default=80, help="playouts on a full-search turn")
    p.add_argument("--fast-cap", type=int, default=20, help="playouts on a fast-search turn")
    p.add_argument("--full-fraction", type=float, default=0.25)
    p.add_argument("--leaf-batch", type=int, default=16,
                   help="leaves evaluated in one network forward pass; larger is "
                        "faster on a GPU but adds virtual-loss bias to the search")
    p.add_argument("--no-cap-randomization", action="store_true")
    p.add_argument("--replay-window", type=int, default=60_000)
    p.add_argument("--swa-every", type=int, default=20_000,
                   help="snapshot every N training samples (0 disables SWA)")
    p.add_argument("--gating-games", type=int, default=10)
    p.add_argument("--no-gating", action="store_true",
                   help="refresh the self-play network every iteration instead of "
                        "using the SWA + gating pipeline (useful for short runs)")
    p.add_argument("--eval-every", type=int, default=5, help="0 disables evaluation")
    p.add_argument(
        "--eval-games",
        type=int,
        default=10,
        help="games per evaluation; the 95%% confidence interval is roughly "
        "+/-16%% at 10 games, +/-6%% at 64 and +/-3.5%% at 200",
    )
    p.add_argument(
        "--eval-opening-moves",
        type=int,
        default=4,
        help="random opening moves in eval games. Search is deterministic, so "
        "without this every game between the same two nets is a replay of one "
        "position per colour and the reported win rate can only be "
        "0%%, 50%% or 100%%",
    )
    p.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="sample eval moves at this temperature instead of playing the top "
        "move (0 = greedy; >0 adds variety but lowers playing strength)",
    )
    p.add_argument("--eval-opponent", default="random",
                   help="'random' (uniform legal moves) or a checkpoint path to "
                        "evaluate against, e.g. runs/old/model.npz. A path is loaded "
                        "once and kept frozen for the whole run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0,
                   help="self-play worker processes; 0 or 1 keeps games in the main "
                        "process. Each worker holds its own copy of the network, so "
                        "on a GPU this multiplies VRAM use")
    p.add_argument("--min-pass-move", type=int, default=0,
                   help="self-play forbids passing before this many moves; helps small "
                        "runs avoid the degenerate 'pass immediately' equilibrium")
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--resume", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    backend = args.backend
    if backend == "auto":
        backend = "torch" if torch_available() else "numpy"
    if backend == "torch" and not torch_available():
        print("PyTorch is not installed; falling back to the NumPy backend", file=sys.stderr)
        backend = "numpy"

    os.makedirs(args.out, exist_ok=True)
    net_cfg = NetConfig(
        board_size=args.size,
        block_count=args.blocks,
        channels=args.channels,
        pool_channels=max(8, args.channels // 2),
        pool_blocks=args.pool_blocks,
    )
    rules = Rules(size=args.size, komi=args.komi, ko=args.ko)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        lr_warmup=args.lr * 0.33,
        lr_warmup_samples=args.lr_warmup_samples,
        window_start=min(args.replay_window, 8_000),
        window_max=args.replay_window,
        swa_snapshot_samples=args.swa_every or 10**12,
        gating_games=args.gating_games,
        gating_wins=(args.gating_games + 1) // 2,
        gating_enabled=not args.no_gating,
    )
    cap_cfg = PlayoutCapConfig(
        enabled=not args.no_cap_randomization,
        full_fraction=args.full_fraction,
        full_cap=args.full_cap,
        fast_cap=args.fast_cap,
        full_cap_late=None,
        fast_cap_late=None,
    )

    device = resolve_torch_device(args.device) if backend == "torch" else "cpu"
    net = build_net(
        net_cfg,
        backend=backend,
        seed=args.seed,
        **({"device": device} if backend == "torch" else {}),
    )
    print(f"backend={backend}  {net!r}")
    if backend == "torch":
        print(f"device={device}")
    print(f"board={args.size}x{args.size} komi={args.komi} ko={args.ko}")
    print(
        f"playouts per turn: {args.full_cap} (full, p={args.full_fraction}) / "
        f"{args.fast_cap} (fast)"
    )

    if args.workers > 1:
        print(f"self-play workers: {args.workers} processes")

    trainer = Trainer(
        net,
        rules,
        train_config=train_cfg,
        loss_config=LossConfig(),
        mcts_config=MCTSConfig(),
        cap_config=cap_cfg,
        selfplay_config=SelfPlayConfig(
            batch_size=max(1, args.leaf_batch),
            temperature_moves=12,
            min_pass_move=args.min_pass_move,
        ),
        backend=backend,
        device=device,
        seed=args.seed,
        workers=args.workers,
        eval_opponent=args.eval_opponent,
        log=print,
    )
    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume}")

    # Load the eval opponent up front so a bad path, a board-size mismatch or a
    # checkpoint without an architecture config fails at startup instead of on
    # the first eval round, which may be hundreds of iterations away.
    if args.eval_opponent != "random" and args.eval_every > 0:
        trainer._get_eval_net()

    # Two overwrite-only checkpoints written by the trainer while it runs:
    #   model.npz      every 10% of the run (and the final iteration);
    #   model_best.npz only when an iteration loss reaches a new low.
    ckpt = os.path.join(args.out, "model.npz")
    best_ckpt = os.path.join(args.out, "model_best.npz")

    start = time.perf_counter()
    try:
        history = trainer.train(
            iterations=args.iterations,
            games_per_iteration=args.games,
            steps_per_iteration=args.steps,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            eval_opening_moves=args.eval_opening_moves,
            eval_temperature=args.eval_temperature,
            checkpoint_path=ckpt,
            best_checkpoint_path=best_ckpt,
        )
    except KeyboardInterrupt:
        # Ctrl+C leaves the worker pool alive; tear it down explicitly and keep
        # the last checkpoint instead of orphaning eight processes on the GPU.
        trainer.close()
        trainer.save(ckpt)
        print(f"\ninterrupted after {len(trainer.history)} iteration(s); saved {ckpt}")
        return 130
    elapsed = time.perf_counter() - start

    if not os.path.exists(ckpt):
        trainer.save(ckpt)
    payload = {
        "args": vars(args),
        "elapsed_seconds": elapsed,
        "backend": backend,
        "parameters": net.num_parameters(),
        "history": [
            {
                "iteration": h.iteration,
                "loss": h.loss,
                "lr": h.lr,
                "samples": h.samples,
                "trained": h.total_samples,
                "replay_games": h.replay_games,
                "replay_samples": h.replay_samples,
                "elapsed": h.elapsed,
                "accepted": h.accepted,
                "win_rate_vs_random": h.win_rate_vs_random,
                "eval_opponent": h.eval_opponent,
                "terms": h.terms,
            }
            for h in history
        ],
    }
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"done in {elapsed / 60:.1f} min; checkpoint {ckpt}")
    if os.path.exists(best_ckpt):
        print(f"best-loss checkpoint {best_ckpt}")
    print(f"positions generated: {trainer.total_samples_produced:,}")
    print(f"samples trained on:  {trainer.total_samples_trained:,}")
    if history and history[0].loss and history[-1].loss:
        print(f"loss {history[0].loss:.3f} -> {history[-1].loss:.3f}")
    rates = [h.win_rate_vs_random for h in history if h.win_rate_vs_random is not None]
    if len(rates) >= 2:
        label = next((h.eval_opponent for h in history if h.eval_opponent), "random")
        print(f"win rate vs {label}: {rates[0]:.1%} -> {rates[-1]:.1%}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BackendUnavailable as exc:  # pragma: no cover - environment dependent
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
