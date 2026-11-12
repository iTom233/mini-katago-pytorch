"""The self-play training loop.

Reproduces the pieces of KataGo's training process that live outside the network
and the search (Wu 2020, Section 2):

* a growing replay window (250k -> 22M samples),
* a per-sample learning rate schedule (2e-5 warm-up, 6e-5, optional 6e-6 decay),
* stochastic weight averaging over the last four snapshots with decay 0.75,
* a gating test before a candidate replaces the self-play network,
* progressive widening: the next larger network is trained concurrently on the
  same data and promoted once its loss catches up with the incumbent's.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..game import Rules
from ..nn.base import BaseNet
from ..nn.config import LossConfig, MCTSConfig, NetConfig, PlayoutCapConfig, TrainConfig
from .gating import evaluate_vs_net, evaluate_vs_random, gating_test
from .replay import ReplayBuffer
from .worker import GameRecord, SelfPlayConfig, play_game


@dataclass
class IterationStats:
    iteration: int
    games: int
    samples: int
    total_samples: int
    loss: float
    lr: float
    replay_games: int
    replay_samples: int
    elapsed: float
    widened_to: Optional[str] = None
    accepted: Optional[bool] = None
    win_rate_vs_random: Optional[float] = None
    eval_opponent: Optional[str] = None
    terms: Dict[str, float] = field(default_factory=dict)


class Trainer:
    def __init__(
        self,
        net: BaseNet,
        rules: Rules,
        train_config: Optional[TrainConfig] = None,
        loss_config: Optional[LossConfig] = None,
        mcts_config: Optional[MCTSConfig] = None,
        cap_config: Optional[PlayoutCapConfig] = None,
        selfplay_config: Optional[SelfPlayConfig] = None,
        backend: str = "auto",
        device: str = "cpu",
        seed: int = 0,
        workers: int = 0,
        eval_opponent: Optional[str] = None,
        log: Optional[Callable[[str], None]] = print,
    ) -> None:
        self.rules = rules
        self.train_config = train_config or TrainConfig()
        self.loss_config = loss_config or LossConfig()
        self.mcts_config = mcts_config or MCTSConfig()
        self.cap_config = cap_config or PlayoutCapConfig()
        self.selfplay_config = selfplay_config or SelfPlayConfig()
        self.backend = backend
        self.device = device if backend == "torch" else "cpu"
        self.seed = seed
        self.workers = max(0, int(workers))
        # "random", a checkpoint path, or None (same as "random")
        self.eval_opponent = eval_opponent or "random"
        self.rng = np.random.default_rng(seed)
        self.log = log or (lambda msg: None)

        self.train_net = net
        # a separate copy holds the gated weights used for self-play
        self.selfplay_net = self._build_net(net.config, seed + 1)
        self.selfplay_net.load_state_dict(net.state_dict())

        self.replay = ReplayBuffer(window_samples=self.train_config.window_start, seed=seed)
        self.snapshots: deque[Dict[str, np.ndarray]] = deque(maxlen=self.train_config.swa_snapshots)
        self.total_samples_produced = 0
        self.total_samples_trained = 0
        self.history: List[IterationStats] = []
        self._last_snapshot_at = 0

        # progressive widening
        self._widen_schedule = list(self.train_config.widening)
        self._next_net: Optional[BaseNet] = None

        # parallel self-play; created lazily on the first iteration that needs it
        self._pool = None

        # frozen evaluation opponent; built on the first eval round
        self._eval_net: Optional[BaseNet] = None

    # ------------------------------------------------------------------ utils
    def _build_net(self, config: NetConfig, seed: int) -> BaseNet:
        """Construct a network on the trainer's backend and device.

        Only the torch backend understands ``device``; the NumPy reference keeps
        its signature, so the argument is passed conditionally.
        """
        from ..nn import build_net

        kwargs = {"seed": seed}
        if self.backend == "torch":
            kwargs["device"] = self.device
        return build_net(config, backend=self.backend, **kwargs)

    def _get_eval_net(self) -> Optional[BaseNet]:
        """Load and cache the evaluation opponent, or ``None`` for random play.

        ``self.eval_opponent`` is either ``"random"`` or a checkpoint path.  The
        checkpoint is read once and then frozen, so every eval round measures
        progress against the same fixed strength — the weights never follow the
        training network.
        """
        if self.eval_opponent == "random":
            return None
        if self._eval_net is None:
            from ..nn.checkpoint import load_checkpoint

            cfg, state, _ = load_checkpoint(self.eval_opponent)
            if cfg is None:
                raise ValueError(
                    f"eval opponent {self.eval_opponent} has no architecture config"
                )
            if cfg.board_size != self.rules.size:
                raise ValueError(
                    f"eval opponent {self.eval_opponent} is "
                    f"{cfg.board_size}x{cfg.board_size}, but training runs on "
                    f"{self.rules.size}x{self.rules.size}"
                )
            self._eval_net = self._build_net(cfg, self.seed + 3)
            self._eval_net.load_state_dict(state)
            self.log(f"eval opponent: {self.eval_opponent}  ({self._eval_net!r})")
        return self._eval_net

    def _play_batch(self, games: int) -> List[GameRecord]:
        """Play ``games`` self-play games with the current self-play weights.

        With ``workers > 1`` the games run in a persistent process pool; the
        serial path keeps the original per-game ``samples_seen`` progression so
        seeded runs are unchanged.
        """
        if self.workers > 1:
            if self._pool is None:
                from .pool import SelfPlayPool

                self._pool = SelfPlayPool(
                    workers=self.workers,
                    net_config=self.selfplay_net.config,
                    rules=self.rules,
                    backend=self.backend,
                    device=self.device,
                    seed=self.seed,
                )
                self.log(f"self-play pool: {self.workers} worker processes")
            seeds = self.rng.integers(0, 2**31 - 1, size=games)
            return self._pool.play(
                self.selfplay_net.state_dict(),
                games,
                seeds,
                self.mcts_config,
                self.cap_config,
                self.selfplay_config,
                samples_seen=self.total_samples_produced,
            )

        records: List[GameRecord] = []
        seen = self.total_samples_produced
        for _ in range(games):
            record = play_game(
                self.selfplay_net,
                self.rules,
                self.mcts_config,
                self.cap_config,
                self.selfplay_config,
                self.rng,
                samples_seen=seen,
            )
            seen += record.num_samples
            records.append(record)
        return records

    def close(self) -> None:
        """Shut down the self-play pool, if one was started."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _snapshot(self) -> None:
        self.snapshots.append(self.train_net.state_dict())

    def _swa_candidate(self) -> Optional[Dict[str, np.ndarray]]:
        """Exponential moving average over the stored snapshots (decay 0.75).

        Seeded with the oldest snapshot so the mixture weights sum to one; the
        newest snapshot therefore carries the largest weight.
        """
        if len(self.snapshots) < self.train_config.swa_snapshots:
            return None
        decay = self.train_config.swa_decay
        items = list(self.snapshots)
        acc: Dict[str, np.ndarray] = {
            k: v.astype(np.float64).copy() for k, v in items[0].items()
        }
        for snap in items[1:]:  # oldest -> newest
            for k, v in snap.items():
                acc[k] = acc[k] * decay + v.astype(np.float64) * (1.0 - decay)
        return {k: v.astype(np.float32) for k, v in acc.items()}

    def _maybe_accept_candidate(self, iteration: int) -> Optional[bool]:
        candidate = self._swa_candidate()
        if candidate is None:
            return None

        probe = self._build_net(self.selfplay_net.config, self.seed + 7)
        probe.load_state_dict(candidate)
        cfg = self.train_config
        accepted, result = gating_test(
            probe,
            self.selfplay_net,
            self.rules,
            games=cfg.gating_games,
            wins_needed=cfg.gating_wins,
            visits=max(16, self.cap_config.fast_cap // 2),
            mcts_config=self.mcts_config,
            rng=self.rng,
        )
        self.log(
            f"  gating test @iter {iteration}: {'ACCEPT' if accepted else 'reject'} {result}"
        )
        if accepted:
            self.selfplay_net.load_state_dict(candidate)
            self.snapshots.clear()
        return accepted

    def _start_widening(self, block_count: int, channels: int) -> None:
        cfg = NetConfig(
            board_size=self.rules.size,
            block_count=block_count,
            channels=channels,
            pool_channels=max(8, channels // 2),
            pool_blocks=self.train_net.config.pool_blocks,
        )
        self._next_net = self._build_net(cfg, self.seed + 11)
        self.log(f"  widening: started training a {block_count}x{channels} candidate")

    # -------------------------------------------------------------- training
    def train(
        self,
        iterations: int = 10,
        games_per_iteration: int = 4,
        steps_per_iteration: int = 50,
        eval_every: int = 0,
        eval_games: int = 10,
        checkpoint_path: Optional[str] = None,
        best_checkpoint_path: Optional[str] = None,
    ) -> List[IterationStats]:
        cfg = self.train_config
        batch_size = cfg.batch_size
        # Two independent, overwrite-only checkpoints:
        #   * ``checkpoint_path``      -- every 10% of the run and the final
        #     iteration, so an interrupted run still leaves a recent model;
        #   * ``best_checkpoint_path`` -- only when the iteration loss reaches a
        # new low, so the best weights survive the final overwrite.
        save_every = max(1, iterations // 10)
        best_loss = float("inf")

        for it in range(iterations):
            start = time.perf_counter()

            # ---------------- self-play with the gated network
            records = self._play_batch(games_per_iteration)
            for record in records:
                self.replay.add_game(record)
                self.total_samples_produced += record.num_samples
            self.replay.set_window(cfg.window_size(self.total_samples_produced))

            # ---------------- gradient steps
            metrics: Dict[str, float] = {"loss": 0.0}
            for _ in range(steps_per_iteration):
                if len(self.replay) == 0:
                    break
                batch = self.replay.sample_batch(batch_size, rng=self.rng, size=self.rules.size)
                lr = cfg.learning_rate(self.total_samples_trained)
                metrics = self.train_net.train_batch(batch, self.loss_config, lr)
                self.total_samples_trained += batch_size

                # concurrent widening candidate trains on the same batches
                if self._next_net is not None:
                    next_metrics = self._next_net.train_batch(batch, self.loss_config, lr)
                    if next_metrics["loss"] <= metrics["loss"]:
                        self.log(
                            f"  widening: candidate loss {next_metrics['loss']:.4f} "
                            f"<= incumbent {metrics['loss']:.4f}, promoting"
                        )
                        self.train_net = self._next_net
                        self.selfplay_net = self._next_net
                        self._next_net = None
                        self.snapshots.clear()
                        if self._widen_schedule:
                            self._widen_schedule.pop(0)

                if (
                    cfg.gating_enabled
                    and self.total_samples_trained - self._last_snapshot_at
                    >= cfg.swa_snapshot_samples
                ):
                    self._last_snapshot_at = self.total_samples_trained
                    self._snapshot()

            accepted = None
            if cfg.gating_enabled and len(self.snapshots) >= cfg.swa_snapshots:
                accepted = self._maybe_accept_candidate(it)
            elif not cfg.gating_enabled:
                # demo mode: the self-play network follows the training weights
                self.selfplay_net.load_state_dict(self.train_net.state_dict())

            # progressive widening: keep a larger candidate training on the same
            # batches; it is promoted once its loss catches up with the incumbent
            if self._next_net is None and self._widen_schedule:
                blocks, channels = self._widen_schedule[0]
                if (blocks, channels) != (
                    self.train_net.config.block_count,
                    self.train_net.config.channels,
                ):
                    self._start_widening(blocks, channels)
                else:
                    self._widen_schedule.pop(0)

            win_rate = None
            eval_label = None
            if eval_every and (it + 1) % eval_every == 0:
                visits = max(8, self.cap_config.fast_cap)
                opponent = self._get_eval_net()
                if opponent is None:
                    eval_label = "random"
                    win_rate = evaluate_vs_random(
                        self.selfplay_net, self.rules, games=eval_games,
                        visits=visits, rng=self.rng,
                    )
                else:
                    eval_label = os.path.basename(self.eval_opponent)
                    win_rate = evaluate_vs_net(
                        self.selfplay_net, opponent, self.rules, games=eval_games,
                        visits=visits, mcts_config=self.mcts_config, rng=self.rng,
                    )
                self.log(f"  eval @iter {it}: win rate vs {eval_label} {win_rate:.1%}")

            stats = IterationStats(
                iteration=it,
                games=games_per_iteration,
                samples=self.total_samples_produced,
                total_samples=self.total_samples_trained,
                loss=float(metrics.get("loss", 0.0)),
                lr=cfg.learning_rate(self.total_samples_trained),
                replay_games=len(self.replay),
                replay_samples=self.replay.num_samples,
                elapsed=time.perf_counter() - start,
                accepted=accepted,
                win_rate_vs_random=win_rate,
                eval_opponent=eval_label,
                terms={k: float(v) for k, v in metrics.items() if k != "loss"},
            )
            self.history.append(stats)
            self.log(
                f"iter {it + 1}/{iterations}  games={games_per_iteration}  "
                f"samples={self.total_samples_produced}  loss={stats.loss:.4f}  "
                f"lr={stats.lr:.2e}  replay={stats.replay_games}g/"
                f"{stats.replay_samples}s  {stats.elapsed:.1f}s"
            )

            if checkpoint_path is not None:
                interval = (it + 1) % save_every == 0
                if interval or it + 1 == iterations:
                    self.save(checkpoint_path)
                    self.log(
                        f"  checkpoint @iter {it + 1}  loss={stats.loss:.4f}"
                        f"{'' if interval else '  (final)'}  -> {checkpoint_path}"
                    )

            if best_checkpoint_path is not None and stats.loss < best_loss:
                best_loss = stats.loss
                self.save(best_checkpoint_path)
                self.log(
                    f"  best checkpoint @iter {it + 1}  loss={stats.loss:.4f}"
                    f"  -> {best_checkpoint_path}"
                )
        self.close()
        return self.history

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        from ..nn.checkpoint import save_checkpoint

        save_checkpoint(
            path,
            self.selfplay_net,
            meta={
                "total_samples_produced": self.total_samples_produced,
                "total_samples_trained": self.total_samples_trained,
                "backend": self.backend,
                "board_size": self.rules.size,
                "komi": self.rules.komi,
                "ko": self.rules.ko,
            },
        )

    def load(self, path: str) -> None:
        from ..nn.checkpoint import load_checkpoint

        _, state, _ = load_checkpoint(path)
        if not state:
            raise ValueError(f"{path} contains no weights")
        self.selfplay_net.load_state_dict(state)
        self.train_net.load_state_dict(state)
