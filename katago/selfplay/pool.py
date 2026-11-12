"""Parallel self-play over a persistent pool of worker processes.

The games of one iteration are independent, so they can be spread over several
processes.  Almost all of a self-play game's cost is Python-level work (rules,
``legal_mask``, ``encode_position``), which never releases the GIL, so threads
give no speedup -- separate processes are required.

Each worker builds its own copy of the network once (the torch backend also
builds its own CUDA context) and only reloads weights when the trainer
publishes a new version.  The pool is created once and reused for the whole
run: under the ``spawn`` start method every child re-imports torch, which costs
a few seconds and is not worth paying once per iteration.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import List, Optional, Sequence

import numpy as np

from ..game import Rules
from ..nn.config import MCTSConfig, NetConfig, PlayoutCapConfig
from .worker import GameRecord, SelfPlayConfig, play_game

# Per-process state, populated once by ``_init_worker``.
_NET = None
_RULES: Optional[Rules] = None
_VERSION = -1


def _init_worker(
    net_config: NetConfig,
    rules: Rules,
    backend: str,
    device: str,
    seed: int,
) -> None:
    global _NET, _RULES
    from ..nn import build_net

    kwargs = {"seed": seed}
    if backend == "torch":
        kwargs["device"] = device
    _NET = build_net(net_config, backend=backend, **kwargs)
    _RULES = rules


def _play_batch(task) -> List[GameRecord]:
    global _VERSION
    (
        state,
        version,
        seeds,
        mcts_config,
        cap_config,
        selfplay_config,
        samples_seen,
    ) = task
    # Weights are only reloaded when the trainer published a new version, so
    # several games at the same version share one copy.
    if version != _VERSION:
        _NET.load_state_dict(state)
        _VERSION = version
    records: List[GameRecord] = []
    seen = samples_seen
    for seed in seeds:
        rng = np.random.default_rng(seed)
        record = play_game(
            _NET,
            _RULES,
            mcts_config,
            cap_config,
            selfplay_config,
            rng,
            samples_seen=seen,
        )
        seen += record.num_samples
        records.append(record)
    return records


class SelfPlayPool:
    """A persistent process pool that plays self-play games in parallel."""

    def __init__(
        self,
        workers: int,
        net_config: NetConfig,
        rules: Rules,
        backend: str,
        device: str,
        seed: int = 0,
    ) -> None:
        if workers < 2:
            raise ValueError("SelfPlayPool needs at least 2 workers")
        self.workers = int(workers)
        self.backend = backend
        self._version = 0
        # ``spawn`` everywhere: forking a process that already initialised CUDA
        # is unsafe, and it also makes Windows and Linux behave identically.
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=self.workers,
            initializer=_init_worker,
            initargs=(net_config, rules, backend, device, seed),
        )

    def play(
        self,
        state,
        games: int,
        seeds: Sequence[int],
        mcts_config: MCTSConfig,
        cap_config: PlayoutCapConfig,
        selfplay_config: SelfPlayConfig,
        samples_seen: int = 0,
    ) -> List[GameRecord]:
        """Play ``games`` games with the given weights and return their records."""
        self._version += 1
        version = self._version
        # Hand each worker a small block of games instead of one game per task.
        # The weights travel with the task, so batching keeps the network off
        # the pipe for all but ``len(groups)`` transfers: at 4.7 MB a copy and
        # 32 tasks this was ~150 MB of pipe traffic per iteration, plus one
        # weight upload per game on every worker.  Two blocks per worker keeps
        # the pool balanced without going back to a copy per game.
        per = max(1, games // (self.workers * 2))
        groups: List[List[int]] = []
        start = 0
        while start < games:
            stop = min(games, start + per)
            groups.append(list(range(start, stop)))
            start = stop
        tasks = [
            (
                state,
                version,
                [int(seeds[j]) for j in group],
                mcts_config,
                cap_config,
                selfplay_config,
                samples_seen,
            )
            for group in groups
        ]
        # chunksize=1 lets the faster cores pick up more blocks as they finish.
        batches = self._pool.map(_play_batch, tasks, chunksize=1)
        return [record for batch in batches for record in batch]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "SelfPlayPool":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
