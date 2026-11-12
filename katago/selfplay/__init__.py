"""Self-play data generation, replay, gating and the training loop."""

from __future__ import annotations

from .gating import (
    MatchResult,
    evaluate_vs_net,
    evaluate_vs_random,
    gating_test,
    play_match_game,
)
from .pool import SelfPlayPool
from .replay import ReplayBuffer
from .trainer import IterationStats, Trainer
from .worker import GameRecord, SelfPlayConfig, play_game

__all__ = [
    "MatchResult",
    "evaluate_vs_net",
    "evaluate_vs_random",
    "gating_test",
    "play_match_game",
    "SelfPlayPool",
    "ReplayBuffer",
    "IterationStats",
    "Trainer",
    "GameRecord",
    "SelfPlayConfig",
    "play_game",
]
