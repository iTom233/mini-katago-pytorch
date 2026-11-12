"""Configuration objects.

Hyper-parameters are named after the papers so the mapping is easy to check:

* ``c_puct = 1.1``, ``c_fpu = 0.2``, root Dirichlet weight ``0.25``,
  ``alpha = 0.03 * 19**2 / N_legal``, root softmax temperature ``1.03``,
  forced-playout ``k = 2``  -- Wu 2020, Sections 2 / 3.1 / 3.2
* ``c_g = 1.5``, ``c_L2 = 3e-5``, ``w_opp = 0.15``, ``w_ownership = 1.5/b**2``,
  ``w_score_pdf = w_score_cdf = 0.02``  -- Wu 2020, Section 2 / 4.1
* batch 256, momentum 0.9, per-sample lr 6e-5 (2e-5 for the first 5M samples),
  replay window 250k -> 22M, SWA over 4 snapshots with decay 0.75,
  gating test 100/200  -- Wu 2020, Section 2
* widening ``(6,96) -> (10,128) -> (15,192) -> (20,256)``  -- Wu 2020, Section 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class NetConfig:
    """Neural network shape."""

    board_size: int = 9
    block_count: int = 6
    channels: int = 64
    #: number of channels projected by a global-pooling layer onto the others
    pool_channels: int = 32
    #: how many of the residual blocks carry a global-pooling bias structure
    pool_blocks: int = 2
    #: whether the policy head also has a global-pooling bias structure
    policy_pool: bool = True
    in_spatial: int = 18
    in_global: int = 10
    use_batch_norm: bool = True

    @property
    def num_points(self) -> int:
        return self.board_size * self.board_size

    @property
    def num_moves(self) -> int:
        return self.num_points + 1

    @property
    def pass_index(self) -> int:
        return self.num_points

    @property
    def score_bins(self) -> int:
        """Score-belief bins cover ``[-(n+1), n+1]`` in whole points."""
        return 2 * (self.num_points + 1) + 1

    @property
    def score_offset(self) -> int:
        return self.num_points + 1

    @property
    def pool_block_indices(self) -> List[int]:
        """Indices of the residual blocks that use global pooling (evenly spaced)."""
        n, k = self.block_count, self.pool_blocks
        if k <= 0 or n <= 0:
            return []
        if k >= n:
            return list(range(n))
        # place k blocks at ~1/(k+1), 2/(k+1), ... of the trunk, like KataGo
        return [max(0, min(n - 1, round(n * (i + 1) / (k + 1)) - 1)) for i in range(k)]

    @property
    def board_avg(self) -> float:
        """``b_avg = 0.5 * (9 + 19)`` from Wu 2020, Appendix A.2."""
        return 0.5 * (9 + 19)

    @property
    def board_var(self) -> float:
        """Variance of board width over the training distribution (Appendix A.2)."""
        widths = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        avg = self.board_avg
        return sum((b - avg) ** 2 for b in widths) / len(widths)

    @staticmethod
    def demo_9x9() -> "NetConfig":
        """Small network that trains in minutes on a CPU."""
        return NetConfig(board_size=9, block_count=4, channels=32, pool_channels=16, pool_blocks=2)

    @staticmethod
    def paper_19x19() -> "NetConfig":
        """The final network size of KataGo's main run and of AlphaZero."""
        return NetConfig(board_size=19, block_count=20, channels=256, pool_channels=128, pool_blocks=3)


@dataclass
class MCTSConfig:
    """Search settings (KataGo defaults)."""

    c_puct: float = 1.1
    c_fpu: float = 0.2
    #: virtual loss applied to leaves awaiting a batched evaluation
    virtual_loss: float = 1.0
    #: Dirichlet noise at the root: (1-eps) * P + eps * Dir(alpha)
    root_noise_eps: float = 0.25
    root_noise_alpha_at_19: float = 0.03
    #: softmax temperature applied to the root priors (SAI trick)
    root_policy_temp: float = 1.03
    #: forced playouts (Wu 2020, Section 3.2)
    forced_playouts: bool = True
    forced_k: float = 2.0
    #: subtract forced playouts back out of the training target
    prune_policy_target: bool = True
    #: resign when the root value drops below this (paper: 5% false positives)
    resign_threshold: float = -0.95
    resign_min_visits: int = 100

    def noise_alpha(self, legal_count: int) -> float:
        """``alpha = 0.03 * 19**2 / N`` scaled to the current number of legal moves."""
        return self.root_noise_alpha_at_19 * 19 * 19 / max(1, legal_count)


@dataclass
class PlayoutCapConfig:
    """Playout cap randomization (Wu 2020, Section 3.1)."""

    enabled: bool = True
    #: fraction of turns that get a full search
    full_fraction: float = 0.25
    full_cap: int = 600
    fast_cap: int = 100
    #: optional annealing to a second (N, n) pair after ``anneal_after_samples``
    full_cap_late: Optional[int] = 1000
    fast_cap_late: Optional[int] = 200
    anneal_after_samples: int = 2_000_000

    def caps(self, samples_seen: int) -> Tuple[int, int]:
        if self.full_cap_late is not None and samples_seen >= self.anneal_after_samples:
            return self.full_cap_late, self.fast_cap_late or self.fast_cap
        return self.full_cap, self.fast_cap


@dataclass
class LossConfig:
    """Loss weights (Wu 2020, Section 2 and 4.1)."""

    c_game: float = 1.5
    c_l2: float = 3e-5
    w_aux_policy: float = 0.15
    w_score_pdf: float = 0.02
    w_score_cdf: float = 0.02
    #: ownership weight also depends on board size: 1.5 / b**2
    w_ownership_scale: float = 1.5

    def w_ownership(self, board_size: int) -> float:
        """Weight applied to the *mean* ownership BCE.

        Wu 2020 weights the *sum* over board points by ``1.5 / b**2``.  We
        average over the ``b**2`` points instead (so the loss is invariant to
        board size in a useful way), which cancels that denominator and leaves
        the factor 1.5.
        """
        return self.w_ownership_scale


@dataclass
class TrainConfig:
    """Self-play training loop settings."""

    batch_size: int = 256
    momentum: float = 0.9
    lr: float = 6e-5
    #: ``lr`` is the step size on the batch-mean gradient.  The paper's 6e-5 is
    #: *per sample* at batch 256, i.e. 0.0154 in this convention.
    lr_warmup_samples: int = 5_000_000
    lr_warmup: float = 2e-5
    lr_decay_at_samples: int = -1
    lr_decay: float = 6e-6
    #: replay window grows from ``window_start`` to ``window_max`` samples
    window_start: int = 250_000
    window_max: int = 22_000_000
    #: stochastic weight averaging
    swa_snapshot_samples: int = 250_000
    swa_snapshots: int = 4
    swa_decay: float = 0.75
    gating_games: int = 200
    gating_wins: int = 100
    #: Run the SWA + gating pipeline.  With ``False`` the self-play network is
    #: simply refreshed from the training weights every iteration.  The demo uses
    #: ``False`` because gating is calibrated for runs of millions of samples;
    #: see the README for the trade-off.
    gating_enabled: bool = True
    #: progressive widening schedule ``[(blocks, channels), ...]``
    widening: List[Tuple[int, int]] = field(default_factory=list)
    #: supervise the auxiliary heads
    use_aux_policy: bool = True
    use_ownership: bool = True
    use_score: bool = True
    #: number of dihedral symmetry transforms applied per sample
    symmetries: int = 8
    seed: int = 0

    def learning_rate(self, samples_seen: int) -> float:
        if samples_seen < self.lr_warmup_samples:
            return self.lr_warmup
        if self.lr_decay_at_samples >= 0 and samples_seen >= self.lr_decay_at_samples:
            return self.lr_decay
        return self.lr

    def window_size(self, total_samples: int) -> int:
        """Replay window grows with the amount of data produced so far."""
        if total_samples <= 0:
            return self.window_start
        return int(min(self.window_max, max(self.window_start, total_samples)))
