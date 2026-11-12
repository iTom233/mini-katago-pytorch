"""PyTorch implementation of the KataGo network.

Follows Wu 2020, Appendix A:

* **Trunk.**  A 5x5 convolution of the 18 binary spatial planes, plus a fully
  connected layer on the 10 global features whose output is added channelwise as
  a bias.  Then ``n`` pre-activation residual blocks (BN -> ReLU -> 3x3 conv ->
  BN -> ReLU -> 3x3 conv -> skip).  Two or three of those blocks, spaced evenly
  through the trunk, instead use a *global-pooling bias structure* (Section 3.3,
  Figure 2).
* **Global pooling layer** (A.2).  Given ``c`` channels it produces ``3c`` values:
  the channel mean, the mean scaled by ``(b - b_avg)/10``, and the channel max.
  The last term is replaced in the value head by a quadratically scaled mean,
  because the value head predicts quantities such as the score difference, which
  scales with the *square* of the board width.
* **Global-pooling bias structure** (Figure 2).  ``G`` is batch-normalised and
  ReLU'd, globally pooled to ``3 * c_G`` values, projected back to ``c_X`` by a
  fully connected layer, and added channelwise to ``X``.
* **Heads.**  A policy head (1x1 conv -> BN -> ReLU -> global-pooling bias -> FC
  to ``b*b + 1``) and a value tower that feeds the win/loss head plus the
  auxiliary ownership, score-belief and opponent-policy heads of Section 4.1 and
  3.4.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet, EvalResult, masked_softmax
from .config import LossConfig, NetConfig

#: ``b_avg = 0.5 * (9 + 19)`` and the matching variance, Wu 2020 Appendix A.2.
B_AVG = 0.5 * (9 + 19)
B_VAR = sum((b - B_AVG) ** 2 for b in range(9, 20)) / 11.0


class GlobalPoolingLayer(nn.Module):
    """Maps ``(B, C, H, W)`` to ``(B, 3C)`` of global statistics."""

    def __init__(self, value_mode: bool = False) -> None:
        super().__init__()
        self.value_mode = value_mode

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        b = float(g.shape[-1])
        mean = g.mean(dim=(2, 3))
        scale1 = (b - B_AVG) / 10.0
        parts = [mean, mean * scale1]
        if self.value_mode:
            scale2 = ((b - B_AVG) ** 2 - B_VAR) / 100.0
            parts.append(mean * scale2)
        else:
            parts.append(g.amax(dim=(2, 3)))
        return torch.cat(parts, dim=1)


class GlobalPoolingBias(nn.Module):
    """Bias a tensor ``X`` using globally pooled statistics of another tensor ``G``."""

    def __init__(self, c_g: int, c_x: int, value_mode: bool = False) -> None:
        super().__init__()
        self.c_g = c_g
        self.c_x = c_x
        self.bn = nn.BatchNorm2d(c_g)
        self.pool = GlobalPoolingLayer(value_mode)
        self.fc = nn.Linear(3 * c_g, c_x)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g = F.relu(self.bn(g))
        pooled = self.pool(g)
        bias = self.fc(pooled)
        return x + bias[:, :, None, None]


class ResBlock(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.bn1(x)))
        y = self.conv2(F.relu(self.bn2(y)))
        return x + y


class GlobalPoolingBlock(nn.Module):
    """Residual block whose middle layer conditions the channels on global context."""

    def __init__(self, channels: int, pool_channels: int) -> None:
        super().__init__()
        self.pool_channels = pool_channels
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gpb = GlobalPoolingBias(pool_channels, channels - pool_channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.bn1(x)))
        pooled = y[:, : self.pool_channels]
        rest = y[:, self.pool_channels :]
        rest = self.gpb(rest, pooled)
        y = torch.cat([pooled, rest], dim=1)
        y = self.conv2(F.relu(self.bn2(y)))
        return x + y


class KataGoNet(nn.Module):
    """The network itself: trunk plus all output heads."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        c = config.channels
        pc = config.pool_channels
        pool_idx = set(config.pool_block_indices)

        self.conv0 = nn.Conv2d(config.in_spatial, c, 5, padding=2, bias=False)
        self.g0 = nn.Linear(config.in_global, c)

        blocks = []
        for i in range(config.block_count):
            if i in pool_idx:
                blocks.append(GlobalPoolingBlock(c, pc))
            else:
                blocks.append(ResBlock(c))
        self.blocks = nn.ModuleList(blocks)

        # --- policy head
        self.policy_conv = nn.Conv2d(c, c, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(c)
        self.policy_gpb = (
            GlobalPoolingBias(pc, c - pc) if config.policy_pool else None
        )
        self.policy_fc = nn.Linear(c * config.num_points, config.num_moves)

        # --- value tower
        self.value_conv = nn.Conv2d(c, c, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(c)
        self.value_gpb = GlobalPoolingBias(pc, c - pc, value_mode=True)
        self.value_fc = nn.Linear(c * config.num_points, c)
        self.head_win = nn.Linear(c, 2)
        self.head_ownership = nn.Linear(c, config.num_points)
        self.head_score = nn.Linear(c, config.score_bins)
        self.head_aux = nn.Linear(c, config.num_moves)
        self._init_heads()

    HEAD_INIT_SCALE = 0.01

    def _init_heads(self) -> None:
        """Small gain on the output layers.

        A randomly initialised head is arbitrarily confident, which makes the
        very first self-play games deterministic; KataGo and AlphaZero both start
        from a near-uniform policy.  Scaling the final layers down gives the
        search something to improve on.
        """
        for layer in (
            self.policy_fc,
            self.head_win,
            self.head_ownership,
            self.head_score,
            self.head_aux,
        ):
            nn.init.normal_(layer.weight, mean=0.0, std=self.HEAD_INIT_SCALE)
            nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------ trunk
    def trunk(self, spatial: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        x = self.conv0(spatial)
        bias = self.g0(glob)
        x = x + bias[:, :, None, None]
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, spatial: torch.Tensor, glob: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = self.trunk(spatial, glob)

        p = F.relu(self.policy_bn(self.policy_conv(t)))
        if self.policy_gpb is not None:
            pc = self.config.pool_channels
            p = torch.cat([p[:, :pc], self.policy_gpb(p[:, pc:], p[:, :pc])], dim=1)
        policy = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(t)))
        # the global statistics of the first ``pool_channels`` bias the rest,
        # exactly as in the residual blocks
        pc = self.config.pool_channels
        v_pooled = v[:, :pc]
        v_rest = self.value_gpb(v[:, pc:], v_pooled)
        v = torch.cat([v_pooled, v_rest], dim=1)
        h = F.relu(self.value_fc(v.flatten(1)))

        return {
            "policy": policy,
            "value": self.head_win(h),
            "ownership": self.head_ownership(h),
            "score": self.head_score(h),
            "aux_policy": self.head_aux(h),
        }


class TorchNet(BaseNet):
    """:class:`BaseNet` adapter around :class:`KataGoNet`."""

    def __init__(
        self,
        config: NetConfig,
        device: str = "cpu",
        seed: int = 0,
        momentum: float = 0.9,
        max_grad_norm: float = 50.0,
    ) -> None:
        super().__init__(config)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.model = KataGoNet(config).to(self.device)
        self.momentum = momentum
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1.0, momentum=momentum
        )
        self._lr = 1.0

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, spatial, global_, legal=None) -> EvalResult:
        self.model.eval()
        sp = torch.from_numpy(np.ascontiguousarray(spatial)).float().to(self.device)
        gl = torch.from_numpy(np.ascontiguousarray(global_)).float().to(self.device)
        out = self.model(sp, gl)
        # Apply the activations on the device and transfer every head back in a
        # single host copy.  Each separate ``.cpu()`` forces a stream sync, and
        # MCTS calls predict() once per leaf batch, so the 5 syncs were costing
        # more than the forward pass itself on small networks.
        probs = [
            F.softmax(out["policy"], dim=1),
            F.softmax(out["value"], dim=1),
            torch.sigmoid(out["ownership"]),
            F.softmax(out["score"], dim=1),
            F.softmax(out["aux_policy"], dim=1),
        ]
        widths = [p.shape[1] for p in probs]
        packed = torch.cat(probs, dim=1).cpu().numpy()
        pol, val, own, score, aux = np.split(packed, np.cumsum(widths)[:-1], axis=1)
        if legal is not None:
            pol = masked_softmax(np.log(np.maximum(pol, 1e-30)), legal)
            aux = masked_softmax(np.log(np.maximum(aux, 1e-30)), legal)
        return EvalResult(
            policy=pol.astype(np.float32, copy=False),
            value=val.astype(np.float32, copy=False),
            ownership=own.astype(np.float32, copy=False),
            score=score.astype(np.float32, copy=False),
            aux_policy=aux.astype(np.float32, copy=False),
        )

    # -------------------------------------------------------------- training
    def _tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=self.device)

    def _masked_log_softmax(self, logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(~legal, -1e30)
        return F.log_softmax(logits, dim=1)

    def train_batch(self, batch: Dict[str, np.ndarray], loss_cfg: LossConfig, lr: float) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        spatial = self._tensor(batch["spatial"])
        glob = self._tensor(batch["global_"])
        legal = self._tensor(batch["legal"], torch.bool) if "legal" in batch else None
        out = self.model(spatial, glob)

        pi = self._tensor(batch["policy"])
        if legal is not None:
            logp = self._masked_log_softmax(out["policy"], legal)
        else:
            logp = F.log_softmax(out["policy"], dim=1)
        loss_policy = -(pi * logp).sum(dim=1).mean()

        z = self._tensor(batch["value"])
        loss_value = -(z * F.log_softmax(out["value"], dim=1)).sum(dim=1).mean()

        total = loss_policy + loss_cfg.c_game * loss_value
        metrics = {"policy": float(loss_policy.detach()), "value": float(loss_value.detach())}

        if batch.get("aux") is not None:
            aux_t = self._tensor(batch["aux"])
            logp_aux = (
                self._masked_log_softmax(out["aux_policy"], legal)
                if legal is not None
                else F.log_softmax(out["aux_policy"], dim=1)
            )
            loss_aux = -(aux_t * logp_aux).sum(dim=1).mean()
            total = total + loss_cfg.w_aux_policy * loss_aux
            metrics["aux_policy"] = float(loss_aux.detach())

        if batch.get("ownership") is not None:
            own_t = self._tensor(batch["ownership"]).reshape(-1)
            own_logits = out["ownership"].reshape(-1)
            loss_own = F.binary_cross_entropy_with_logits(own_logits, own_t)
            total = total + loss_cfg.w_ownership(cfg.board_size) * loss_own
            metrics["ownership"] = float(loss_own.detach())
        if batch.get("score") is not None:
            score_t = self._tensor(batch["score"], torch.long)
            logp_score = F.log_softmax(out["score"], dim=1)
            loss_pdf = -logp_score.gather(1, score_t[:, None]).mean()
            cdf_pred = torch.cumsum(torch.exp(logp_score), dim=1)
            onehot = F.one_hot(score_t, cfg.score_bins).float()
            cdf_true = torch.cumsum(onehot, dim=1)
            loss_cdf = ((cdf_pred - cdf_true) ** 2).sum(dim=1).mean()
            total = total + loss_cfg.w_score_pdf * loss_pdf + loss_cfg.w_score_cdf * loss_cdf
            metrics["score_pdf"] = float(loss_pdf.detach())
            metrics["score_cdf"] = float(loss_cdf.detach())

        metrics["loss"] = float(total.detach())

        self.optimizer.zero_grad(set_to_none=True)
        # ``lr`` is the step size applied to the *batch-mean* gradient, the same
        # convention as the NumPy reference backend, so one set of command-line
        # values trains both backends identically.  Wu 2020 Section 2 quotes a
        # per-sample lr of 6e-5 at batch 256; that is a step size of
        # 6e-5 * 256 = 0.0154 in this convention.
        total.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        return metrics

    # ---------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in self.model.state_dict().items()
        }

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None:
        current = self.model.state_dict()
        payload = {}
        for k, v in state.items():
            if k in current and current[k].shape == tuple(np.shape(v)):
                payload[k] = torch.as_tensor(v, dtype=current[k].dtype)
        if strict:
            loaded = {k for k in payload if not k.endswith("num_batches_tracked")}
            expected = {k for k in current if not k.endswith("num_batches_tracked")}
            if loaded != expected:
                raise ValueError(
                    "checkpoint does not match this network: "
                    f"missing={sorted(expected - loaded)[:3]} "
                    f"unexpected={sorted(loaded - expected)[:3]}"
                )
        self.model.load_state_dict(payload, strict=False)

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def train(self, mode: bool = True) -> "TorchNet":
        self.model.train(mode)
        return self

    def eval(self) -> "TorchNet":
        self.model.eval()
        return self
