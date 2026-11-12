"""NumPy reference implementation of the KataGo network.

Mirrors the PyTorch model's topology:

    trunk:  5x5 conv  +  linear(global vector) as a per-channel bias
            -> n pre-activation residual blocks, of which ``pool_blocks`` use the
               Global Pooling Bias structure of Wu 2020 Section 3.3 / Figure 2
    heads:  policy  (1x1 conv -> global-pooling bias -> FC to S*S+1)
            value   (1x1 conv -> value-mode pooling -> FC)
            ownership / score-belief / auxiliary-policy heads off the value tower

It deliberately omits BatchNorm: the point of this backend is to *verify* the
pipeline end to end on a machine without PyTorch, and every op here has a
finite-difference gradient check in ``tests/test_autograd.py``.  The PyTorch model
in :mod:`katago.nn.model` is the one that follows the paper exactly (BN included).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import autograd as ag
from .autograd import Tensor
from .base import BaseNet, EvalResult, masked_softmax, softmax_rows
from .config import LossConfig, NetConfig


def _he(rng, shape, fan_in):
    return (rng.normal(0.0, np.sqrt(2.0 / fan_in), shape)).astype(np.float32)


class NumpyNet(BaseNet):
    """Same architecture family as the PyTorch model, minus BatchNorm."""

    def __init__(self, config: NetConfig, seed: int = 0, momentum: float = 0.9,
                 weight_decay: float = 0.0, max_grad_norm: float = 50.0) -> None:
        super().__init__(config)
        self.rng = np.random.default_rng(seed)
        self.params: Dict[str, Tensor] = {}
        self._build()
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self._velocity = {k: np.zeros_like(v.data) for k, v in self.params.items()}

    # ------------------------------------------------------------- building
    def _p(self, name: str, shape, fan_in: Optional[int] = None, zero: bool = False,
           init_scale: float = 1.0) -> Tensor:
        if zero:
            data = np.zeros(shape, dtype=np.float32)
        else:
            data = _he(self.rng, shape, fan_in or shape[-1]) * init_scale
        t = Tensor(data, requires_grad=True)
        self.params[name] = t
        return t

    def _conv(self, name: str, cin: int, cout: int, k: int, init_scale: float = 1.0) -> None:
        self._p(name + "_w", (cout, cin, k, k), cin * k * k, init_scale=init_scale)
        self._p(name + "_b", (cout,), zero=True)

    def _lin(self, name: str, cin: int, cout: int, zero_bias: bool = True,
             init_scale: float = 1.0) -> None:
        self._p(name + "_w", (cout, cin), cin, init_scale=init_scale)
        self._p(name + "_b", (cout,), zero=zero_bias)

    #: Output heads are initialised with a small gain so the network starts with
    #: a near-uniform policy and a value of ~0.5.  Without this the random head
    #: is arbitrarily confident, self-play becomes deterministic and the search
    #: never produces a usefully spread policy target.
    HEAD_INIT_SCALE = 0.01

    def _build(self) -> None:
        cfg = self.config
        c = cfg.channels
        s = self.HEAD_INIT_SCALE
        self._conv("c0", cfg.in_spatial, c, 5)
        self._lin("g0", cfg.in_global, c)
        for i in range(cfg.block_count):
            self._conv(f"b{i}_c1", c, c, 3)
            self._conv(f"b{i}_c2", c, c, 3)
        for i in cfg.pool_block_indices:
            self._lin(f"b{i}_gp", 3 * cfg.pool_channels, c - cfg.pool_channels)
        # policy tower
        self._conv("ph", c, c, 1)
        if cfg.policy_pool:
            self._lin("ph_gp", 3 * cfg.pool_channels, c - cfg.pool_channels)
        self._lin("p", c * cfg.num_points, cfg.num_moves, init_scale=s)
        # value tower
        self._conv("vh", c, c, 1)
        self._lin("vh_gp", 3 * cfg.pool_channels, c - cfg.pool_channels)
        self._lin("v1", c * cfg.num_points, c)
        self._lin("v_win", c, 2, init_scale=s)
        self._lin("v_own", c, cfg.num_points, init_scale=s)
        self._lin("v_score", c, cfg.score_bins, init_scale=s)
        self._lin("v_aux", c, cfg.num_moves, init_scale=s)

    # ------------------------------------------------------------ components
    def _pool_features(self, g: Tensor, value_mode: bool) -> Tensor:
        """Global pooling layer: mean, board-scaled mean, then max or var-scaled mean."""
        cfg = self.config
        m = ag.mean_hw(g)  # (B, C)
        b = float(cfg.board_size)
        s1 = (b - cfg.board_avg) / 10.0
        part1 = m
        part2 = ag.scale(m, s1)
        if value_mode:
            s2 = ((b - cfg.board_avg) ** 2 - cfg.board_var) / 100.0
            part3 = ag.scale(m, s2)
        else:
            part3 = ag.max_hw(g)
        return ag.concat1([part1, part2, part3])

    def _gpb(self, g: Tensor, name: str, n_out: int, value_mode: bool) -> Tensor:
        """Global pooling bias vector of shape ``(B, n_out)``."""
        feats = self._pool_features(g, value_mode)
        return ag.linear(feats, self.params[name + "_w"], self.params[name + "_b"])

    def _add_channel_bias(self, x: Tensor, bias: Tensor, prefix_channels: int) -> Tensor:
        """Add ``bias`` (B, C-prefix) to channels ``[prefix:]`` of ``x``."""
        bsz = x.shape[0]
        zeros = Tensor(np.zeros((bsz, prefix_channels), dtype=np.float32))
        delta = ag.concat1([zeros, bias])  # (B, C)
        return ag.add(x, ag.reshape(delta, (bsz, x.shape[1], 1, 1)))

    def _res_block(self, t: Tensor, i: int) -> Tensor:
        p = self.params
        y = ag.relu(t)
        y = ag.conv2d(y, p[f"b{i}_c1_w"], p[f"b{i}_c1_b"], pad=1)
        y = ag.relu(y)
        y = ag.conv2d(y, p[f"b{i}_c2_w"], p[f"b{i}_c2_b"], pad=1)
        return ag.add(t, y)

    def _pool_block(self, t: Tensor, i: int) -> Tensor:
        cfg = self.config
        p = self.params
        pc = cfg.pool_channels
        y = ag.relu(t)
        y = ag.conv2d(y, p[f"b{i}_c1_w"], p[f"b{i}_c1_b"], pad=1)
        g = ag.narrow_channels(y, 0, pc)
        delta = self._gpb(g, f"b{i}_gp", cfg.channels - pc, value_mode=False)
        y = self._add_channel_bias(y, delta, pc)
        y = ag.relu(y)
        y = ag.conv2d(y, p[f"b{i}_c2_w"], p[f"b{i}_c2_b"], pad=1)
        return ag.add(t, y)

    # -------------------------------------------------------------- forward
    def _forward(self, spatial: Tensor, glob: Tensor) -> Dict[str, Tensor]:
        cfg = self.config
        p = self.params
        bsz = spatial.shape[0]
        t = ag.conv2d(spatial, p["c0_w"], p["c0_b"], pad=2)
        gb = ag.linear(glob, p["g0_w"], p["g0_b"])
        t = ag.add(t, ag.reshape(gb, (bsz, cfg.channels, 1, 1)))

        pool_idx = set(cfg.pool_block_indices)
        for i in range(cfg.block_count):
            t = self._pool_block(t, i) if i in pool_idx else self._res_block(t, i)

        # policy tower
        y = ag.relu(ag.conv2d(t, p["ph_w"], p["ph_b"], pad=0))
        if cfg.policy_pool:
            g = ag.narrow_channels(y, 0, cfg.pool_channels)
            delta = self._gpb(g, "ph_gp", cfg.channels - cfg.pool_channels, value_mode=False)
            y = self._add_channel_bias(y, delta, cfg.pool_channels)
        pol = ag.linear(ag.flatten(y), p["p_w"], p["p_b"])

        # value tower
        v = ag.relu(ag.conv2d(t, p["vh_w"], p["vh_b"], pad=0))
        v = self._add_channel_bias(
            v, self._gpb(ag.narrow_channels(v, 0, cfg.pool_channels), "vh_gp",
                         cfg.channels - cfg.pool_channels, value_mode=True),
            cfg.pool_channels,
        )
        h = ag.relu(ag.linear(ag.flatten(v), p["v1_w"], p["v1_b"]))
        return {
            "policy": pol,
            "value": ag.linear(h, p["v_win_w"], p["v_win_b"]),
            "ownership": ag.linear(h, p["v_own_w"], p["v_own_b"]),
            "score": ag.linear(h, p["v_score_w"], p["v_score_b"]),
            "aux_policy": ag.linear(h, p["v_aux_w"], p["v_aux_b"]),
        }

    # ------------------------------------------------------------ inference
    def predict(self, spatial, global_, legal=None) -> EvalResult:
        cfg = self.config
        out = self._forward(Tensor(spatial), Tensor(global_))
        pol = masked_softmax(out["policy"].data, legal)
        val = softmax_rows(out["value"].data)
        own = 1.0 / (1.0 + np.exp(-out["ownership"].data))
        score = softmax_rows(out["score"].data)
        aux = masked_softmax(out["aux_policy"].data, legal)
        return EvalResult(
            policy=pol,
            value=val,
            ownership=own.astype(np.float32),
            score=score,
            aux_policy=aux,
        )

    # ------------------------------------------------------------- training
    def train_batch(self, batch: Dict[str, np.ndarray], loss_cfg: LossConfig, lr: float) -> Dict[str, float]:
        cfg = self.config
        spatial = Tensor(batch["spatial"])
        glob = Tensor(batch["global_"])
        out = self._forward(spatial, glob)

        legal = batch.get("legal")
        policy_logits = out["policy"]
        if legal is not None:
            policy_logits = _mask_logits(policy_logits, legal)

        pi = batch["policy"]
        loss_policy = ag.softmax_cross_entropy(policy_logits, pi)
        total = loss_policy

        metrics = {"policy": float(loss_policy.data)}

        if "aux" in batch and batch["aux"] is not None:
            aux_logits = out["aux_policy"]
            if legal is not None:
                aux_logits = _mask_logits(aux_logits, legal)
            loss_aux = ag.softmax_cross_entropy(aux_logits, batch["aux"])
            total = ag.add(total, ag.scale(loss_aux, loss_cfg.w_aux_policy))
            metrics["aux_policy"] = float(loss_aux.data)

        z = batch["value"]
        loss_value = ag.softmax_cross_entropy(out["value"], z)
        total = ag.add(total, ag.scale(loss_value, loss_cfg.c_game))
        metrics["value"] = float(loss_value.data)

        if "ownership" in batch and batch["ownership"] is not None:
            w = loss_cfg.w_ownership(cfg.board_size)
            loss_own = ag.sigmoid_bce(out["ownership"], batch["ownership"])
            total = ag.add(total, ag.scale(loss_own, w))
            metrics["ownership"] = float(loss_own.data)

        if "score" in batch and batch["score"] is not None:
            n = out["score"].shape[0]
            bins = np.zeros((n, cfg.score_bins), dtype=np.float32)
            rows = np.arange(n)
            bins[rows, batch["score"]] = 1.0
            loss_pdf = ag.softmax_cross_entropy(out["score"], bins)
            loss_cdf = ag.softmax_cdf_loss(out["score"], bins)
            total = ag.add(total, ag.scale(loss_pdf, loss_cfg.w_score_pdf))
            total = ag.add(total, ag.scale(loss_cdf, loss_cfg.w_score_cdf))
            metrics["score_pdf"] = float(loss_pdf.data)
            metrics["score_cdf"] = float(loss_cdf.data)

        metrics["loss"] = float(total.data)

        for p in self.params.values():
            p.grad = None
        total.backward()

        # ``lr`` is the step size applied to the *batch-mean* gradient, the same
        # convention as the PyTorch backend.  Wu 2020 Section 2 quotes a
        # per-sample lr of 6e-5 at batch 256, i.e. 0.0154 in this convention.  A
        # global-norm clip keeps the CPU demo stable at learning rates that are
        # large by paper standards.
        grads = {name: p.grad for name, p in self.params.items() if p.grad is not None}
        if self.max_grad_norm:
            norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
            if norm > self.max_grad_norm:
                k = self.max_grad_norm / (norm + 1e-12)
                for g in grads.values():
                    g *= k
        for name, g in grads.items():
            p = self.params[name]
            if loss_cfg.c_l2:
                g = g + loss_cfg.c_l2 * p.data
            self._velocity[name] = self.momentum * self._velocity[name] + g
            p.data -= lr * self._velocity[name]
        return metrics

    # ---------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.data.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None:
        missing = [k for k in self.params if k not in state]
        unexpected = [k for k in state if k not in self.params]
        mismatched = [
            k
            for k, v in state.items()
            if k in self.params and tuple(self.params[k].data.shape) != tuple(np.shape(v))
        ]
        if strict and (missing or mismatched):
            raise ValueError(
                "checkpoint does not match this network: "
                f"missing={missing[:3]} "
                f"shape-mismatch={[(k, self.params[k].data.shape, np.shape(state[k])) for k in mismatched[:3]]} "
                f"unexpected={unexpected[:3]}"
            )
        for k, v in state.items():
            if k in self.params and self.params[k].data.shape == np.shape(v):
                self.params[k].data[...] = v
        self._velocity = {k: np.zeros_like(v.data) for k, v in self.params.items()}

    def num_parameters(self) -> int:
        return int(sum(v.data.size for v in self.params.values()))


def _mask_logits(t: Tensor, legal: np.ndarray) -> Tensor:
    masked = np.where(legal, t.data, -1e30).astype(np.float32)
    out = Tensor(masked, _children=(t,), _op="mask")

    def _bw():
        t._accumulate(out.grad * legal.astype(np.float32))

    out._backward = _bw
    return out
