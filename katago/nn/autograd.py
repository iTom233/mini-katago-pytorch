"""A small reverse-mode autodiff engine in NumPy.

Why this exists: the project's primary network is PyTorch (:mod:`katago.nn.model`).
This module implements the handful of operations that network needs -- ``conv2d``,
``linear``, ``relu``, channel mean/max pooling, ``concat``, and the loss heads --
so the *same architecture family* can be constructed, trained and gradient-checked
on a machine with no PyTorch (and no network access to install it).

It is a reference/verification backend, not a performance backend: expect order
one to two orders of magnitude slower than PyTorch on the same shapes.

Only ``float32`` is used.  Every op implements ``forward`` and ``backward``; the
gradients are verified against finite differences in ``tests/test_autograd.py``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

DTYPE = np.float32


class Tensor:
    """A dense array plus its gradient and a closure computing that gradient."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Sequence["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad or any(c.requires_grad for c in _children)
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    # ------------------------------------------------------------------ shape
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    # ---------------------------------------------------------------- helpers
    def _accumulate(self, g: np.ndarray) -> None:
        g = np.asarray(g, dtype=DTYPE)
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad += g

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Reverse-mode sweep from this node (must be a scalar)."""
        topo: List[Tensor] = []
        seen = set()

        def build(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, DTYPE)
        for t in reversed(topo):
            if t.grad is not None:
                t._backward()


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``g`` down to ``shape`` (undo NumPy broadcasting)."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


def _ensure(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


# --------------------------------------------------------------------- elementwise


def add(a, b) -> Tensor:
    a, b = _ensure(a), _ensure(b)
    out = Tensor(a.data + b.data, _children=(a, b), _op="add")

    def _bw():
        if a.requires_grad:
            a._accumulate(_unbroadcast(out.grad, a.shape))
        if b.requires_grad:
            b._accumulate(_unbroadcast(out.grad, b.shape))

    out._backward = _bw
    return out


def mul(a, b) -> Tensor:
    a, b = _ensure(a), _ensure(b)
    out = Tensor(a.data * b.data, _children=(a, b), _op="mul")

    def _bw():
        if a.requires_grad:
            a._accumulate(_unbroadcast(out.grad * b.data, a.shape))
        if b.requires_grad:
            b._accumulate(_unbroadcast(out.grad * a.data, b.shape))

    out._backward = _bw
    return out


def scale(x: Tensor, k: float) -> Tensor:
    out = Tensor(x.data * k, _children=(x,), _op="scale")

    def _bw():
        x._accumulate(out.grad * k)

    out._backward = _bw
    return out


def relu(x: Tensor) -> Tensor:
    mask = x.data > 0
    out = Tensor(x.data * mask, _children=(x,), _op="relu")

    def _bw():
        x._accumulate(out.grad * mask)

    out._backward = _bw
    return out


def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    out = Tensor(x.data.reshape(shape), _children=(x,), _op="reshape")

    def _bw():
        x._accumulate(out.grad.reshape(x.shape))

    out._backward = _bw
    return out


def flatten(x: Tensor) -> Tensor:
    return reshape(x, (x.shape[0], int(np.prod(x.shape[1:]))))


def narrow_channels(x: Tensor, start: int, stop: int) -> Tensor:
    """Slice channels ``[start:stop]`` of a ``(B, C, H, W)`` tensor."""
    out = Tensor(x.data[:, start:stop], _children=(x,), _op="narrow")

    def _bw():
        g = np.zeros_like(x.data)
        g[:, start:stop] = out.grad
        x._accumulate(g)

    out._backward = _bw
    return out


# -------------------------------------------------------------------------- conv


def _col_indices(
    c: int, h_pad: int, w_pad: int, kh: int, kw: int, stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Index arrays mapping a ``(c, kh, kw)`` kernel over a padded image.

    Returns ``(k, i, j, out_h, out_w)`` such that
    ``cols = padded[:, k, i, j]`` has shape ``(N, c*kh*kw, out_h*out_w)``.
    """
    out_h = (h_pad - kh) // stride + 1
    out_w = (w_pad - kw) // stride + 1
    a = np.repeat(np.arange(out_h), out_w)  # output row
    b = np.tile(np.arange(out_w), out_h)  # output column
    i0 = np.tile(np.repeat(np.arange(kh), kw), c)  # kernel row offset
    j0 = np.tile(np.tile(np.arange(kw), kh), c)  # kernel col offset
    k0 = np.repeat(np.arange(c), kh * kw)
    i = i0[:, None] + a[None, :] * stride
    j = j0[:, None] + b[None, :] * stride
    k = np.broadcast_to(k0[:, None], i.shape)
    return k, i, j, out_h, out_w


def conv2d(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, pad: int = 1, stride: int = 1
) -> Tensor:
    """2-D convolution. ``x`` is ``(N, C, H, W)``, ``w`` is ``(Cout, C, kh, kw)``."""
    n, c, h, wd = x.shape
    cout, cin, kh, kw = w.shape
    assert c == cin, f"channel mismatch: input {c} vs kernel {cin}"
    h_pad, w_pad = h + 2 * pad, wd + 2 * pad
    x_pad = np.pad(x.data, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x.data
    k, i, j, out_h, out_w = _col_indices(c, h_pad, w_pad, kh, kw, stride)
    cols = x_pad[:, k, i, j]  # (N, C*kh*kw, out_h*out_w)
    w_mat = w.data.reshape(cout, -1)  # (Cout, C*kh*kw)
    out_data = np.einsum("oc,ncp->nop", w_mat, cols)
    if b is not None:
        out_data = out_data + b.data.reshape(1, -1, 1)
    children = (x, w) if b is None else (x, w, b)
    out = Tensor(out_data.reshape(n, cout, out_h, out_w), _children=children, _op="conv")

    def _bw():
        dout = out.grad.reshape(n, cout, out_h * out_w)
        if w.requires_grad:
            dw = np.einsum("nop,ncp->oc", dout, cols).reshape(w.shape)
            w._accumulate(dw)
        if x.requires_grad:
            dcols = np.einsum("oc,nop->ncp", w_mat, dout)
            dx_pad = np.zeros((n, c, h_pad, w_pad), dtype=DTYPE)
            np.add.at(dx_pad, (slice(None), k, i, j), dcols)
            dx = dx_pad[:, :, pad : pad + h, pad : pad + wd] if pad else dx_pad
            x._accumulate(dx)
        if b is not None and b.requires_grad:
            b._accumulate(out.grad.sum(axis=(0, 2, 3)))

    out._backward = _bw
    return out


# ------------------------------------------------------------------------ linear


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """``x`` is ``(B, K)``; ``w`` is ``(M, K)``; returns ``(B, M)``."""
    assert x.data.ndim == 2, f"linear expects a 2-D input, got {x.shape}"
    out_data = x.data @ w.data.T
    if b is not None:
        out_data = out_data + b.data.reshape(1, -1)
    children = (x, w) if b is None else (x, w, b)
    out = Tensor(out_data, _children=children, _op="linear")

    def _bw():
        if w.requires_grad:
            w._accumulate(out.grad.T @ x.data)
        if b is not None and b.requires_grad:
            b._accumulate(out.grad.sum(axis=0))
        if x.requires_grad:
            x._accumulate(out.grad @ w.data)

    out._backward = _bw
    return out


# ------------------------------------------------------------------------ pooling


def mean_hw(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    out = Tensor(x.data.mean(axis=(2, 3)), _children=(x,), _op="mean_hw")

    def _bw():
        g = np.broadcast_to(out.grad[:, :, None, None], x.shape) / (h * w)
        x._accumulate(g)

    out._backward = _bw
    return out


def max_hw(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    flat = x.data.reshape(b, c, h * w)
    idx = flat.argmax(axis=2)
    out_data = np.take_along_axis(flat, idx[:, :, None], axis=2)[:, :, 0]
    out = Tensor(out_data, _children=(x,), _op="max_hw")

    def _bw():
        g = np.zeros_like(flat)
        np.put_along_axis(g, idx[:, :, None], out.grad[:, :, None], axis=2)
        x._accumulate(g.reshape(x.shape))

    out._backward = _bw
    return out


def concat1(parts: Iterable[Tensor]) -> Tensor:
    """Concatenate along the channel axis (axis 1)."""
    parts = list(parts)
    out = Tensor(np.concatenate([p.data for p in parts], axis=1), _children=tuple(parts), _op="cat")
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    for p in parts:
        bounds.append((cursor, cursor + p.shape[1]))
        cursor += p.shape[1]

    def _bw():
        for p, (a, b) in zip(parts, bounds):
            if p.requires_grad:
                p._accumulate(out.grad[:, a:b])

    out._backward = _bw
    return out


# ------------------------------------------------------------------------- losses


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_cross_entropy(logits: Tensor, target: np.ndarray) -> Tensor:
    """Mean over the batch of ``-sum(target * log_softmax(logits))``."""
    z = logits.data
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    s = e.sum(axis=1, keepdims=True)
    logp = z - m - np.log(s)
    n = z.shape[0]
    loss = float(-(target * logp).sum() / n)
    out = Tensor(loss, _children=(logits,), _op="softmax_ce")

    def _bw():
        grad = (np.exp(logp) - target) / n
        logits._accumulate(grad * out.grad)

    out._backward = _bw
    return out


def sigmoid_bce(logits: Tensor, target: np.ndarray) -> Tensor:
    """Binary cross-entropy with soft targets in ``[0, 1]``; mean over elements."""
    z = logits.data
    # numerically stable log-sigmoid
    log_sig = -np.log1p(np.exp(-np.abs(z))) - np.maximum(-z, 0)
    log_one_minus = -np.log1p(np.exp(-np.abs(z))) - np.maximum(z, 0)
    n = z.size
    loss = float(-(target * log_sig + (1.0 - target) * log_one_minus).sum() / n)
    out = Tensor(loss, _children=(logits,), _op="sigmoid_bce")

    def _bw():
        p = 1.0 / (1.0 + np.exp(-z))
        logits._accumulate((p - target) / n * out.grad)

    out._backward = _bw
    return out


def softmax_cdf_loss(logits: Tensor, target_onehot: np.ndarray) -> Tensor:
    """Squared error between predicted and target cumulative distributions.

    This is the "cdf" term of KataGo's score-belief loss (Wu 2020, Section 4.1):
    it pushes probability mass to sit *near* the true score even when the exact
    bin is not hit.
    """
    p = softmax(logits.data, axis=1)
    cdf = np.cumsum(p, axis=1)
    cdf_t = np.cumsum(target_onehot, axis=1)
    diff = cdf - cdf_t
    n = logits.shape[0]
    loss = float((diff ** 2).sum() / n)
    out = Tensor(loss, _children=(logits,), _op="cdf")

    def _bw():
        # d/dp of sum_k (sum_{j<=k} p_j - t_k)^2  =  2 * reverse_cumsum(diff)
        d = 2.0 * diff / n
        rev = np.cumsum(d[:, ::-1], axis=1)[:, ::-1]
        dp = rev
        grad = p * (dp - (p * dp).sum(axis=1, keepdims=True))
        logits._accumulate(grad * out.grad)

    out._backward = _bw
    return out


# --------------------------------------------------------------------- optimiser


class SGD:
    """Stochastic gradient descent with momentum, matching the paper's setting."""

    def __init__(self, params: Sequence[Tensor], lr: float = 6e-5, momentum: float = 0.9,
                 weight_decay: float = 0.0) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.velocity[i] = self.momentum * self.velocity[i] + g
            p.data -= self.lr * self.velocity[i]
