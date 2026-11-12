"""Numerical gradient checks for the NumPy autodiff engine."""

from __future__ import annotations

import numpy as np

from katago.nn import autograd as ag

RNG = np.random.default_rng(1234)


def _check(tensors, build_loss, eps=1e-3, tol=2e-2, note=""):
    """Compare analytic gradients against central finite differences."""
    loss = build_loss()
    for t in tensors:
        t.grad = None
    loss.backward()
    for idx, t in enumerate(tensors):
        analytic = t.grad.copy()
        numeric = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            mi = it.multi_index
            orig = t.data[mi]
            t.data[mi] = orig + eps
            hi = build_loss().data
            t.data[mi] = orig - eps
            lo = build_loss().data
            t.data[mi] = orig
            numeric[mi] = (hi - lo) / (2 * eps)
        denom = max(1e-6, np.abs(analytic).max(), np.abs(numeric).max())
        err = np.abs(analytic - numeric).max() / denom
        assert err < tol, f"{note} tensor {idx}: relative grad error {err:.4f}"


def test_grad_add_broadcast():
    a = ag.Tensor(RNG.normal(size=(2, 3)), requires_grad=True)
    b = ag.Tensor(RNG.normal(size=(1, 3)), requires_grad=True)
    _check([a, b], lambda: ag.add(a, b).data.sum() * ag.Tensor(1.0) if False else _sum(ag.add(a, b)), note="add")


def _sum(t: ag.Tensor):
    """Sum a tensor into a fresh scalar Tensor with a backward path."""
    out = ag.Tensor(t.data.sum(), _children=(t,), _op="sum")

    def _bw():
        t._accumulate(np.ones_like(t.data) * out.grad)

    out._backward = _bw
    return out


def test_grad_sum_then_mul():
    a = ag.Tensor(RNG.normal(size=(2, 3)), requires_grad=True)
    k = ag.Tensor(np.float32(2.5), requires_grad=True)
    _check([a, k], lambda: ag.mul(_sum(a), k), note="mul")


def test_grad_relu():
    a = ag.Tensor(RNG.normal(size=(4, 5)), requires_grad=True)
    _check([a], lambda: _sum(ag.relu(a)), note="relu")


def test_grad_linear():
    x = ag.Tensor(RNG.normal(size=(3, 6)).astype(np.float32), requires_grad=True)
    w = ag.Tensor(RNG.normal(size=(4, 6)).astype(np.float32), requires_grad=True)
    b = ag.Tensor(RNG.normal(size=(4,)).astype(np.float32), requires_grad=True)
    _check([x, w, b], lambda: _sum(ag.linear(x, w, b)), note="linear")


def test_grad_conv2d_pad1():
    x = ag.Tensor(RNG.normal(size=(2, 3, 5, 5)).astype(np.float32), requires_grad=True)
    w = ag.Tensor(RNG.normal(size=(4, 3, 3, 3)).astype(np.float32) * 0.5, requires_grad=True)
    b = ag.Tensor(RNG.normal(size=(4,)).astype(np.float32), requires_grad=True)
    _check([x, w, b], lambda: _sum(ag.conv2d(x, w, b, pad=1)), note="conv2d")


def test_grad_conv2d_pad0_5x5():
    x = ag.Tensor(RNG.normal(size=(1, 2, 6, 6)).astype(np.float32), requires_grad=True)
    w = ag.Tensor(RNG.normal(size=(3, 2, 5, 5)).astype(np.float32) * 0.4, requires_grad=True)
    _check([x, w], lambda: _sum(ag.conv2d(x, w, None, pad=0)), note="conv5x5")


def test_grad_mean_hw_and_max_hw():
    x = ag.Tensor(RNG.normal(size=(2, 4, 3, 3)).astype(np.float32), requires_grad=True)
    _check([x], lambda: _sum(ag.mean_hw(x)), note="mean_hw")
    x2 = ag.Tensor(RNG.normal(size=(2, 4, 3, 3)).astype(np.float32), requires_grad=True)
    _check([x2], lambda: _sum(ag.max_hw(x2)), note="max_hw", tol=5e-2)


def test_grad_concat_and_narrow():
    a = ag.Tensor(RNG.normal(size=(2, 3, 2, 2)).astype(np.float32), requires_grad=True)
    b = ag.Tensor(RNG.normal(size=(2, 5, 2, 2)).astype(np.float32), requires_grad=True)
    _check([a, b], lambda: _sum(ag.concat1([a, b])), note="concat")
    c = ag.Tensor(RNG.normal(size=(2, 6, 2, 2)).astype(np.float32), requires_grad=True)
    _check([c], lambda: _sum(ag.narrow_channels(c, 1, 4)), note="narrow")


def test_grad_softmax_ce():
    z = ag.Tensor(RNG.normal(size=(4, 7)).astype(np.float32), requires_grad=True)
    target = np.exp(RNG.normal(size=(4, 7)).astype(np.float32))
    target /= target.sum(axis=1, keepdims=True)
    _check([z], lambda: ag.softmax_cross_entropy(z, target), note="softmax_ce")


def test_grad_sigmoid_bce():
    z = ag.Tensor(RNG.normal(size=(4, 5)).astype(np.float32), requires_grad=True)
    target = (RNG.random((4, 5)) > 0.5).astype(np.float32)
    _check([z], lambda: ag.sigmoid_bce(z, target), note="sigmoid_bce")


def test_grad_cdf_loss():
    z = ag.Tensor((RNG.normal(size=(3, 9)) * 2).astype(np.float32), requires_grad=True)
    tgt = np.zeros((3, 9), dtype=np.float32)
    tgt[0, 2] = 1.0
    tgt[1, 5] = 1.0
    tgt[2, 8] = 1.0
    _check([z], lambda: ag.softmax_cdf_loss(z, tgt), note="cdf")


def test_softmax_cdf_loss_prefers_nearby_bin():
    """The cdf term should punish a distant bin far more than an adjacent one."""
    z_near = np.zeros((1, 20), dtype=np.float32)
    z_near[0, 11] = 4.0  # near the true bin 10
    z_far = np.zeros((1, 20), dtype=np.float32)
    z_far[0, 2] = 4.0  # far away
    tgt = np.zeros((1, 20), dtype=np.float32)
    tgt[0, 10] = 1.0
    near = ag.softmax_cdf_loss(ag.Tensor(z_near), tgt).data
    far = ag.softmax_cdf_loss(ag.Tensor(z_far), tgt).data
    assert near < far, f"cdf loss should be smaller for a near miss ({near} vs {far})"


def test_sgd_reduces_simple_quadratic():
    """SGD with momentum must reduce a simple convex objective."""
    w = ag.Tensor(np.array([[3.0, -2.0]], dtype=np.float32), requires_grad=True)
    target = np.array([[1.0, 0.0]], dtype=np.float32)
    opt = ag.SGD([w], lr=0.1, momentum=0.9)

    def loss_value() -> float:
        return float(ag.softmax_cross_entropy(w, target).data)

    first = loss_value()
    for _ in range(50):
        opt.zero_grad()
        loss = ag.softmax_cross_entropy(w, target)
        loss.backward()
        opt.step()
    after = loss_value()
    assert after < first, f"SGD did not decrease the objective: {first:.4f} -> {after:.4f}"
    # the objective wants w[:, 0] > w[:, 1]
    assert w.data[0, 0] > w.data[0, 1]
