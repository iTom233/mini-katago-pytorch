"""Tiny assertion helpers so the suite runs with or without pytest installed."""

from __future__ import annotations

import contextlib
import math


class SkipTest(Exception):
    """Raised by a test that cannot run in this environment (e.g. no PyTorch)."""


class _Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.value = None
        self._ctx = None

    def __enter__(self):
        self._ctx = contextlib.suppress()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type.__name__}, nothing raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc
        return True


def raises(exc_type):
    """Stand-in for ``pytest.raises`` used as a context manager."""
    return _Raises(exc_type)


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other) -> bool:
        if isinstance(self.expected, (list, tuple)):
            return len(other) == len(self.expected) and all(
                math.isclose(float(a), float(b), rel_tol=self.rel, abs_tol=self.abs)
                for a, b in zip(other, self.expected)
            )
        return math.isclose(
            float(other), float(self.expected), rel_tol=self.rel, abs_tol=self.abs
        )

    def __repr__(self) -> str:
        return f"approx({self.expected!r})"


def approx(expected, rel=1e-6, abs=1e-12):
    """Stand-in for ``pytest.approx``."""
    return _Approx(expected, rel=rel, abs=abs)
