"""Minimal test runner: ``python tests/run_tests.py [pattern ...]``.

Prefers pytest when it is installed, otherwise discovers ``test_*`` functions in
``tests/test_*.py`` and runs them directly.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)


def discover(patterns):
    modules = []
    for name in sorted(os.listdir(HERE)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        stem = name[:-3]
        if patterns and not any(p in stem for p in patterns):
            continue
        modules.append(importlib.import_module(stem))
    return modules


def main(argv):
    patterns = argv[1:]
    modules = discover(patterns)
    if not modules:
        print("no test modules found")
        return 1
    total = passed = failed = skipped = 0
    failures = []
    start = time.time()
    for module in modules:
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            total += 1
            label = f"{module.__name__}::{attr}"
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                if type(exc).__name__ == "SkipTest":
                    skipped += 1
                    print(f"SKIP {label}: {exc}")
                    continue
                failed += 1
                failures.append((label, exc, traceback.format_exc()))
                print(f"FAIL {label}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
    elapsed = time.time() - start
    print()
    for label, exc, tb in failures:
        print(f"===== {label} =====")
        print(tb)
    print(f"{passed}/{total} passed, {failed} failed, {skipped} skipped in {elapsed:.2f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
