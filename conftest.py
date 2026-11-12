"""Allow ``pytest`` to run from the project root without installing the package."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(ROOT, "tests")
for path in (ROOT, TESTS):
    if path not in sys.path:
        sys.path.insert(0, path)
