"""KataGo-style Go engine in PyTorch.

Layout
------
``katago.game``        rules, board, scoring, input features
``katago.nn``          neural network (PyTorch primary, NumPy reference backend)
``katago.mcts``        KataGo-flavoured Monte-Carlo tree search
``katago.selfplay``    self-play data generation, replay, training loop
``katago.inference``   playing engine and GTP front-end
``katago.utils``       symmetry transforms, SGF, misc helpers
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
