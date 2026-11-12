"""A small GTP (Go Text Protocol) front-end.

Implements the commands needed to drive the engine from any Go GUI::

    protocol_version  name  version  list_commands  boardsize  komi
    clear_board  play  genmove  undo  showboard  quit
    kata-analyze  (a tiny subset of KataGo's analysis command)

Coordinates follow GTP: columns ``A..T`` skipping ``I``, rows counted from the
bottom, so ``D4`` on a 9x9 board is the 4-4 point.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, TextIO

import numpy as np

from ..game import BLACK, PASS_MOVE, Position, Rules, WHITE
from ..nn.base import BaseNet
from .engine import RESIGN, Engine, gtp_to_move, move_to_gtp

NAME = "KataGo-PyTorch"
VERSION = "0.1.0"


class GTPEngine:
    def __init__(
        self,
        net: BaseNet,
        rules: Optional[Rules] = None,
        visits: int = 200,
        resign_threshold: Optional[float] = None,
        out: Optional[TextIO] = None,
    ) -> None:
        self.base_rules = rules or Rules(size=net.config.board_size)
        self.rules = self.base_rules
        self.net = net
        self.visits = visits
        self.resign_threshold = resign_threshold
        self.out = out or sys.stdout
        self.engine = Engine(net, self.rules, visits=visits)
        self.position = Position.empty(self.rules)
        self._history: List[Position] = []

    # ------------------------------------------------------------- commands
    def cmd_protocol_version(self, args: str) -> str:
        return "2"

    def cmd_name(self, args: str) -> str:
        return NAME

    def cmd_version(self, args: str) -> str:
        return VERSION

    def cmd_list_commands(self, args: str) -> str:
        return " ".join(sorted(self._commands()))

    def cmd_boardsize(self, args: str) -> str:
        size = int(args.strip())
        self.rules = self.base_rules.successor(size=size, max_moves=0)
        self.engine.rules = self.rules
        self.position = Position.empty(self.rules)
        self._history.clear()
        return ""

    def cmd_komi(self, args: str) -> str:
        komi = float(args.strip())
        self.rules = self.rules.successor(komi=komi)
        self.engine.rules = self.rules
        self.position = Position(
            rules=self.rules,
            grid=self.position.grid,
            to_play=self.position.to_play,
            ko_point=self.position.ko_point,
            move_history=self.position.move_history,
            consecutive_passes=self.position.consecutive_passes,
        )
        return ""

    def cmd_clear_board(self, args: str) -> str:
        self.position = Position.empty(self.rules)
        self._history.clear()
        return ""

    def cmd_play(self, args: str) -> str:
        parts = args.split()
        if len(parts) != 2:
            raise ValueError("play expects '<colour> <vertex>'")
        color_s, vertex = parts
        color = BLACK if color_s.lower() in ("b", "black") else WHITE
        if color != self.position.to_play:
            expected = "black" if self.position.to_play == BLACK else "white"
            raise ValueError(f"out of turn: {expected} to play")
        move = gtp_to_move(vertex, self.rules.size)
        self._history.append(self.position)
        self.position = self.position.play(move)
        return ""

    def cmd_undo(self, args: str) -> str:
        if not self._history:
            raise ValueError("cannot undo")
        self.position = self._history.pop()
        return ""

    def cmd_showboard(self, args: str) -> str:
        return "\n" + str(self.position)

    def cmd_genmove(self, args: str) -> str:
        color_s = args.strip()
        color = BLACK if color_s.lower() in ("b", "black") else WHITE
        if color != self.position.to_play:
            self.position = Position(
                rules=self.position.rules,
                grid=self.position.grid,
                to_play=color,
                ko_point=self.position.ko_point,
                move_history=self.position.move_history,
                consecutive_passes=self.position.consecutive_passes,
            )
        move = self.engine.genmove(
            self.position, visits=self.visits, resign_threshold=self.resign_threshold
        )
        if move == RESIGN:
            return "resign"
        self._history.append(self.position)
        self.position = self.position.play(move)
        return move_to_gtp(move, self.rules.size)

    def cmd_kata_analyze(self, args: str) -> str:
        """Report the engine's analysis of the current position."""
        a = self.engine.analyze(self.position, visits=self.visits)
        moves = " ".join(
            f"{move_to_gtp(m, self.rules.size)}:{p:.3f}" for m, p in a.top_moves(5)
        )
        pv = " ".join(move_to_gtp(m, self.rules.size) for m in a.principal_variation)
        return (
            f"winrate {a.win_rate:.3f} scoreLead {a.score_mean:+.1f} "
            f"visits {a.visits}\ninfo move {moves}\npv {pv}"
        )

    def cmd_quit(self, args: str) -> str:
        raise SystemExit(0)

    # ------------------------------------------------------------- dispatch
    def _commands(self):
        return {
            "protocol_version": self.cmd_protocol_version,
            "name": self.cmd_name,
            "version": self.cmd_version,
            "list_commands": self.cmd_list_commands,
            "boardsize": self.cmd_boardsize,
            "komi": self.cmd_komi,
            "clear_board": self.cmd_clear_board,
            "play": self.cmd_play,
            "undo": self.cmd_undo,
            "showboard": self.cmd_showboard,
            "genmove": self.cmd_genmove,
            "kata-analyze": self.cmd_kata_analyze,
            "quit": self.cmd_quit,
        }

    def handle(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""
        parts = line.split(None, 1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        handler = self._commands().get(cmd)
        if handler is None:
            return f"? unknown command: {cmd}"
        try:
            result = handler(args)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - GTP reports errors as text
            return f"? {exc}"
        # GTP responses are ``= <payload>``; an empty payload still needs the space
        return "= " + result

    def loop(self, stdin=None) -> None:  # pragma: no cover - interactive
        stdin = stdin or sys.stdin
        for line in stdin:
            try:
                response = self.handle(line)
            except SystemExit:
                break
            if response:
                print(response, file=self.out, flush=True)
                print("", file=self.out, flush=True)
