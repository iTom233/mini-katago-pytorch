#!/usr/bin/env python
"""A mouse-driven Go board for a trained checkpoint.

    python scripts/gui.py --checkpoint runs/demo9x9/model.npz --visits 200

Left click plays a move.  The toolbar handles pass, undo, resign and switching
colour.  The engine searches on a worker thread, so the window stays responsive.

Only Tkinter (part of the Python standard library) is required, so the GUI works
with either backend and with no extra dependencies.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from katago.game import BLACK, PASS_MOVE, WHITE, Position, Rules  # noqa: E402
from katago.game.features import encode_position  # noqa: E402
from katago.inference import Engine, move_to_gtp  # noqa: E402
from scripts._shared import build_from_checkpoint  # noqa: E402

BOARD_BG = "#dcb35c"
GRID_COLOR = "#4a3520"
STAR_COLOR = "#3a2a18"
BLACK_STONE = "#1d1d1d"
WHITE_STONE = "#f7f5f0"
LAST_MOVE = "#e53935"

#: star points for the board sizes KataGo trains on
STAR_POINTS = {
    9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
    13: [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)],
    19: [
        (3, 3), (3, 9), (3, 15),
        (9, 3), (9, 9), (9, 15),
        (15, 3), (15, 9), (15, 15),
    ],
}

GTP_LETTERS = "ABCDEFGHJKLMNOPQRST"


class BoardGUI:
    """Canvas board plus a control panel, wired to :class:`Engine`."""

    def __init__(self, root: tk.Tk, net, cfg, args, scale: float = 1.0) -> None:
        self.root = root
        self.net = net
        self.cfg = cfg
        self.scale = scale
        self.rules = Rules(size=cfg.board_size, komi=args.komi)
        self.board_px = int(args.board_px * scale)
        self.panel_px = int(250 * scale)

        self.human_color = tk.IntVar(value=BLACK if args.human_color == "black" else WHITE)
        self.visits = tk.IntVar(value=args.visits)
        self.show_analysis = tk.BooleanVar(value=not args.no_analysis)
        self.status_var = tk.StringVar()
        self.info_var = tk.StringVar()

        self.positions = [Position.empty(self.rules)]
        self.position = self.positions[0]
        self.moves: list[int] = []
        self.busy = False
        self.game_over = False
        self.generation = 0
        self.results: queue.Queue = queue.Queue()
        #: (mover, win rate, score mean, ownership) for the current position
        self.current_eval = None
        self.engine_pv: list[int] = []
        self.engine_elapsed = 0.0
        self.hover = None

        self.margin = 0.0
        self.cell = 0.0
        self.buttons: list[ttk.Button] = []

        self._build_ui()
        self._refresh_eval()
        self._draw_board()
        self._update_status()
        self._update_info()

        self.root.after(40, self._poll)
        # let the window paint before the engine's first move
        self.root.after(200, self._maybe_ai)

    # ------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        root = self.root
        backend = type(self.net).__name__.replace("Net", "").lower() or "net"
        root.title(f"KataGo-PyTorch — {self.rules.size}x{self.rules.size} ({backend})")
        root.configure(bg="#eceff1")
        root.resizable(False, False)

        left = ttk.Frame(root, padding=(10, 10, 6, 10))
        left.grid(row=0, column=0, sticky="n")
        self.canvas = tk.Canvas(
            left,
            width=self.board_px,
            height=self.board_px,
            bg=BOARD_BG,
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

        panel = ttk.Frame(root, padding=(6, 10, 12, 10))
        panel.grid(row=0, column=1, sticky="n")

        title_font = ("Microsoft YaHei UI", max(12, int(14 * self.scale)), "bold")
        ttk.Label(panel, text="KataGo-PyTorch", font=title_font).pack(anchor="w")
        ttk.Label(
            panel,
            text=f"{self.rules.size} 路 · 贴目 {self.rules.komi}",
            foreground="#607d8b",
        ).pack(anchor="w", pady=(0, 10))

        color_row = ttk.Frame(panel)
        color_row.pack(fill="x", pady=2)
        ttk.Label(color_row, text="你执").pack(side="left")
        ttk.Radiobutton(
            color_row, text="黑", variable=self.human_color, value=BLACK,
            command=self._new_game,
        ).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(
            color_row, text="白", variable=self.human_color, value=WHITE,
            command=self._new_game,
        ).pack(side="left")

        visit_row = ttk.Frame(panel)
        visit_row.pack(fill="x", pady=4)
        ttk.Label(visit_row, text="搜索量").pack(side="left")
        ttk.Combobox(
            visit_row,
            textvariable=self.visits,
            values=(50, 100, 200, 400, 800, 1600),
            width=6,
            state="readonly",
        ).pack(side="left", padx=6)
        ttk.Label(visit_row, text="visits").pack(side="left")

        ttk.Checkbutton(
            panel,
            text="显示形势图",
            variable=self.show_analysis,
            command=self._on_toggle_analysis,
        ).pack(anchor="w", pady=(2, 6))

        button_grid = ttk.Frame(panel)
        button_grid.pack(fill="x", pady=(2, 2))
        button_grid.columnconfigure(0, weight=1)
        button_grid.columnconfigure(1, weight=1)

        new_btn = ttk.Button(button_grid, text="新对局", command=self._new_game)
        new_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        undo_btn = ttk.Button(button_grid, text="悔棋", command=self._undo)
        undo_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        pass_btn = ttk.Button(button_grid, text="Pass", command=self._pass)
        pass_btn.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        resign_btn = ttk.Button(button_grid, text="认输", command=self._resign)
        resign_btn.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        self.buttons = [undo_btn, pass_btn, resign_btn]

        ttk.Separator(panel).pack(fill="x", pady=8)

        ttk.Label(
            panel,
            textvariable=self.status_var,
            font=("Microsoft YaHei UI", max(9, int(10 * self.scale)), "bold"),
            wraplength=self.panel_px,
        ).pack(anchor="w")
        ttk.Label(
            panel,
            textvariable=self.info_var,
            justify="left",
            wraplength=self.panel_px,
        ).pack(anchor="w", pady=(6, 0))

        ttk.Label(
            panel,
            text="左键落子 · 绿圈为悬停位置",
            foreground="#90a4ae",
        ).pack(anchor="w", pady=(12, 0))

    # --------------------------------------------------------------- drawing
    def _xy(self, col: int, row: int) -> tuple[float, float]:
        return self.margin + col * self.cell, self.margin + row * self.cell

    def _point_at(self, x: float, y: float):
        size = self.rules.size
        col = int(round((x - self.margin) / self.cell))
        row = int(round((y - self.margin) / self.cell))
        if not (0 <= col < size and 0 <= row < size):
            return None
        cx, cy = self._xy(col, row)
        if (x - cx) ** 2 + (y - cy) ** 2 > (self.cell * 0.55) ** 2:
            return None
        return col, row

    def _draw_board(self) -> None:
        c = self.canvas
        c.delete("all")
        size = self.rules.size
        px = self.board_px
        self.margin = px * 0.08
        self.cell = (px - 2 * self.margin) / (size - 1)
        line = max(1, int(self.scale))

        c.create_rectangle(0, 0, px, px, fill=BOARD_BG, outline="")
        c.create_rectangle(
            self.margin * 0.35, self.margin * 0.35,
            px - self.margin * 0.35, px - self.margin * 0.35,
            outline="#b98a45", width=max(1, int(2 * self.scale)),
        )

        for i in range(size):
            p = self.margin + i * self.cell
            c.create_line(self.margin, p, px - self.margin, p, fill=GRID_COLOR, width=line)
            c.create_line(p, self.margin, p, px - self.margin, fill=GRID_COLOR, width=line)

        star_r = max(2.0, self.cell * 0.075)
        for col, row in STAR_POINTS.get(size, []):
            x, y = self._xy(col, row)
            c.create_oval(x - star_r, y - star_r, x + star_r, y + star_r,
                          fill=STAR_COLOR, outline="")

        label_font = ("Microsoft YaHei UI", max(8, int(9 * self.scale)))
        for i in range(size):
            p = self.margin + i * self.cell
            c.create_text(p, px - self.margin * 0.42, text=GTP_LETTERS[i],
                          fill="#6b4a22", font=label_font)
            c.create_text(self.margin * 0.42, p, text=str(size - i),
                          fill="#6b4a22", font=label_font)

        if self.show_analysis.get() and self.current_eval is not None:
            self._draw_ownership()

        grid = self.position.grid
        for row in range(size):
            for col in range(size):
                if grid[row, col] != 0:
                    self._draw_stone(col, row, int(grid[row, col]))

        if self.moves:
            last = self.moves[-1]
            if 0 <= last < self.cfg.num_points:
                col, row = last % size, last // size
                x, y = self._xy(col, row)
                r = self.cell * 0.16
                on_black = int(grid[row, col]) == BLACK
                c.create_oval(x - r, y - r, x + r, y + r,
                              fill=LAST_MOVE if on_black else "#c62828", outline="")

    def _draw_stone(self, col: int, row: int, color: int) -> None:
        c = self.canvas
        x, y = self._xy(col, row)
        r = self.cell * 0.46
        c.create_oval(x - r + 1.5, y - r + 2.5, x + r + 1.5, y + r + 2.5,
                      fill="#a9824b", outline="")
        if color == BLACK:
            c.create_oval(x - r, y - r, x + r, y + r, fill=BLACK_STONE, outline="#000000")
            c.create_oval(x - r * 0.45, y - r * 0.55, x - r * 0.05, y - r * 0.15,
                          fill="#4a4a4a", outline="")
        else:
            c.create_oval(x - r, y - r, x + r, y + r, fill=WHITE_STONE, outline="#9e9e9e")
            c.create_oval(x - r * 0.45, y - r * 0.55, x - r * 0.05, y - r * 0.15,
                          fill="#ffffff", outline="")

    def _draw_ownership(self) -> None:
        mover, _, _, own = self.current_eval
        size = self.rules.size
        grid = self.position.grid
        half = self.cell * 0.46
        for row in range(size):
            for col in range(size):
                if grid[row, col] != 0:
                    continue
                idx = row * size + col
                share = float(own[idx]) if mover == BLACK else 1.0 - float(own[idx])
                if abs(share - 0.5) < 0.12:
                    continue
                x, y = self._xy(col, row)
                if share > 0.5:
                    fill, stipple = "#1b1b1b", "gray25"
                else:
                    fill, stipple = "#ffffff", "gray50"
                self.canvas.create_rectangle(
                    x - half, y - half, x + half, y + half,
                    fill=fill, stipple=stipple, outline="",
                )

    # ---------------------------------------------------------------- events
    def _on_click(self, event) -> None:
        if self.busy or self.game_over or self.position.is_terminal:
            return
        if self.position.to_play != self.human_color.get():
            return
        pt = self._point_at(event.x, event.y)
        if pt is None:
            return
        col, row = pt
        move = row * self.rules.size + col
        if not bool(self.position.legal_mask()[move]):
            self.status_var.set("这里不能落子（已有棋子 / 劫 / 自杀）")
            return
        self._play(move)

    def _on_motion(self, event) -> None:
        self.canvas.delete("hover")
        if self.busy or self.game_over or self.position.is_terminal:
            return
        if self.position.to_play != self.human_color.get():
            return
        pt = self._point_at(event.x, event.y)
        if pt is None:
            return
        col, row = pt
        if not bool(self.position.legal_mask()[row * self.rules.size + col]):
            return
        x, y = self._xy(col, row)
        r = self.cell * 0.46
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline="#2e7d32", width=max(1, int(2 * self.scale)), tags="hover",
        )

    def _on_leave(self, _event) -> None:
        self.canvas.delete("hover")

    # --------------------------------------------------------------- actions
    def _new_game(self) -> None:
        self.generation += 1
        self.busy = False
        self.game_over = False
        self.positions = [Position.empty(self.rules)]
        self.position = self.positions[0]
        self.moves = []
        self.engine_pv = []
        self.engine_elapsed = 0.0
        self._set_buttons(True)
        self._refresh_eval()
        self._draw_board()
        self._update_status()
        self._update_info()
        self.root.after(60, self._maybe_ai)

    def _undo(self) -> None:
        if self.busy or len(self.positions) <= 1:
            return
        self.positions.pop()
        self.moves.pop()
        # if it is now the engine's turn again, the human's move is still the
        # last move on the board; step over it too
        if len(self.positions) > 1 and self.positions[-1].to_play != self.human_color.get():
            self.positions.pop()
            self.moves.pop()
        self.position = self.positions[-1]
        self.game_over = False
        self.engine_pv = []
        self.engine_elapsed = 0.0
        self._refresh_eval()
        self._draw_board()
        self._update_status()
        self._update_info()

    def _pass(self) -> None:
        if self.busy or self.game_over or self.position.is_terminal:
            return
        if self.position.to_play != self.human_color.get():
            return
        self._play(PASS_MOVE)

    def _resign(self) -> None:
        if self.game_over:
            return
        self.game_over = True
        self.busy = False
        self.generation += 1
        self._set_buttons(True)
        winner = WHITE if self.human_color.get() == BLACK else BLACK
        self.status_var.set(
            f"你已认输，{'黑' if winner == BLACK else '白'}胜。点「新对局」再来一盘。"
        )

    def _play(self, move: int, pv=None, elapsed: float = 0.0) -> None:
        self.positions.append(self.position.play(move))
        self.moves.append(move)
        self.position = self.positions[-1]
        self.engine_pv = list(pv or [])
        self.engine_elapsed = elapsed
        self._refresh_eval()
        self._draw_board()
        self._update_status()
        self._update_info()
        if self.position.is_terminal:
            self._finish()
        else:
            self._maybe_ai()

    # ---------------------------------------------------------------- engine
    def _maybe_ai(self) -> None:
        if self.game_over or self.busy or self.position.is_terminal:
            return
        if self.position.to_play == self.human_color.get():
            return
        self.busy = True
        self._set_buttons(False)
        self._update_status()
        gen = self.generation
        threading.Thread(
            target=self._ai_worker,
            args=(self.position, int(self.visits.get()), gen),
            daemon=True,
        ).start()

    def _ai_worker(self, position: Position, visits: int, gen: int) -> None:
        started = time.perf_counter()
        try:
            engine = Engine(self.net, self.rules, visits=visits, seed=gen)
            analysis = engine.analyze(position, visits=visits)
            self.results.put((gen, "ok", analysis, time.perf_counter() - started))
        except Exception as exc:  # noqa: BLE001 - surfaced in the status bar
            self.results.put((gen, "error", exc, 0.0))

    def _poll(self) -> None:
        try:
            while True:
                gen, kind, payload, elapsed = self.results.get_nowait()
                if gen != self.generation:
                    continue
                self.busy = False
                self._set_buttons(True)
                if kind == "error":
                    self.status_var.set(f"AI 出错：{payload}")
                    continue
                if self.game_over or self.position.to_play == self.human_color.get():
                    self._update_status()
                    continue
                analysis = payload
                self._play(analysis.move, analysis.principal_variation, elapsed)
        except queue.Empty:
            pass
        except Exception as exc:  # noqa: BLE001 - never kill the poll loop
            self.busy = False
            self._set_buttons(True)
            self.status_var.set(f"AI 落子失败：{exc}")
        finally:
            self.root.after(40, self._poll)

    # --------------------------------------------------------------- helpers
    def _refresh_eval(self) -> None:
        """Evaluate the current position with the network (cheap, no search)."""
        if not self.show_analysis.get():
            self.current_eval = None
            return
        try:
            spatial, glob = encode_position(self.position)
            legal = self.position.legal_mask()
            out = self.net.predict(spatial[None], glob[None], legal[None])
            probs = out.score[0]
            bins = np.arange(len(probs)) - self.cfg.score_offset
            self.current_eval = (
                self.position.to_play,
                float(out.value[0, 1]),
                float((probs * bins).sum()),
                out.ownership[0].copy(),
            )
        except Exception:  # noqa: BLE001 - analysis is optional
            self.current_eval = None

    def _set_buttons(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        for button in self.buttons:
            button.state(state)

    def _update_status(self) -> None:
        if self.game_over:
            return
        turn = "黑" if self.position.to_play == BLACK else "白"
        who = "你" if self.position.to_play == self.human_color.get() else "AI"
        extra = "  ·  AI 思考中…" if self.busy else ""
        self.status_var.set(f"轮到：{turn}（{who}）{extra}")

    def _update_info(self) -> None:
        if self.game_over:
            return
        if self.current_eval is None:
            self.info_var.set("形势：—")
            return
        mover, win_rate, score_mean, _ = self.current_eval
        if mover == BLACK:
            black_win, black_score = win_rate, score_mean
        else:
            black_win, black_score = 1.0 - win_rate, -score_mean
        lines = [
            f"黑方胜率：{black_win:.1%}",
            f"黑方目差：{black_score:+.1f}",
        ]
        if self.engine_pv:
            pv = " ".join(self._gtp(m) for m in self.engine_pv[:5])
            lines.append(f"AI 主变：{pv}")
            lines.append(f"AI 用时：{self.engine_elapsed:.2f}s")
        self.info_var.set("\n".join(lines))

    def _finish(self) -> None:
        self.game_over = True
        self.busy = False
        self._set_buttons(True)
        score = self.position.score()
        if abs(score) < 1e-9:
            result = "和棋"
        elif score > 0:
            result = f"黑胜 {abs(score):.1f} 目"
        else:
            result = f"白胜 {abs(score):.1f} 目"
        self.status_var.set(f"对局结束：{result}（贴目 {self.rules.komi}）")

    def _gtp(self, move: int) -> str:
        if move == PASS_MOVE or move == self.cfg.num_points:
            return "pass"
        return move_to_gtp(move, self.rules.size)

    def _on_toggle_analysis(self) -> None:
        self._refresh_eval()
        self._draw_board()
        self._update_info()


def _enable_dpi_awareness() -> None:
    """Ask Windows for physical pixels so the board is not bitmap-scaled."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001 - older Windows, or no shcore
        try:
            from ctypes import windll

            windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def _configure_fonts(scale: float) -> None:
    size = max(9, int(10 * scale))
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family="Microsoft YaHei UI", size=size)
        except tk.TclError:
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    p.add_argument("--size", type=int, default=None, help="override the board size")
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--pool-blocks", type=int, default=None)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--visits", type=int, default=200)
    p.add_argument("--board-px", type=int, default=600,
                   help="board width in logical pixels")
    p.add_argument("--human-color", default="black", choices=["black", "white"])
    p.add_argument("--no-analysis", action="store_true",
                   help="start with the ownership overlay switched off")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _enable_dpi_awareness()
    net, cfg = build_from_checkpoint(
        backend=args.backend,
        checkpoint=args.checkpoint,
        size=args.size,
        blocks=args.blocks,
        channels=args.channels,
        pool_blocks=args.pool_blocks,
    )

    root = tk.Tk()
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
        scale = max(1.0, dpi / 96.0)
    except tk.TclError:
        scale = 1.0
    _configure_fonts(scale)

    gui = BoardGUI(root, net, cfg, args, scale)
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = max(0, (root.winfo_screenwidth() - w) // 2)
    y = max(0, (root.winfo_screenheight() - h) // 2)
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    del gui
    return 0


if __name__ == "__main__":
    sys.exit(main())
