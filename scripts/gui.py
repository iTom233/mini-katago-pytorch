#!/usr/bin/env python
"""A mouse-driven Go board for a trained checkpoint.

    python scripts/gui.py --checkpoint runs/demo9x9/model.npz --visits 200

Two modes:

* **play** -- you take one colour and the engine answers.
* **analysis** -- you place stones for either colour and the engine reports
  its top ten moves with visits, win rate and score.  The numbers refine live
  while the search runs and freeze once the visit budget is spent.

The engine searches on a worker thread, so the window stays responsive.

Only Tkinter (part of the Python standard library) is required, so the GUI works
with either backend and with no extra dependencies.
"""

from __future__ import annotations

import argparse
import os
import queue
import random
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
from katago.inference import AnalysisSession, Engine, move_to_gtp  # noqa: E402
from katago.utils.sound import play_stone_click, prepare_click  # noqa: E402
from scripts._shared import build_from_checkpoint  # noqa: E402

LAST_MOVE = "#e53935"

#: board looks; ``grain`` feeds the seeded background texture
BOARD_THEMES = {
    "wood": {
        "label": "木制",
        "bg": "#dcb35c",
        "border": "#b98a45",
        "grid": "#4a3520",
        "star": "#3a2a18",
        "coord": "#6b4a22",
        "hover": "#2e7d32",
        "shadow": "#b08c53",
        "shadow_soft": "#cba76c",
        "texture": "wood",
        "grain": ("#d3a852", "#e6c479"),
    },
    "stone": {
        "label": "石制",
        "bg": "#c9ced1",
        "border": "#8f979c",
        "grid": "#3d4449",
        "star": "#2b3136",
        "coord": "#555f66",
        "hover": "#2e7d32",
        "shadow": "#98a0a4",
        "shadow_soft": "#b6bcbf",
        "texture": "stone",
        "grain": ("#bcc2c5", "#d8dcde"),
    },
    "space": {
        "label": "星空",
        "bg": "#080d20",
        "border": "#25305c",
        "grid": "#42548c",
        "star": "#89a4f0",
        "coord": "#8296c9",
        "hover": "#7ee787",
        "shadow": "#000000",
        "shadow_soft": "#050a18",
        "texture": "space",
        "grain": ("#ffffff", "#bcd0ff", "#ffeab8"),
    },
}

BOARD_STYLE_BY_LABEL = {theme["label"]: key for key, theme in BOARD_THEMES.items()}

#: stone looks: rim -> shade -> body -> gloss -> spark fake a lit sphere
STONE_THEMES = {
    "classic": {
        "label": "黑白",
        "black": {
            "rim": "#000000",
            "shade": "#1a1a1a",
            "body": "#2e2e2e",
            "gloss": "#5c5c5c",
            "spark": "#9e9e9e",
            "vein": None,
        },
        "white": {
            "rim": "#9e9e9e",
            "shade": "#d9d4c9",
            "body": "#f7f5f0",
            "gloss": "#ffffff",
            "spark": "#ffffff",
            "vein": None,
        },
    },
    "gem": {
        "label": "玛瑙翡翠",
        "black": {
            "rim": "#0d0503",
            "shade": "#231009",
            "body": "#3a1a12",
            "gloss": "#7b3f2b",
            "spark": "#c07a45",
            "vein": "#8a4a2f",
        },
        "white": {
            "rim": "#7d9c80",
            "shade": "#cfe3c9",
            "body": "#eef6ea",
            "gloss": "#ffffff",
            "spark": "#d9f2d4",
            "vein": "#a8c9a4",
        },
    },
}

STONE_STYLE_BY_LABEL = {theme["label"]: key for key, theme in STONE_THEMES.items()}

#: interaction modes
MODE_PLAY = "play"
MODE_ANALYZE = "analyze"

#: marker colours by rank; everything past three shares the light blue
CANDIDATE_FILL = {1: "#0d47a1", 2: "#1565c0", 3: "#1976d2"}
CANDIDATE_FILL_OTHER = "#64b5f6"

#: flat light palette for the control panel -- one accent for primary actions,
#: muted ink for labels, faint ink for captions, so the eye lands on the board
PALETTE = {
    "app": "#e8ecf2",
    "panel": "#ffffff",
    "edge": "#d9e0ea",
    "card": "#f5f7fa",
    "card_edge": "#e3e9f2",
    "ink": "#1e2733",
    "ink_soft": "#55657a",
    "ink_faint": "#8b97a7",
    "accent": "#2f6feb",
    "accent_dark": "#1f56c4",
    "accent_soft": "#e7eeff",
    "danger": "#b93a32",
    "danger_soft": "#fdf1f0",
}

#: the board never shrinks below this many logical pixels, however small the window
MIN_BOARD_PX = 240

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
        self.panel_px = int(310 * scale)

        self.mode = tk.StringVar(value=args.mode)
        self.board_style = tk.StringVar(value=BOARD_THEMES[args.board_style]["label"])
        self.stone_style = tk.StringVar(value=STONE_THEMES[args.stone_style]["label"])
        self.human_color = tk.IntVar(value=BLACK if args.human_color == "black" else WHITE)
        self.visits = tk.IntVar(value=args.visits)
        self.analysis_visits = tk.IntVar(value=max(1, args.visits))
        self.show_analysis = tk.BooleanVar(value=not args.no_analysis)
        self.sound_on = tk.BooleanVar(value=not args.no_sound)
        self.status_var = tk.StringVar()
        self.info_var = tk.StringVar()
        self.hint_var = tk.StringVar()

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

        #: incremental analysis, refreshed from the worker thread
        self.analysis_report = None
        self.analysis_done = 0
        self.analysis_target = 0
        self.analysis_seconds = 0.0
        self.analysis_stopped = False
        self.analysis_stop: threading.Event | None = None

        self.margin = 0.0
        self.cell = 0.0
        self.buttons: list[ttk.Button] = []
        self.play_buttons: list[ttk.Button] = []
        self.analysis_buttons: list[ttk.Button] = []
        self.stop_button: ttk.Button | None = None
        #: the window is resizable; the board follows the largest square it gets
        self.panel: tk.Frame | None = None
        #: the white card inside the panel's hairline border
        self.body: ttk.Frame | None = None
        self.status_label: ttk.Label | None = None
        self._board_pad = (10, 10, 6, 10)
        self._wrap = self.panel_px
        #: width of the caption gutter that aligns every settings row
        self._gutter = int(42 * self.scale)
        self._row_gap = max(3, int(3 * self.scale))
        self.fullscreen = False
        self._resize_job = None
        self._pending_board_px = self.board_px
        self._panel_need: dict[str, int] = {}
        #: tallest info block seen, so the min height can follow a growing report
        self._info_lines = 0

        self._build_ui()
        self._refresh_eval()
        self._draw_board()
        self._update_status()
        self._update_info()

        self.root.after(40, self._poll)
        # let the window paint before the engine's first search
        if self.mode.get() == MODE_ANALYZE:
            self.root.after(200, self._start_analysis)
        else:
            self.root.after(200, self._maybe_ai)

    # ------------------------------------------------------------------- ui
    def _build_ui(self) -> None:
        root = self.root
        backend = type(self.net).__name__.replace("Net", "").lower() or "net"
        root.title(
            f"Mini-Katago — {self.rules.size}x{self.rules.size} ({backend}) · F11 全屏"
        )
        p = PALETTE
        root.configure(bg=p["app"])
        root.resizable(True, True)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", self._exit_fullscreen)
        self._install_styles()

        left = ttk.Frame(root, style="App.TFrame", padding=self._board_pad)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            left,
            width=self.board_px,
            height=self.board_px,
            bg=self._board_theme()["bg"],
            highlightthickness=0,
            cursor="hand2",
        )
        # no sticky: the square stays centred when the window grows
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        left.bind("<Configure>", self._on_board_resize)

        # a hairline frame around a white card lifts the panel off the backdrop
        edge = tk.Frame(root, bg=p["edge"])
        edge.grid(row=0, column=1, sticky="nsw", padx=(0, 12), pady=12)
        panel = ttk.Frame(edge, style="Panel.TFrame", padding=(16, 10, 16, 10))
        panel.pack(fill="both", expand=True)
        self.panel = edge
        self.body = panel
        self._wrap = max(220, self.panel_px - int(34 * self.scale))

        head = ttk.Frame(panel)
        head.pack(fill="x")
        ttk.Label(head, text="Mini-Katago", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text=backend, style="Chip.TLabel").pack(side="right", pady=(4, 0))
        ttk.Label(
            panel,
            text=f"{self.rules.size} 路 · 贴目 {self.rules.komi}",
            style="Caption.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        grid = self._settings_grid(panel, pady=(8, 0))

        mode_ctl = self._row(grid, 0, "模式")
        ttk.Radiobutton(
            mode_ctl,
            text="对弈",
            value=MODE_PLAY,
            variable=self.mode,
            command=self._on_mode_change,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_ctl,
            text="分析",
            value=MODE_ANALYZE,
            variable=self.mode,
            command=self._on_mode_change,
        ).pack(side="left", padx=(16, 0))

        option_ctl = self._row(grid, 1)
        ttk.Checkbutton(
            option_ctl,
            text="显示形势图",
            variable=self.show_analysis,
            command=self._on_toggle_analysis,
        ).pack(side="left")
        ttk.Checkbutton(
            option_ctl,
            text="落子声",
            variable=self.sound_on,
        ).pack(side="left", padx=(14, 0))

        look_ctl = self._row(grid, 2, "外观")
        ttk.Label(look_ctl, text="棋盘", style="Caption.TLabel").pack(side="left")
        board_box = ttk.Combobox(
            look_ctl,
            textvariable=self.board_style,
            values=[theme["label"] for theme in BOARD_THEMES.values()],
            state="readonly",
            width=5,
        )
        board_box.pack(side="left", padx=(4, 12))
        board_box.bind("<<ComboboxSelected>>", self._on_look_change)
        ttk.Label(look_ctl, text="棋子", style="Caption.TLabel").pack(side="left")
        stone_box = ttk.Combobox(
            look_ctl,
            textvariable=self.stone_style,
            values=[theme["label"] for theme in STONE_THEMES.values()],
            state="readonly",
            width=7,
        )
        stone_box.pack(side="left", padx=(4, 0))
        stone_box.bind("<<ComboboxSelected>>", self._on_look_change)

        # the two panels share one slot and are swapped by _show_mode_frames
        container = ttk.Frame(panel)
        container.pack(fill="x", pady=(2, 0))
        self.play_frame = ttk.Frame(container)
        self.analysis_frame = ttk.Frame(container)
        self._build_play_frame(self.play_frame)
        self._build_analysis_frame(self.analysis_frame)

        strip = tk.Frame(
            panel,
            bg=p["card"],
            highlightthickness=1,
            highlightbackground=p["card_edge"],
        )
        strip.pack(fill="x", pady=(12, 0))
        tk.Frame(
            strip, bg=p["accent"], width=max(3, int(3 * self.scale))
        ).pack(side="left", fill="y")
        inner = ttk.Frame(strip, style="Card.TFrame", padding=(10, 6, 10, 7))
        inner.pack(side="left", fill="x", expand=True)
        self.status_label = ttk.Label(
            inner,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=self._wrap,
            justify="left",
        )
        self.status_label.pack(anchor="w")
        ttk.Label(
            inner,
            textvariable=self.info_var,
            style="CardBody.TLabel",
            wraplength=self._wrap,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

        ttk.Label(
            panel,
            textvariable=self.hint_var,
            style="Hint.TLabel",
            wraplength=self._wrap,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        self._show_mode_frames()

    def _install_styles(self) -> None:
        """Flat light controls: one accent for the primary action, muted labels.

        ``clam`` is the only built-in theme that honours colours, so the whole
        panel gets one consistent surface instead of the native grey bevels.
        """
        p = PALETTE
        style = ttk.Style(self.root)
        style.theme_use("clam")

        def font(size: int, weight: str = "normal") -> tuple:
            return ("Microsoft YaHei UI", max(8, int(size * self.scale)), weight)

        body_size = 10
        style.configure(
            ".",
            background=p["panel"],
            foreground=p["ink"],
            fieldbackground="#ffffff",
            bordercolor=p["edge"],
            focuscolor=p["panel"],
            troughcolor=p["card"],
            selectbackground=p["accent_soft"],
            selectforeground=p["ink"],
        )
        style.configure("TFrame", background=p["panel"])
        style.configure("Panel.TFrame", background=p["panel"])
        style.configure("App.TFrame", background=p["app"])
        style.configure("Card.TFrame", background=p["card"])

        style.configure("TLabel", background=p["panel"], foreground=p["ink"])
        style.configure(
            "Title.TLabel",
            font=("Microsoft YaHei UI", max(13, int(15 * self.scale)), "bold"),
            foreground=p["ink"],
        )
        style.configure(
            "Chip.TLabel",
            background=p["accent_soft"],
            foreground=p["accent"],
            font=font(9, "bold"),
            padding=(7, 1),
        )
        style.configure("Caption.TLabel", foreground=p["ink_soft"], font=font(9))
        style.configure("Gutter.TLabel", foreground=p["ink_faint"], font=font(9, "bold"))
        style.configure("Soft.TLabel", foreground=p["ink_soft"])
        style.configure("Hint.TLabel", foreground=p["ink_faint"], font=font(8))
        style.configure("Status.TLabel", foreground=p["ink"], font=font(10, "bold"))
        style.configure("CardBody.TLabel", background=p["card"], foreground=p["ink_soft"])

        # clam ships ``-width -11`` (eleven average characters), which would make
        # every button -- and therefore the whole panel -- far too wide
        style.configure(
            "TButton",
            padding=(8, 3),
            width=0,
            background="#f2f5fa",
            foreground=p["ink"],
            bordercolor=p["edge"],
            lightcolor="#f2f5fa",
            darkcolor="#f2f5fa",
            relief="flat",
            focusthickness=0,
        )
        style.map(
            "TButton",
            background=[("disabled", "#f5f7fb"), ("pressed", "#dce4f1"), ("active", "#e8eef8")],
            foreground=[("disabled", "#aeb8c6")],
            bordercolor=[("disabled", p["edge"]), ("active", "#cfd9e8")],
        )
        style.configure(
            "Primary.TButton",
            background=p["accent"],
            foreground="#ffffff",
            bordercolor=p["accent"],
            lightcolor=p["accent"],
            darkcolor=p["accent"],
            font=font(body_size, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("disabled", "#b7c6e6"), ("pressed", p["accent_dark"]), ("active", "#3b7cf2")],
            foreground=[("disabled", "#f2f6ff")],
            bordercolor=[("disabled", "#b7c6e6"), ("pressed", p["accent_dark"]), ("active", "#3b7cf2")],
        )
        style.configure(
            "Danger.TButton",
            background=p["danger_soft"],
            foreground=p["danger"],
            bordercolor="#eed6d3",
            lightcolor=p["danger_soft"],
            darkcolor=p["danger_soft"],
        )
        style.map(
            "Danger.TButton",
            background=[("disabled", "#faf5f4"), ("pressed", "#f7e2df"), ("active", "#fbeae7")],
            foreground=[("disabled", "#d3b0ac")],
            bordercolor=[("active", "#e2bdb8")],
        )

        for name in ("TCheckbutton", "TRadiobutton"):
            style.configure(
                name,
                background=p["panel"],
                foreground=p["ink_soft"],
                focuscolor=p["panel"],
                padding=(0, 2),
            )
            style.map(
                name,
                background=[("active", p["panel"])],
                foreground=[("active", p["ink"]), ("disabled", p["ink_faint"])],
                indicatorcolor=[("selected", p["accent"])],
            )

        style.configure(
            "TCombobox",
            padding=(6, 2),
            arrowsize=max(10, int(11 * self.scale)),
            fieldbackground="#ffffff",
            background="#f2f5fa",
            foreground=p["ink"],
            arrowcolor=p["ink_soft"],
            bordercolor=p["edge"],
            lightcolor="#ffffff",
            darkcolor="#ffffff",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#ffffff"), ("disabled", p["card"])],
            foreground=[("disabled", p["ink_faint"])],
            background=[("active", "#e8eef8")],
            bordercolor=[("focus", p["accent"])],
            selectbackground=[("readonly", "#ffffff")],
            selectforeground=[("readonly", p["ink"])],
        )
        style.configure(
            "TSpinbox",
            padding=(6, 2),
            fieldbackground="#ffffff",
            background="#f2f5fa",
            foreground=p["ink"],
            arrowcolor=p["ink_soft"],
            bordercolor=p["edge"],
        )
        style.map(
            "TSpinbox",
            fieldbackground=[("disabled", p["card"])],
            bordercolor=[("focus", p["accent"])],
        )
        # the dropdown is a plain Tk listbox, styled through the option db
        for option, value in (
            ("*TCombobox*Listbox.background", "#ffffff"),
            ("*TCombobox*Listbox.foreground", p["ink"]),
            ("*TCombobox*Listbox.selectBackground", p["accent_soft"]),
            ("*TCombobox*Listbox.selectForeground", p["ink"]),
            ("*TCombobox*Listbox.borderWidth", 0),
        ):
            self.root.option_add(option, value)

        # ttk does not derive the row height from the font, so a DPI-scaled
        # font overflows the default 20px rows; measure it and set both
        self._tree_font = tkfont.Font(
            family="Microsoft YaHei UI", size=max(8, int(9 * self.scale))
        )
        self._tree_font_bold = tkfont.Font(
            family="Microsoft YaHei UI", size=max(8, int(9 * self.scale)), weight="bold"
        )
        style.configure(
            "Candidates.Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground=p["ink"],
            font=self._tree_font,
            borderwidth=0,
            relief="flat",
            rowheight=self._tree_font.metrics("linespace") + max(2, int(2 * self.scale)),
        )
        style.map(
            "Candidates.Treeview",
            background=[("selected", p["accent_soft"])],
            foreground=[("selected", p["ink"])],
        )
        style.configure(
            "Candidates.Treeview.Heading",
            font=font(9, "bold"),
            background=p["card"],
            foreground=p["ink_soft"],
            bordercolor=p["edge"],
            lightcolor=p["card"],
            darkcolor=p["card"],
            relief="flat",
            padding=(0, max(1, int(2 * self.scale))),
        )
        style.map(
            "Candidates.Treeview.Heading",
            background=[("active", "#e9eff8")],
        )

    def _settings_grid(self, parent: tk.Misc, pady=(0, 0)) -> ttk.Frame:
        """Settings grid: a muted caption gutter, then the controls."""
        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=pady)
        grid.columnconfigure(0, minsize=self._gutter)
        grid.columnconfigure(1, weight=1)
        return grid

    def _row(self, grid: ttk.Frame, row: int, label: str = "") -> ttk.Frame:
        """One settings row; the caption column keeps every control aligned."""
        if label:
            ttk.Label(grid, text=label, style="Gutter.TLabel").grid(
                row=row, column=0, sticky="w", pady=(0, self._row_gap)
            )
        controls = ttk.Frame(grid)
        controls.grid(row=row, column=1, sticky="w", pady=(0, self._row_gap))
        return controls

    def _action_grid(self, grid: ttk.Frame, row: int, columns: int = 2) -> ttk.Frame:
        """Even columns of buttons, separated from the settings above."""
        buttons = ttk.Frame(grid)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        for col in range(columns):
            buttons.columnconfigure(col, weight=1, uniform="action")
        # _grid_button reads this to keep the outer edges flush
        buttons.columns = columns
        return buttons

    def _grid_button(
        self,
        grid: ttk.Frame,
        text: str,
        command,
        row: int,
        col: int,
        style_name: str = "TButton",
        columnspan: int = 1,
    ) -> ttk.Button:
        """Place a button on the action grid with an even gutter."""
        half = max(2, int(5 * self.scale) // 2)
        last = getattr(grid, "columns", 2) - 1
        if col == 0:
            padx = (0, half)
        elif col >= last:
            padx = (half, 0)
        else:
            padx = (half, half)
        button = ttk.Button(grid, text=text, command=command, style=style_name)
        button.grid(
            row=row,
            column=col,
            columnspan=columnspan,
            sticky="ew",
            padx=padx,
            pady=(0, half * 2),
        )
        return button

    def _build_play_frame(self, parent: ttk.Frame) -> None:
        grid = self._settings_grid(parent)

        color_ctl = self._row(grid, 0, "你执")
        ttk.Radiobutton(
            color_ctl, text="黑", variable=self.human_color, value=BLACK,
            command=self._new_game,
        ).pack(side="left")
        ttk.Radiobutton(
            color_ctl, text="白", variable=self.human_color, value=WHITE,
            command=self._new_game,
        ).pack(side="left", padx=(14, 0))

        visit_ctl = self._row(grid, 1, "搜索量")
        ttk.Combobox(
            visit_ctl,
            textvariable=self.visits,
            values=(50, 100, 200, 400, 800, 1600),
            width=6,
            state="readonly",
        ).pack(side="left")
        ttk.Label(visit_ctl, text="visits", style="Caption.TLabel").pack(
            side="left", padx=(6, 0)
        )

        buttons = self._action_grid(grid, 2)
        self._grid_button(buttons, "新对局", self._new_game, 0, 0, "Primary.TButton")
        undo_btn = self._grid_button(buttons, "悔棋", self._undo, 0, 1)
        pass_btn = self._grid_button(buttons, "Pass", self._pass, 1, 0)
        resign_btn = self._grid_button(
            buttons, "认输", self._resign, 1, 1, "Danger.TButton"
        )
        self.play_buttons = [undo_btn, pass_btn, resign_btn]

    def _build_analysis_frame(self, parent: ttk.Frame) -> None:
        grid = self._settings_grid(parent)

        visit_ctl = self._row(grid, 0, "分析量")
        ttk.Spinbox(
            visit_ctl,
            from_=1,
            to=100000,
            increment=100,
            textvariable=self.analysis_visits,
            width=7,
        ).pack(side="left")
        ttk.Label(visit_ctl, text="visits", style="Caption.TLabel").pack(
            side="left", padx=(6, 0)
        )

        buttons = self._action_grid(grid, 1, columns=3)
        restart_btn = self._grid_button(
            buttons, "重新分析", self._start_analysis, 0, 0, "Primary.TButton",
            columnspan=2,
        )
        self.stop_button = self._grid_button(buttons, "停止", self._stop_analysis, 0, 2)
        undo_btn = self._grid_button(buttons, "悔棋", self._undo, 1, 0)
        pass_btn = self._grid_button(buttons, "Pass", self._pass, 1, 1)
        clear_btn = self._grid_button(
            buttons, "清空棋谱", self._new_game, 1, 2, "Danger.TButton"
        )
        self.analysis_buttons = [restart_btn, undo_btn, pass_btn, clear_btn]

        # a card behind the table keeps it apart from the raw controls above
        tree_card = tk.Frame(
            grid,
            bg=PALETTE["card"],
            highlightthickness=1,
            highlightbackground=PALETTE["card_edge"],
        )
        tree_card.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.candidate_tree = ttk.Treeview(
            tree_card,
            style="Candidates.Treeview",
            columns=("rank", "move", "visits", "win", "score"),
            show="headings",
            height=10,
            selectmode="none",
        )
        columns = (
            ("rank", "#", 26),
            ("move", "选点", 50),
            ("visits", "访问", 58),
            ("win", "胜率", 62),
            ("score", "目差", 56),
        )
        for key, title, width in columns:
            self.candidate_tree.heading(key, text=title)
            self.candidate_tree.column(
                key, width=int(width * self.scale), anchor="center", stretch=True
            )
        self.candidate_tree.tag_configure("band", background="#f8fafd")
        self.candidate_tree.tag_configure(
            "top", foreground="#17458f", font=self._tree_font_bold
        )
        self.candidate_tree.pack(fill="x", padx=1, pady=1)

    # ---------------------------------------------------------------- window
    def _toggle_fullscreen(self, event=None) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _exit_fullscreen(self, event=None) -> None:
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)

    def _on_board_resize(self, event) -> None:
        """Fit the board to the largest square the window leaves for it."""
        pad_x = self._board_pad[0] + self._board_pad[2]
        pad_y = self._board_pad[1] + self._board_pad[3]
        smallest = int(MIN_BOARD_PX * self.scale)
        side = max(smallest, min(event.width - pad_x, event.height - pad_y))
        if side == self._pending_board_px:
            return
        self._pending_board_px = side
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        # a 19x19 board is ~1000 canvas items; redraw once the drag settles
        self._resize_job = self.root.after(60, self._apply_board_resize)

    def _apply_board_resize(self) -> None:
        self._resize_job = None
        if self._pending_board_px == self.board_px:
            return
        self.board_px = self._pending_board_px
        self.canvas.configure(width=self.board_px, height=self.board_px)
        self._draw_board()

    def panel_min_height(self) -> int:
        """Requested height of the side panel in the current mode."""
        self.root.update_idletasks()
        return self.panel.winfo_reqheight()

    def _sync_window_minsize(self) -> None:
        """Let the window shrink, but never so far that the panel clips."""
        # ttk reports a short request until the pending relayout has run, so
        # remember the tallest request seen per mode instead of one read
        mode = self.mode.get()
        need = self.panel_min_height()
        self._panel_need[mode] = max(self._panel_need.get(mode, 0), need)
        need_h = min(self._panel_need[mode], self.root.winfo_screenheight() - 80)
        need_w = self.panel.winfo_reqwidth() + int(300 * self.scale)
        self.root.minsize(need_w, need_h)
        if (
            not self.fullscreen
            and self.root.winfo_ismapped()
            and self.root.winfo_height() < need_h
        ):
            # switching to the taller analysis panel grows the window for it
            self.root.geometry(f"{self.root.winfo_width()}x{need_h}")

    def _show_mode_frames(self) -> None:
        """Swap the play and analysis panels inside their shared slot."""
        if self.mode.get() == MODE_ANALYZE:
            self.play_frame.pack_forget()
            self.analysis_frame.pack(fill="x")
            self.buttons = self.analysis_buttons
            self.hint_var.set("点击棋盘打谱 · 圈内数字为排名 · F11 全屏")
        else:
            self.analysis_frame.pack_forget()
            self.play_frame.pack(fill="x")
            self.buttons = self.play_buttons
            self.hint_var.set("左键落子 · 绿圈为悬停位置 · F11 全屏")
        self._sync_stop_button()
        self._sync_window_minsize()

    # --------------------------------------------------------------- drawing
    def _board_theme(self) -> dict:
        return BOARD_THEMES[BOARD_STYLE_BY_LABEL[self.board_style.get()]]

    def _stone_palette(self, color: int) -> dict:
        theme = STONE_THEMES[STONE_STYLE_BY_LABEL[self.stone_style.get()]]
        return theme["black"] if color == BLACK else theme["white"]

    def _on_look_change(self, _event=None) -> None:
        """Repaint with the newly selected board or stone look."""
        self._draw_board()

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
        theme = self._board_theme()
        size = self.rules.size
        px = self.board_px
        self.margin = px * 0.08
        self.cell = (px - 2 * self.margin) / (size - 1)
        line = max(1, int(self.scale))

        c.configure(bg=theme["bg"])
        c.create_rectangle(0, 0, px, px, fill=theme["bg"], outline="")
        self._draw_board_texture(theme, px)
        c.create_rectangle(
            self.margin * 0.35, self.margin * 0.35,
            px - self.margin * 0.35, px - self.margin * 0.35,
            outline=theme["border"], width=max(1, int(2 * self.scale)),
        )

        for i in range(size):
            p = self.margin + i * self.cell
            c.create_line(self.margin, p, px - self.margin, p,
                          fill=theme["grid"], width=line)
            c.create_line(p, self.margin, p, px - self.margin,
                          fill=theme["grid"], width=line)

        star_r = max(2.0, self.cell * 0.075)
        for col, row in STAR_POINTS.get(size, []):
            x, y = self._xy(col, row)
            c.create_oval(x - star_r, y - star_r, x + star_r, y + star_r,
                          fill=theme["star"], outline="")

        label_font = ("Microsoft YaHei UI", max(8, int(9 * self.scale)))
        for i in range(size):
            p = self.margin + i * self.cell
            c.create_text(p, px - self.margin * 0.42, text=GTP_LETTERS[i],
                          fill=theme["coord"], font=label_font)
            c.create_text(self.margin * 0.42, p, text=str(size - i),
                          fill=theme["coord"], font=label_font)

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

        self._draw_candidates()

    def _draw_board_texture(self, theme: dict, px: float) -> None:
        """Paint the grain under the grid; seeded so a resize keeps the look."""
        c = self.canvas
        rng = random.Random(20240607)
        kind = theme["texture"]
        grain = theme["grain"]

        if kind == "wood":
            # long horizontal grain lines, plus a couple of knots
            for _ in range(max(12, int(px / 16))):
                y = rng.uniform(0.0, px)
                shade = grain[rng.randrange(len(grain))]
                c.create_line(
                    0, y, px, y + rng.uniform(-px * 0.012, px * 0.012),
                    fill=shade, width=max(1, int(rng.uniform(1.0, 2.4) * self.scale)),
                )
            for _ in range(3):
                x, y = rng.uniform(px * 0.12, px * 0.88), rng.uniform(px * 0.12, px * 0.88)
                rx, ry = rng.uniform(px * 0.03, px * 0.06), rng.uniform(px * 0.01, px * 0.02)
                c.create_oval(x - rx, y - ry, x + rx, y + ry,
                              outline=grain[0], width=max(1, int(self.scale)))
        elif kind == "stone":
            # speckle plus a few pale veins
            for _ in range(150):
                x, y = rng.uniform(0.0, px), rng.uniform(0.0, px)
                r = rng.uniform(px * 0.004, px * 0.016)
                c.create_oval(x - r, y - r, x + r, y + r,
                              fill=grain[rng.randrange(len(grain))], outline="")
            for _ in range(6):
                x0, y0 = rng.uniform(0.0, px), rng.uniform(0.0, px)
                x1 = x0 + rng.uniform(-px * 0.6, px * 0.6)
                y1 = y0 + rng.uniform(-px * 0.6, px * 0.6)
                c.create_line(x0, y0, x1, y1, fill=grain[0],
                              width=max(1, int(rng.uniform(0.8, 1.6) * self.scale)))
        else:  # space
            # faint nebulae, a dense dust of stars, then a few bright ones
            for _ in range(4):
                x = rng.uniform(px * 0.1, px * 0.9)
                y = rng.uniform(px * 0.1, px * 0.9)
                r = rng.uniform(px * 0.12, px * 0.26)
                c.create_oval(x - r, y - r, x + r, y + r, fill="#111a38", outline="")
            for _ in range(160):
                x, y = rng.uniform(0.0, px), rng.uniform(0.0, px)
                r = rng.uniform(px * 0.0012, px * 0.0035)
                c.create_oval(x - r, y - r, x + r, y + r,
                              fill=grain[rng.randrange(len(grain))], outline="")
            for _ in range(6):
                x = rng.uniform(px * 0.05, px * 0.95)
                y = rng.uniform(px * 0.05, px * 0.95)
                r = rng.uniform(px * 0.004, px * 0.007)
                c.create_oval(x - r * 2.2, y - r * 2.2, x + r * 2.2, y + r * 2.2,
                              outline="#24305c", width=1)
                c.create_oval(x - r, y - r, x + r, y + r, fill="#ffffff", outline="")

    def _draw_candidates(self) -> None:
        """Number the top analysis moves on the board (rank 1..10)."""
        self.canvas.delete("candidate")
        if self.mode.get() != MODE_ANALYZE or self.analysis_report is None:
            return
        size = self.rules.size
        grid = self.position.grid
        font = ("Microsoft YaHei UI", max(7, int(self.cell * 0.34)), "bold")
        line = max(1, int(self.scale))
        for rank, info in enumerate(self.analysis_report.moves[:10], 1):
            move = info.move
            if move < 0 or move >= self.cfg.num_points:
                continue  # pass has no intersection; it stays in the table
            col, row = move % size, move // size
            if grid[row, col] != 0:
                continue
            x, y = self._xy(col, row)
            r = self.cell * 0.40
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=CANDIDATE_FILL.get(rank, CANDIDATE_FILL_OTHER),
                outline="#ffffff", width=line, tags="candidate",
            )
            self.canvas.create_text(
                x, y, text=str(rank), fill="#ffffff", font=font, tags="candidate",
            )

    def _draw_stone(self, col: int, row: int, color: int) -> None:
        c = self.canvas
        x, y = self._xy(col, row)
        r = self.cell * 0.46
        pal = self._stone_palette(color)
        board = self._board_theme()

        # drop shadow: a wide soft disc plus a tighter one under the stone
        c.create_oval(
            x - r * 1.03 + r * 0.14, y - r * 1.03 + r * 0.18,
            x + r * 1.03 + r * 0.14, y + r * 1.03 + r * 0.18,
            fill=board["shadow_soft"], outline="",
        )
        c.create_oval(
            x - r + r * 0.05, y - r + r * 0.07,
            x + r + r * 0.05, y + r + r * 0.07,
            fill=board["shadow"], outline="",
        )
        # rim -> shaded ring -> body give the stone a convex profile
        c.create_oval(x - r, y - r, x + r, y + r, fill=pal["rim"], outline="")
        outer = r * 0.94
        c.create_oval(x - outer, y - outer, x + outer, y + outer,
                      fill=pal["shade"], outline="")
        inner = r * 0.83
        c.create_oval(
            x - inner - r * 0.03, y - inner - r * 0.04,
            x + inner - r * 0.03, y + inner - r * 0.04,
            fill=pal["body"], outline="",
        )
        vein = pal.get("vein")
        if vein:
            # agate banding / jade mottling across the lower half
            c.create_arc(
                x - r * 0.66, y - r * 0.40, x + r * 0.66, y + r * 0.58,
                start=205, extent=130, style="arc", outline=vein,
                width=max(1, int(1.6 * self.scale)),
            )
        # specular highlight top-left, plus a soft bounce bottom-right
        c.create_oval(
            x - r * 0.58, y - r * 0.70, x - r * 0.04, y - r * 0.16,
            fill=pal["gloss"], outline="",
        )
        c.create_oval(
            x - r * 0.42, y - r * 0.56, x - r * 0.20, y - r * 0.34,
            fill=pal["spark"], outline="",
        )
        c.create_oval(
            x + r * 0.18, y + r * 0.44, x + r * 0.58, y + r * 0.70,
            fill=pal["gloss"], outline="",
        )

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
        if self.game_over or self.position.is_terminal:
            return
        if self.mode.get() == MODE_PLAY:
            if self.busy or self.position.to_play != self.human_color.get():
                return
        # in analysis mode the board stays live during a search: the move wins
        # over the running analysis, which is cancelled in _play
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
        if self.game_over or self.position.is_terminal:
            return
        if self.mode.get() == MODE_PLAY:
            if self.busy or self.position.to_play != self.human_color.get():
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
            outline=self._board_theme()["hover"],
            width=max(1, int(2 * self.scale)), tags="hover",
        )

    def _on_leave(self, _event) -> None:
        self.canvas.delete("hover")

    # --------------------------------------------------------------- actions
    def _new_game(self) -> None:
        self._halt_analysis()
        self.game_over = False
        self.positions = [Position.empty(self.rules)]
        self.position = self.positions[0]
        self.moves = []
        self.engine_pv = []
        self.engine_elapsed = 0.0
        self._reset_analysis_state()
        self._set_buttons(True)
        self._refresh_eval()
        self._draw_board()
        self._fill_candidate_table()
        self._update_status()
        self._update_info()
        if self.mode.get() == MODE_ANALYZE:
            self.root.after(60, self._start_analysis)
        else:
            self.root.after(60, self._maybe_ai)

    def _undo(self) -> None:
        if self.mode.get() == MODE_PLAY and self.busy:
            return
        if len(self.positions) <= 1:
            return
        self._halt_analysis()
        self.positions.pop()
        self.moves.pop()
        if self.mode.get() == MODE_PLAY:
            # if it is now the engine's turn again, the human's move is still
            # the last move on the board; step over it too
            if len(self.positions) > 1 and self.positions[-1].to_play != self.human_color.get():
                self.positions.pop()
                self.moves.pop()
        self.position = self.positions[-1]
        self.game_over = False
        self.engine_pv = []
        self.engine_elapsed = 0.0
        self._reset_analysis_state()
        self._set_buttons(True)
        self._refresh_eval()
        self._draw_board()
        self._fill_candidate_table()
        self._update_status()
        self._update_info()
        if self.mode.get() == MODE_ANALYZE:
            self._start_analysis()

    def _pass(self) -> None:
        if self.game_over or self.position.is_terminal:
            return
        if self.mode.get() == MODE_PLAY:
            if self.busy or self.position.to_play != self.human_color.get():
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
        if self.mode.get() == MODE_ANALYZE:
            self._halt_analysis()
        self.positions.append(self.position.play(move))
        self.moves.append(move)
        self.position = self.positions[-1]
        if move != PASS_MOVE and self.sound_on.get():
            play_stone_click(self.root.bell)
        self.engine_pv = list(pv or [])
        self.engine_elapsed = elapsed
        self._reset_analysis_state()
        self._refresh_eval()
        self._draw_board()
        self._fill_candidate_table()
        self._update_status()
        self._update_info()
        if self.position.is_terminal:
            self._finish()
        elif self.mode.get() == MODE_ANALYZE:
            self._start_analysis()
        else:
            self._maybe_ai()

    # --------------------------------------------------------------- analysis
    def _reset_analysis_state(self) -> None:
        """Drop the current report so the board stops showing stale candidates."""
        self.analysis_report = None
        self.analysis_done = 0
        self.analysis_target = 0
        self.analysis_seconds = 0.0
        self.analysis_stopped = False

    def _halt_analysis(self) -> None:
        """Cancel a running search and invalidate its pending results."""
        if self.analysis_stop is not None:
            self.analysis_stop.set()
            self.analysis_stop = None
        self.generation += 1
        self.busy = False

    def _start_analysis(self) -> None:
        if self.mode.get() != MODE_ANALYZE:
            return
        self._halt_analysis()
        self._reset_analysis_state()
        self._fill_candidate_table()
        if self.position.is_terminal:
            self._draw_candidates()
            self._update_status()
            self._update_info()
            return
        try:
            budget = int(self.analysis_visits.get())
        except (tk.TclError, ValueError):
            budget = 200
        budget = max(1, min(100_000, budget))
        self.analysis_visits.set(budget)
        self.analysis_target = budget
        self.busy = True
        self._set_buttons(False)
        self._update_status()
        self._update_info()
        gen = self.generation
        stop = threading.Event()
        self.analysis_stop = stop
        threading.Thread(
            target=self._analysis_worker,
            args=(self.position, budget, gen, stop),
            daemon=True,
        ).start()

    def _analysis_worker(
        self, position: Position, budget: int, gen: int, stop: threading.Event
    ) -> None:
        started = time.perf_counter()
        try:
            session = AnalysisSession(self.net, self.rules, batch_size=8, seed=gen)
            session.set_position(position)
            # show the raw priors immediately so the table is never empty while
            # the first playouts run
            self.results.put((gen, "progress", (session.report(), 0, budget, 0.0), 0.0))
            done = 0
            chunk = max(8, min(64, budget // 20))
            last_report = 0.0
            while done < budget and not stop.is_set():
                report, ran = session.step(min(chunk, budget - done))
                done += ran
                now = time.perf_counter()
                # throttle snapshots by wall clock so a fast search cannot flood
                # the event loop with table redraws
                if now - last_report >= 0.05 or done >= budget or ran <= 0:
                    last_report = now
                    elapsed = now - started
                    self.results.put(
                        (gen, "progress", (report, done, budget, elapsed), elapsed)
                    )
                if ran <= 0:
                    break
            elapsed = time.perf_counter() - started
            self.results.put(
                (gen, "done",
                 (session.report(), done, budget, elapsed, stop.is_set()), elapsed)
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the status bar
            self.results.put((gen, "error", exc, 0.0))

    def _stop_analysis(self) -> None:
        if not self.busy or self.mode.get() != MODE_ANALYZE:
            return
        self._halt_analysis()
        self.analysis_stopped = True
        self._set_buttons(True)
        self._update_status()
        self._update_info()

    # ----------------------------------------------------------------- engine
    def _maybe_ai(self) -> None:
        if self.mode.get() != MODE_PLAY:
            return
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
            self.results.put((gen, "move", analysis, time.perf_counter() - started))
        except Exception as exc:  # noqa: BLE001 - surfaced in the status bar
            self.results.put((gen, "error", exc, 0.0))

    def _poll(self) -> None:
        try:
            pending = []
            while True:
                try:
                    pending.append(self.results.get_nowait())
                except queue.Empty:
                    break
            # a fast search can enqueue many snapshots between two polls; only
            # the last one is worth drawing
            latest_progress = None
            for gen, kind, payload, elapsed in pending:
                if gen != self.generation:
                    continue
                if kind == "progress":
                    latest_progress = payload
                    continue
                if latest_progress is not None:
                    self._apply_progress(latest_progress)
                    latest_progress = None
                if kind == "error":
                    self.busy = False
                    self._set_buttons(True)
                    self.status_var.set(f"引擎出错：{payload}")
                elif kind == "move":
                    self.busy = False
                    self._set_buttons(True)
                    if self.game_over or self.mode.get() != MODE_PLAY:
                        self._update_status()
                        continue
                    if self.position.to_play == self.human_color.get():
                        self._update_status()
                        continue
                    analysis = payload
                    self._play(analysis.move, analysis.principal_variation, elapsed)
                elif kind == "done":
                    self.busy = False
                    report, done, budget, seconds, stopped = payload
                    self.analysis_report = report
                    self.analysis_done = done
                    self.analysis_target = budget
                    self.analysis_seconds = seconds
                    self.analysis_stopped = stopped
                    self._set_buttons(True)
                    self._refresh_analysis_view()
            if latest_progress is not None:
                self._apply_progress(latest_progress)
        except Exception as exc:  # noqa: BLE001 - never kill the poll loop
            self.busy = False
            self._set_buttons(True)
            self.status_var.set(f"引擎失败：{exc}")
        finally:
            self.root.after(40, self._poll)

    def _apply_progress(self, payload) -> None:
        report, done, budget, seconds = payload
        self.analysis_report = report
        self.analysis_done = done
        self.analysis_target = budget
        self.analysis_seconds = seconds
        self._refresh_analysis_view()

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
        self._sync_stop_button()

    def _sync_stop_button(self) -> None:
        if self.stop_button is None:
            return
        running = self.busy and self.mode.get() == MODE_ANALYZE
        self.stop_button.state(["!disabled"] if running else ["disabled"])

    def _refresh_analysis_view(self) -> None:
        if self.mode.get() != MODE_ANALYZE:
            return
        self._fill_candidate_table()
        self._draw_candidates()
        self._update_info()

    def _fill_candidate_table(self) -> None:
        tree = self.candidate_tree
        tree.delete(*tree.get_children())
        if self.analysis_report is None:
            return
        for rank, info in enumerate(self.analysis_report.moves[:10], 1):
            if info.visits > 0:
                win = f"{info.win_rate:.1%}"
                score = f"{info.score_mean:+.1f}"
            else:
                win, score = "—", "—"
            tags = ["top"] if rank <= 3 else []
            if rank % 2 == 0:
                tags.append("band")
            tree.insert(
                "", "end",
                values=(rank, self._gtp(info.move), info.visits, win, score),
                tags=tuple(tags),
            )

    def _update_status(self) -> None:
        if self.mode.get() == MODE_ANALYZE:
            if self.game_over:
                return
            turn = "黑" if self.position.to_play == BLACK else "白"
            if self.position.is_terminal:
                self.status_var.set("棋局已结束 · 点「清空棋谱」重新开始")
                self._set_status_tone("done")
            elif self.busy:
                self.status_var.set(f"分析模式 · 轮到{turn} · 分析中…")
                self._set_status_tone("busy")
            else:
                self.status_var.set(f"分析模式 · 轮到{turn} · 点击棋盘打谱")
                self._set_status_tone("idle")
            return
        if self.game_over:
            return
        turn = "黑" if self.position.to_play == BLACK else "白"
        who = "你" if self.position.to_play == self.human_color.get() else "AI"
        extra = "  ·  AI 思考中…" if self.busy else ""
        self.status_var.set(f"轮到：{turn}（{who}）{extra}")
        self._set_status_tone("busy" if self.busy else "idle")

    def _set_status_tone(self, tone: str) -> None:
        """Colour the headline: accent while the engine thinks, red when done."""
        if self.status_label is None:
            return
        colour = {"idle": PALETTE["ink"], "busy": PALETTE["accent"], "done": PALETTE["danger"]}
        self.status_label.configure(foreground=colour[tone])

    def _set_info(self, text: str) -> None:
        """Fill the info block, resyncing the min height when it gains lines.

        The block starts with two lines and grows to four once a report (or a
        PV) arrives, so the height measured at startup would clip the tail.
        """
        if text != self.info_var.get():
            self.info_var.set(text)
        lines = text.count("\n") + 1
        if lines > self._info_lines:
            self._info_lines = lines
            self._sync_window_minsize()

    def _update_info(self) -> None:
        if self.mode.get() == MODE_ANALYZE:
            self._update_analysis_info()
            return
        if self.game_over:
            return
        if self.current_eval is None:
            self._set_info("形势：—")
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
        self._set_info("\n".join(lines))

    def _update_analysis_info(self) -> None:
        if self.position.is_terminal:
            self._set_info("棋局已结束，没有可分析的候选点")
            return
        report = self.analysis_report
        if report is None:
            self._set_info("形势：等待分析…")
            return
        black_win, black_score = self._black_view(report.win_rate, report.score_mean)
        lines = [
            f"黑方胜率：{black_win:.1%}",
            f"黑方目差：{black_score:+.1f}",
        ]
        if self.analysis_target:
            if self.busy:
                state = "分析中"
            elif self.analysis_stopped:
                state = "已停止"
            else:
                state = "已完成"
            lines.append(
                f"搜索：{self.analysis_done} / {self.analysis_target} visits（{state}）"
            )
        if self.analysis_seconds:
            lines.append(f"用时：{self.analysis_seconds:.2f}s")
        self._set_info("\n".join(lines))

    def _black_view(self, win_rate: float, score_mean: float) -> tuple[float, float]:
        """Flip a value from the player to move's point of view to Black's."""
        if self.position.to_play == BLACK:
            return win_rate, score_mean
        return 1.0 - win_rate, -score_mean

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

    def _on_mode_change(self) -> None:
        """Keep the stones, swap the panel, and restart the search for this turn."""
        self._halt_analysis()
        self.game_over = False
        self.engine_pv = []
        self.engine_elapsed = 0.0
        self._reset_analysis_state()
        self._show_mode_frames()
        self._set_buttons(True)
        self._refresh_eval()
        self._draw_board()
        self._fill_candidate_table()
        self._update_status()
        self._update_info()
        if self.mode.get() == MODE_ANALYZE:
            self._start_analysis()
        else:
            self.root.after(80, self._maybe_ai)

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
    p.add_argument("--visits", type=int, default=200,
                   help="search budget for play mode / analysis budget")
    p.add_argument("--mode", default=MODE_PLAY, choices=(MODE_PLAY, MODE_ANALYZE),
                   help="start in human-vs-engine play or in free analysis mode")
    p.add_argument("--board-px", type=int, default=600,
                   help="board width in logical pixels")
    p.add_argument("--human-color", default="black", choices=["black", "white"])
    p.add_argument("--board-style", default="wood", choices=tuple(BOARD_THEMES),
                   help="board look: wood, stone or a starfield")
    p.add_argument("--stone-style", default="classic", choices=tuple(STONE_THEMES),
                   help="stone look: plain black/white or agate and jade")
    p.add_argument("--no-analysis", action="store_true",
                   help="start with the ownership overlay switched off")
    p.add_argument("--no-sound", action="store_true",
                   help="start with the stone click switched off")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _enable_dpi_awareness()
    if not args.no_sound:
        try:
            # resolve the click up front so the first stone is never delayed
            prepare_click()
        except OSError:
            pass
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
    # ttk applies the candidate-tree row height one idle round after the style
    # is configured, so update_idletasks() alone would measure the panel short
    root.update()
    gui._sync_window_minsize()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    w = min(root.winfo_width(), screen_w - 40)
    h = min(root.winfo_height(), screen_h - 80)
    x = max(0, (screen_w - w) // 2)
    y = max(0, (screen_h - h) // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.mainloop()
    del gui
    return 0


if __name__ == "__main__":
    sys.exit(main())
