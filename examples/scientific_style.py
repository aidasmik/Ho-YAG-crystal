"""Shared visual language for the native thin-disk laboratory application."""

from __future__ import annotations

from tkinter import ttk

from cycler import cycler
from matplotlib import rcParams


PAPER = "#f4f6f5"
PANEL = "#ffffff"
INK = "#1c2b32"
MUTED = "#586a72"
RULE = "#ccd5d6"
NAVY = "#204e63"
NAVY_DARK = "#173b4c"
TEAL = "#176f78"
AMBER = "#8a611e"
SOFT = "#e8eeee"


def apply_scientific_style(root):
    """Apply a compact, platform-stable ttk theme and plot typography."""
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=("Segoe UI", 9), background=PAPER, foreground=INK)
    style.configure("TFrame", background=PAPER)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=PAPER, foreground=INK)
    style.configure("Panel.TLabel", background=PANEL, foreground=INK)
    style.configure("Input.TLabel", background=PAPER, foreground=MUTED,
                    font=("Segoe UI", 9))
    style.configure("Field.TLabel", background=PAPER, foreground=INK,
                    font=("Segoe UI", 9))
    style.configure("Eyebrow.TLabel", background=PAPER, foreground=NAVY,
                    font=("Segoe UI", 9, "bold"))
    style.configure("Info.TLabel", background=PAPER, foreground=MUTED,
                    font=("Segoe UI", 9))
    style.configure("Mono.TLabel", background=PAPER, foreground=INK,
                    font=("Cascadia Mono", 9))
    style.configure("Card.TFrame", background=PANEL, borderwidth=1,
                    relief="solid")
    style.configure("Card.TLabel", background=PANEL, foreground=MUTED,
                    font=("Segoe UI", 8, "bold"))
    style.configure("Metric.TLabel", background=PANEL, foreground=INK,
                    font=("Cascadia Mono", 15, "bold"))

    style.configure("TNotebook", background=PAPER, borderwidth=0,
                    tabmargins=(4, 4, 4, 0))
    style.configure("TNotebook.Tab", background=SOFT, foreground=MUTED,
                    borderwidth=0, padding=(12, 7), font=("Segoe UI", 9))
    style.map("TNotebook.Tab",
              background=[("selected", PANEL), ("active", "#e1e9e9")],
              foreground=[("selected", NAVY), ("active", INK)])

    style.configure("TEntry", fieldbackground=PANEL, foreground=INK,
                    bordercolor=RULE, lightcolor=RULE, darkcolor=RULE,
                    padding=(5, 4))
    style.configure("TCombobox", fieldbackground=PANEL, foreground=INK,
                    background=PANEL, bordercolor=RULE, lightcolor=RULE,
                    darkcolor=RULE, arrowcolor=NAVY, padding=(5, 4))
    style.map("TCombobox", fieldbackground=[("readonly", PANEL)],
              foreground=[("readonly", INK)])
    style.configure("TButton", background=SOFT, foreground=INK,
                    bordercolor=RULE, lightcolor=RULE, darkcolor=RULE,
                    padding=(10, 6), font=("Segoe UI", 9))
    style.map("TButton", background=[("active", "#dce5e6"),
                                      ("disabled", "#eff2f2")],
              foreground=[("disabled", "#89969a")])
    style.configure("Accent.TButton", background=NAVY, foreground=PANEL,
                    bordercolor=NAVY, lightcolor=NAVY, darkcolor=NAVY,
                    padding=(11, 7), font=("Segoe UI", 9, "bold"))
    style.map("Accent.TButton", background=[("active", NAVY_DARK),
                                             ("disabled", "#aebfc4")],
              foreground=[("active", PANEL), ("disabled", PANEL)])
    style.configure("TLabelframe", background=PAPER, bordercolor=RULE,
                    lightcolor=RULE, darkcolor=RULE, borderwidth=1,
                    relief="solid")
    style.configure("TLabelframe.Label", background=PAPER, foreground=NAVY,
                    font=("Segoe UI", 9, "bold"))
    style.configure("TSeparator", background=RULE)
    style.configure("TProgressbar", background=TEAL, troughcolor=SOFT,
                    bordercolor=SOFT)
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=INK, rowheight=24, borderwidth=0)
    style.configure("Treeview.Heading", background=SOFT, foreground=NAVY,
                    font=("Segoe UI", 9, "bold"), padding=(6, 5))

    rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "medium",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#84949a",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": PANEL,
        "axes.facecolor": PANEL,
        "savefig.facecolor": PANEL,
        "axes.prop_cycle": cycler(
            color=[NAVY, "#b66b32", TEAL, "#7d5e87", "#667c39"]),
    })
    return style
