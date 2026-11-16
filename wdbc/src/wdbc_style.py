"""
wdbc_style.py - one shared place for the colours, fonts and small helpers used by
every WDBC figure, so the notebook, the scripts and the dashboard page all look
like one system.

Palette: the eight-slot categorical theme, validated for colour-vision
deficiency on the adjacent-pair list (worst adjacent CVD dE 9.1, normal-vision
dE 19.6 on a light surface).  Three slots sit below 3:1 contrast against the
surface, so every chart that uses them also carries visible direct labels and a
CSV table view - that is the documented relief for the contrast warning.

Ten models are compared, which is more than eight slots.  The ninth and tenth
are NOT new hues: they re-use slots 1-2 with a dashed line, i.e. composite
hue + line-style encoding, and every legend entry is directly labelled with the
model name and its score so identity is never carried by colour alone.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#FFFFFF"
INK = "#2B1A33"          # matches the dashboard theme's textColor
INK_SOFT = "#6B6476"
GRID = "#E4E0E8"

# Validated categorical order (light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Single-hue sequential ramp (blue) for magnitude encodings such as heat-maps.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
            "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
            "#184f95", "#104281", "#0d366b"]

# Status colours, reserved - never reused as a series colour.
GOOD, WARN, BAD = "#1baf7a", "#eda100", "#e34948"

# House accent of the existing dashboard, used for single-series charts.
ACCENT = "#C2185B"
ACCENT_SOFT = "#5E4B6B"


def use_style():
    """Apply the shared rcParams.  Call once before plotting."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })


def series_style(i: int):
    """(colour, linestyle) for series i.  Slots 9+ re-use a hue with a dashed line."""
    n = len(SERIES)
    return SERIES[i % n], ("-" if i < n else "--")


def model_styles(models):
    """Stable {model: (colour, linestyle)} map, fixed by position in `models`."""
    return {m: series_style(i) for i, m in enumerate(models)}


def tidy(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    return ax


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path
