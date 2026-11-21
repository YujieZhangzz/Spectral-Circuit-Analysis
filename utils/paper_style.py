"""Shared matplotlib style for paper figures.

Serif font (STIX Two Text) by default, single-column (3.5in) and double-column
(7in) size presets, and TrueType fonts embedded in PDFs (pdf.fonttype=42).

Usage:
    from utils.paper_style import paper_rc, paper_size

    with plt.rc_context(paper_rc("single")):
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.52))
        ax.plot(...)
        ax.set_xlabel("x", fontweight="bold")
        plt.savefig("foo.pdf", bbox_inches="tight")

To change the style everywhere, edit FONT_PRESET / PAPER_PRESETS / TYPE_STYLE.
"""

from __future__ import annotations
from typing import Dict, Tuple


# Single/double column size presets
# (NeurIPS / ICML / ACL: single 3.25-3.5in, double 6.75-7.0in)

PAPER_PRESETS: Dict[str, dict] = {
    "single": {
        "figwidth":  3.5,         # inches
        "label_fs":  8.5,
        "tick_fs":   7.5,
        "legend_fs": 7.5,
        "title_fs":  9.0,
        "linewidth": 1.6,
        "marker_sz": 4.0,
    },
    "double": {
        "figwidth":  7.0,
        "label_fs":  9.5,
        "tick_fs":   8.5,
        "legend_fs": 8.5,
        "title_fs":  10.0,
        "linewidth": 1.8,
        "marker_sz": 5.0,
    },
}
DEFAULT_PRESET = "single"


# Font presets

FONT_PRESET = "serif"       # "serif" | "sans" | "cm"

FONT_RCS = {
    "serif": {
        "font.family":      "serif",
        "font.serif":       ["STIX Two Text", "STIXGeneral",
                             "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    },
    "sans": {
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "Liberation Sans",
                             "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    },
    "cm": {
        "font.family":      "serif",
        "font.serif":       ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    },
}


# Colors per unit type

TYPE_STYLE = {
    "attn":  {"color": "#1F6FB2", "label": "Attention", "marker": "o"},
    "mlp":   {"color": "#D7793C", "label": "MLP",       "marker": "s"},
    "other": {"color": "#7B6FB2", "label": "Other",     "marker": "^"},
}

# General appearance
BG_AXES     = "white"
GRID_COLOR  = "#D5D5D5"
SPINE_COLOR = "#333333"


# Public API

def paper_rc(preset: str = DEFAULT_PRESET,
             font_preset: str = None) -> dict:
    """Return an rc dict for use with `plt.rc_context(...)`."""
    cfg  = PAPER_PRESETS[preset]
    font = FONT_RCS[font_preset or FONT_PRESET]
    return {
        "pdf.fonttype":      42,           # embed TrueType fonts
        "ps.fonttype":       42,
        "font.size":         cfg["tick_fs"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.edgecolor":    SPINE_COLOR,
        "axes.labelcolor":   "#222222",
        "axes.titlesize":    cfg["title_fs"],
        "axes.titleweight":  "bold",
        "axes.labelsize":    cfg["label_fs"],
        "axes.labelweight":  "bold",
        "xtick.color":       SPINE_COLOR,
        "ytick.color":       SPINE_COLOR,
        "xtick.labelsize":   cfg["tick_fs"],
        "ytick.labelsize":   cfg["tick_fs"],
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "legend.fontsize":   cfg["legend_fs"],
        "legend.frameon":    False,
        "lines.linewidth":   cfg["linewidth"],
        "lines.markersize":  cfg["marker_sz"],
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
        "savefig.dpi":       250,
        **font,
    }


def paper_size(preset: str = DEFAULT_PRESET,
               aspect: float = 1.52) -> Tuple[float, float]:
    """`aspect = width / height`. Default 1.52 is close to the golden ratio."""
    w = PAPER_PRESETS[preset]["figwidth"]
    return (w, w / aspect)


def get_preset(preset: str = DEFAULT_PRESET) -> dict:
    """Return the raw preset fields (linewidth, marker_sz, *_fs, ...)."""
    return PAPER_PRESETS[preset]


def apply_paper_style_globally(preset: str = DEFAULT_PRESET,
                               font_preset: str = None) -> None:
    """Apply the paper rcParams globally (called once at startup by run.py).

    Sets fonts, font sizes, bold labels/titles, TrueType PDF output and hides
    top/right spines. figsize is left to each script.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(paper_rc(preset, font_preset))
