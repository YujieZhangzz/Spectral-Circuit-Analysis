"""Re-render each subplot of the parameter-study figures as its own
single-column 600-dpi PDF (utils.paper_style).

Reads sweep_1d.csv and sweep_2d_*.csv written by parameter_study.py and writes
one PDF per threshold (1-D) or per category (2-D), plus a standalone legend,
to cross_task/<CT_RUN>/consistency_share/parameter_study/pdf/.

Usage: CT_RUN=run_gpt_12tasks python cross_task/parameter_study_split.py
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.paper_style import paper_rc, paper_size   # noqa: E402

# Config
RUN_ROOT  = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")
SRC_DIR   = RUN_ROOT / "consistency_share" / "parameter_study"
OUT_DIR   = SRC_DIR / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRITERION = "mean"
DPI       = 600

# Defaults (must match parameter_study.py)
TH_DEFAULT = {
    "universal_min":        0.08,
    "universal_mean":       0.20,
    "within_default":       0.10,
    "margin":               0.05,
    "task_dominance":       0.30,
    "task_outside_overlap": 0.15,
}

CATEGORY_KEYS = ["universal", "family_GT", "family_SVA", "task", "mixed"]
CATEGORY_LABELS = {
    "universal":  "Universal",
    "family_GT":  "Family-GT",
    "family_SVA": "Family-SVA",
    "task":       "Task-specific",
    "mixed":      "Mixed",
}
CATEGORY_COLORS = {
    "universal":  "#d62728",
    "family_GT":  "#1f77b4",
    "family_SVA": "#2ca02c",
    "task":       "#ff7f0e",
    "mixed":      "#888888",
}

PRETTY = {
    "universal":            r"$\theta_{\mathrm{univ}}$",
    "universal_min":        r"$\theta_{\mathrm{univ}}$",
    "universal_mean":       r"$\theta_{\mathrm{univ}}$",
    "within_default":       r"$\theta_F^{\mathrm{floor}}$",
    "margin":               r"$\delta$",
    "task_dominance":       r"$\theta_{\mathrm{dom}}$",
    "task_outside_overlap": r"$\theta_{\mathrm{out}}$",
}


# 1-D sweep splits
def split_1d():
    csv_path = SRC_DIR / "sweep_1d.csv"
    rows_by_param = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["criterion"] != CRITERION:
                continue
            rows_by_param[r["param"]].append(r)

    rc = paper_rc("single")
    for param, rows in rows_by_param.items():
        rows.sort(key=lambda r: float(r["value"]))
        xs = np.array([float(r["value"]) for r in rows])

        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.60))
            for k in CATEGORY_KEYS:
                ys = np.array([int(r[k]) for r in rows])
                ax.plot(xs, ys, label=CATEGORY_LABELS[k],
                        color=CATEGORY_COLORS[k])
            if param == "universal":
                default_x = TH_DEFAULT[f"universal_{CRITERION}"]
            else:
                default_x = TH_DEFAULT[param]
            ax.axvline(default_x, color="black", linestyle="--",
                       linewidth=0.8, alpha=0.6)
            ax.set_xlabel(PRETTY[param])
            ax.set_ylabel("Unit count")
            ax.grid(alpha=0.25, linewidth=0.5)
            # Legend is written separately (legend_1d_*.pdf) so it can be
            # placed once above a multi-panel figure.
            out = OUT_DIR / f"sweep_1d_{CRITERION}_{param}.pdf"
            fig.savefig(out, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"  [saved] {out}")


# 2-D sweep splits
def split_2d(tag, x_param, y_param, x_label, y_label, default_xy):
    csv_path = SRC_DIR / f"sweep_2d_{tag}_{CRITERION}.csv"
    rows = list(csv.DictReader(open(csv_path)))

    xs = sorted({float(r[x_param]) for r in rows})
    ys = sorted({float(r[y_param]) for r in rows})
    nX, nY = len(xs), len(ys)
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}

    panels = {
        "universal":    np.zeros((nY, nX), dtype=np.int32),
        "family_GT":    np.zeros((nY, nX), dtype=np.int32),
        "family_SVA":   np.zeros((nY, nX), dtype=np.int32),
        "family_total": np.zeros((nY, nX), dtype=np.int32),
    }
    for r in rows:
        i = y_idx[float(r[y_param])]; j = x_idx[float(r[x_param])]
        for k in panels:
            panels[k][i, j] = int(r[k])

    titles = {
        "universal":    "Universal",
        "family_GT":    "Family-GT",
        "family_SVA":   "Family-SVA",
        "family_total": "Family total",
    }

    rc = paper_rc("single")
    for k, mat in panels.items():
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.20))
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap="Blues",
                           extent=[min(xs), max(xs), min(ys), max(ys)])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(titles[k])
            ax.scatter([default_xy[0]], [default_xy[1]],
                       c="red", s=40, marker="x", linewidths=1.4)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6.5)
            out = OUT_DIR / f"sweep_2d_{tag}_{CRITERION}_{k}.pdf"
            fig.savefig(out, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"  [saved] {out}")


def make_legend_1d():
    """Standalone horizontal legend for the 1-D sweep panels."""
    rc = paper_rc("single")
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.0, 0.35))
        ax  = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        handles = [
            plt.Line2D([0], [0], color=CATEGORY_COLORS[k],
                       linewidth=plt.rcParams["lines.linewidth"],
                       label=CATEGORY_LABELS[k])
            for k in CATEGORY_KEYS
        ]
        leg = ax.legend(
            handles=handles, loc="center", ncol=len(handles),
            frameon=False, handlelength=1.6, columnspacing=1.4,
            fontsize=plt.rcParams["legend.fontsize"],
        )
        # Crop tightly to the legend's bbox so there is no whitespace
        # on either side. bbox_inches="tight" alone does not always do
        # this when the axes patch is still rendered.
        fig.canvas.draw()
        leg_bbox = leg.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        out = OUT_DIR / f"legend_1d_{CRITERION}.pdf"
        fig.savefig(out, dpi=DPI, bbox_inches=leg_bbox, pad_inches=0.02)
        plt.close(fig)
        print(f"  [saved] {out}")


def main():
    print(f"Output dir: {OUT_DIR}")
    print("\nLegend")
    make_legend_1d()
    print("\n1-D splits")
    split_1d()
    print("\n2-D splits (theta_F^floor x delta)")
    split_2d(
        tag="floor_x_margin",
        x_param="margin",
        y_param="within_default",
        x_label=r"$\delta$  (margin)",
        y_label=r"$\theta_F^{\mathrm{floor}}$",
        default_xy=(TH_DEFAULT["margin"], TH_DEFAULT["within_default"]),
    )
    print("\n2-D splits (theta_univ x delta)")
    univ_key = f"universal_{CRITERION}"
    split_2d(
        tag="univ_x_margin",
        x_param="margin",
        y_param=univ_key,
        x_label=r"$\delta$  (margin)",
        y_label=r"$\theta_{\mathrm{univ}}$",
        default_xy=(TH_DEFAULT["margin"], TH_DEFAULT[univ_key]),
    )


if __name__ == "__main__":
    main()
