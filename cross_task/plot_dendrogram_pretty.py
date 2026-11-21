"""Horizontal task dendrogram for the 12-task cross-task result.

Reads task_linkage and task_names from
cross_task/<CT_RUN>/<variant>/cross_task_result.pkl and writes
<variant>/dendrogram_pretty.{pdf,png}.

Usage:
    python cross_task/plot_dendrogram_pretty.py                    # all variants
    python cross_task/plot_dendrogram_pretty.py consistency_share
"""
from __future__ import annotations
import os, sys, pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

from utils.paper_style import paper_rc

ROOT = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")

PRETTY = {
    "col_obj":        "Colored Objects",
    "country_cap":    "Country → Capital",
    "gender_bias":    "Gender Bias",
    "gp":             "Gender Pronoun",
    "gt":             "Greater-Than",
    "gt_price":       "Greater-Than (price)",
    "ioi_strong":     "IOI",
    "person_country": "Person → Country",
    "person_pro":     "Person → Pronoun",
    "sva":            "SVA",
    "sva_blimp":      "SVA (BLiMP)",
    "sva_isare":      "SVA (is/are)",
}

PALETTE = ["#1F6FB2", "#D7793C", "#2CA02C", "#9467BD",
           "#E377C2", "#17BECF", "#BCBD22", "#8C564B"]


def plot_one(pkl_path: Path, out_base: Path, title: str,
             color_threshold: float = 0.82,
             x_pad: float = 0.03) -> None:
    r = pickle.loads(pkl_path.read_bytes())
    Z, names = r["task_linkage"], r["task_names"]
    labels = [PRETTY.get(n, n) for n in names]

    d_min = float(Z[:, 2].min())
    d_max = float(Z[:, 2].max())

    rc = paper_rc("double")
    rc["axes.spines.left"]   = False
    rc["axes.spines.bottom"] = True
    rc["axes.titlesize"]     = 12

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        set_link_color_palette(PALETTE)

        dendrogram(
            Z,
            labels=labels,
            ax=ax,
            orientation="left",
            color_threshold=color_threshold,
            above_threshold_color="#888888",
            leaf_font_size=10.5,
        )

        ax.axvline(color_threshold, ls="--", lw=0.9,
                   color="#B22222", alpha=0.7)
        ax.text(color_threshold, ax.get_ylim()[1],
                f"  cut = {color_threshold:.2f}",
                color="#B22222", fontsize=8, va="top", ha="left")

        # Zoom into the active range: drop the empty 0..d_min span so
        # small differences between merges are visible.
        x_lo = max(0.0, d_min - x_pad)
        x_hi = d_max + x_pad
        ax.set_xlim(x_hi, x_lo)

        ax.set_xlabel("Distance  (1 − similarity)")
        ax.set_title(title, pad=10)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3)
        ax.grid(axis="x", ls=":", lw=0.5, color="#CCCCCC", alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(out_base.with_suffix(".png"), dpi=300,
                    bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_base.with_suffix('.pdf')}")
        print(f"[saved] {out_base.with_suffix('.png')}")


def main() -> None:
    variants = sys.argv[1:] or [d.name for d in ROOT.iterdir() if d.is_dir()]
    for v in variants:
        pkl = ROOT / v / "cross_task_result.pkl"
        if not pkl.exists():
            print(f"[skip] {pkl} not found")
            continue
        plot_one(pkl, ROOT / v / "dendrogram_pretty", title=v)


if __name__ == "__main__":
    main()
