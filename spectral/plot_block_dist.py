"""
Per-block distribution of unit importance scores:
  top:    Phase 5b (ranked_units_by_block.csv)
  bottom: Phase 6 direct SVD (ranked_units_direct_by_block.csv)
One column per block; attn blocks (even ids) blue, MLP blocks (odd ids) orange.

Usage: python spectral/plot_block_dist.py [results/<experiment_name>]
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _read_by_block(csv_path: Path, imp_col: str = "importance") -> dict:
    data: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            bid = int(row["block_id"])
            data.setdefault(bid, []).append(float(row[imp_col]))
    return data


def _is_attn_block(bid: int) -> bool:
    return bid % 2 == 0   # even = attn, odd = MLP (layer_attn_mlp blocking)


def _plot_one(ax, data, title, bid_order, block_label_fn) -> None:
    xs, ys, cols = [], [], []
    xticks, xlabels = [], []
    for pos, bid in enumerate(bid_order):
        vals  = data.get(bid, [])
        color = "#4c8fd6" if _is_attn_block(bid) else "#e8834f"
        jitter = np.random.default_rng(bid).uniform(-0.25, 0.25, len(vals))
        xs.extend([pos + j for j in jitter])
        ys.extend(vals)
        cols.extend([color] * len(vals))
        if vals:
            ax.hlines(np.mean(vals), pos - 0.35, pos + 0.35,
                      color="red", linewidth=1.2, zorder=5)
        xticks.append(pos)
        xlabels.append(block_label_fn(bid))

    ax.scatter(xs, ys, c=cols, s=18, alpha=0.75, zorder=4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("importance")
    ax.set_xlabel("block")
    ax.grid(axis="y", alpha=0.3)

    attn_patch = mpatches.Patch(color="#4c8fd6", label="attn block")
    mlp_patch  = mpatches.Patch(color="#e8834f", label="MLP block")
    mean_line  = plt.Line2D([0], [0], color="red", linewidth=1.2,
                            label="block mean")
    ax.legend(handles=[attn_patch, mlp_patch, mean_line],
              fontsize=8, loc="upper right")


def plot_block_importance_dist(results_dir: Path) -> None:
    """
    Plot ranked_units_by_block.csv (Phase 5b) and
    ranked_units_direct_by_block.csv (Phase 6) side by side.
    A missing file skips its panel.
    """
    results_dir = Path(results_dir)
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_5b = results_dir / "ranked_units_by_block.csv"
    csv_6  = results_dir / "ranked_units_direct_by_block.csv"

    if not csv_5b.exists() and not csv_6.exists():
        print(f"[plot_block_dist] no input CSV in {results_dir}, skip.")
        return

    with open(results_dir / "blocks.json") as f:
        raw_blocks = json.load(f)
    blocks_meta = [b for b in raw_blocks if "unit_ids" in b]
    n_blocks = len(blocks_meta)

    def block_label(bid):
        b = blocks_meta[bid]
        return b.get("label", f"B{bid}")

    bid_order = list(range(n_blocks))

    panels = []
    if csv_5b.exists():
        panels.append(("Phase 5b (block-SVD importance, NOT normalized)",
                       _read_by_block(csv_5b)))
    if csv_6.exists():
        panels.append(("Phase 6 (direct SVD importance = total_var / d_u)",
                       _read_by_block(csv_6)))

    # double-column width: many blocks per panel
    from utils.paper_style import paper_size
    fig_w, fig_h_unit = paper_size("double", aspect=3.0)
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(fig_w, fig_h_unit * len(panels)),
                             constrained_layout=True, squeeze=False)
    for ax, (title, data) in zip(axes[:, 0], panels):
        _plot_one(ax, data, title, bid_order, block_label)

    out = plots_dir / "block_importance_dist.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    print(f"[Plot] {out}")


if __name__ == "__main__":
    default_dir = "results/gpt2_small_ioi_strong_N50_M10"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_dir)
    plot_block_importance_dist(target)
