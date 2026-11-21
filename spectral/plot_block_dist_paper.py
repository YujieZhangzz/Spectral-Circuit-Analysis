"""
Paper-style version of block_importance_dist.png.

One x-tick per layer, with attn (left) and MLP (right) side by side. The score
is ||M_u||_F^2 = sum over (sigma_B/sqrt(d_block))^2 weighted by U_B[idx_u, :],
i.e. the unit's energy share in the block-SVD basis.

Usage:
    python spectral/plot_block_dist_paper.py [results/<experiment_name>] [--log]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paper_style import paper_rc, paper_size, TYPE_STYLE


def _read_by_block(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "block_id":  int(r["block_id"]),
                "layer":     int(r["layer"]),
                "unit_type": r["unit_type"],
                "label":     r["label"],
                "importance": float(r["importance"]),
            })
    return rows


def _split_attn_mlp(rows):
    by_layer = {}
    for r in rows:
        d = by_layer.setdefault(r["layer"], {"attn": [], "mlp": []})
        if r["unit_type"] == "attn_head":
            d["attn"].append(r["importance"])
        else:
            d["mlp"].append(r["importance"])
    return by_layer


def plot(results_dir: Path, log_y: bool = True, out_name: str = "block_importance_dist.png"):
    results_dir = Path(results_dir)
    csv_path = results_dir / "ranked_units_by_block.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows = _read_by_block(csv_path)
    by_layer = _split_attn_mlp(rows)
    layers = sorted(by_layer.keys())

    attn_color = TYPE_STYLE["attn"]["color"]
    mlp_color  = TYPE_STYLE["mlp"]["color"]
    mean_color = "#C0392B"

    # geometry
    pair_offset = 0.22                 # attn / mlp offset within a layer slot
    jitter_w    = 0.14                 # half-width of the within-column jitter
    rng         = np.random.default_rng(0)

    with plt.rc_context(paper_rc("double")):
        fig_w, fig_h = paper_size("double", aspect=2.4)
        # extra headroom in height so the legend can sit above the data
        fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.4),
                               constrained_layout=True)

        for L in layers:
            attn_vals = by_layer[L]["attn"]
            mlp_vals  = by_layer[L]["mlp"]

            # attn (left of layer center)
            x_a = L - pair_offset
            if attn_vals:
                jx = rng.uniform(-jitter_w, jitter_w, len(attn_vals))
                ax.scatter(x_a + jx, attn_vals,
                           c=attn_color, s=14, alpha=0.78, linewidths=0,
                           zorder=3)
                ax.hlines(np.mean(attn_vals), x_a - 0.18, x_a + 0.18,
                          color=mean_color, linewidth=1.4, zorder=5)

            # mlp (right of layer center)
            x_m = L + pair_offset
            if mlp_vals:
                jx = rng.uniform(-jitter_w, jitter_w, len(mlp_vals))
                ax.scatter(x_m + jx, mlp_vals,
                           c=mlp_color, s=18, alpha=0.85, linewidths=0,
                           marker="s", zorder=3)
                ax.hlines(np.mean(mlp_vals), x_m - 0.18, x_m + 0.18,
                          color=mean_color, linewidth=1.4, zorder=5)

        # divider lines between layers
        for L in layers[:-1]:
            ax.axvline(L + 0.5, color="#E0E0E0", linewidth=0.6, zorder=1)

        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{L}" for L in layers])
        # keep both axes at the same sizes for visual consistency
        AX_LABEL_FS = 12
        AX_TICK_FS  = 10
        ax.set_xlabel("Transformer layer  (attn left  $\\bullet$  MLP right)",
                      fontsize=AX_LABEL_FS)
        # short y-label; the formula goes in the caption
        ax.set_ylabel("Unit importance within block", fontsize=AX_LABEL_FS)
        ax.tick_params(axis="both", labelsize=AX_TICK_FS)

        if log_y:
            ax.set_yscale("log")
            # dense minor ticks with labels at 2 / 5 of each decade for readability
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=(2.0, 5.0), numticks=24))
            ax.yaxis.set_minor_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.tick_params(axis="y", which="minor", labelsize=AX_TICK_FS - 2,
                           labelcolor="#666666", length=2.0)

        ax.grid(axis="y", which="major", alpha=0.45, linewidth=0.6)
        ax.grid(axis="y", which="minor", alpha=0.18, linewidth=0.4)
        ax.set_axisbelow(True)

        ax.set_xlim(layers[0] - 0.6, layers[-1] + 0.6)

        # tighten y-range to the data so the spread is visible (no wasted space)
        all_vals = [v for L in layers for v in
                    by_layer[L]["attn"] + by_layer[L]["mlp"]]
        vmin, vmax = min(all_vals), max(all_vals)
        if log_y:
            ax.set_ylim(vmin / 1.6, vmax * 1.4)
        else:
            pad = 0.05 * (vmax - vmin)
            ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

        attn_h = mpatches.Patch(color=attn_color, label="Attention head")
        mlp_h  = mpatches.Patch(color=mlp_color,  label="MLP unit")
        mean_h = plt.Line2D([0], [0], color=mean_color, linewidth=1.4,
                            label="group mean")
        # legend in the upper left, which is empty in the data
        # (early layers, high values)
        ax.legend(handles=[attn_h, mlp_h, mean_h],
                  loc="upper left", bbox_to_anchor=(0.005, 0.995),
                  ncol=1, frameon=True, framealpha=0.9,
                  edgecolor="#CCCCCC",
                  handlelength=1.4, borderpad=0.6)

        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_png = plots_dir / out_name
        out_pdf = plots_dir / out_name.replace(".png", ".pdf")
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"[Plot] {out_png}")
        print(f"[Plot] {out_pdf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", nargs="?",
                   default="results/"
                           "gpt2_small_ioi_strong_N100_M5_ori")
    p.add_argument("--linear", action="store_true",
                   help="use linear y-axis (default: log)")
    args = p.parse_args()
    plot(Path(args.results_dir), log_y=not args.linear)
