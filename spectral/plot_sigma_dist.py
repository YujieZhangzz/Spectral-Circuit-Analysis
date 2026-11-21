"""
Per-block distribution of σ_k² / d_block.

Figure 1: scree curves (σ_k² / d_block vs. rank k); attn blocks in blues, MLP
          blocks in oranges, shade by layer.
Figure 2: total normalised variance per block and top-1 dominance σ_1² / Σ σ_k².

Usage: python spectral/plot_sigma_dist.py [results/<experiment_name>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def plot_sigma_distributions(
    results_dir: Path,
    n_layers: int | None = None,
) -> None:
    """
    Read blocks.json and sigma_block{k}.pt from results_dir and write
    sigma_scree_per_block.png and sigma_total_per_block.png to plots/.
    """
    results_dir = Path(results_dir)
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "blocks.json") as f:
        raw = json.load(f)
    blocks = [b for b in raw if "unit_ids" in b]
    n_blocks = len(blocks)
    if n_blocks == 0:
        print(f"[plot_sigma_dist] no blocks found in {results_dir}/blocks.json")
        return

    # Load sigma for each block
    sigmas = []
    for b in blocks:
        k   = b["block_id"]
        sig = torch.load(results_dir / f"sigma_block{k}.pt",
                         weights_only=True).float().numpy()
        d   = b["d_k"]
        sigmas.append(sig ** 2 / d)          # (rho_B,) normalised variance

    is_attn = [b["unit_types"] == ["attn_head"] for b in blocks]

    # infer n_layers
    if n_layers is None:
        n_layers = max(max(b["layers"]) for b in blocks) + 1

    # Figure 1: scree curves (double-column width)
    from utils.paper_style import paper_size
    fig1, ax1 = plt.subplots(figsize=paper_size("double", aspect=2.4))
    attn_cmap = cm.Blues
    mlp_cmap  = cm.Oranges

    for idx, (b, sv) in enumerate(zip(blocks, sigmas)):
        layer = b["layers"][0]
        frac  = 0.3 + 0.7 * (layer / max(n_layers - 1, 1))
        color = attn_cmap(frac) if is_attn[idx] else mlp_cmap(frac)
        ax1.plot(np.arange(1, len(sv) + 1), sv,
                 color=color, linewidth=0.8, alpha=0.85)

    legend_handles = [
        Line2D([0], [0], color=attn_cmap(0.7), lw=2, label="attn block (early->dark)"),
        Line2D([0], [0], color=mlp_cmap(0.7),  lw=2, label="MLP block  (early->dark)"),
    ]
    ax1.legend(handles=legend_handles, fontsize=9)
    ax1.set_xlabel("rank k (principal direction)")
    ax1.set_ylabel("σ_k² / d_block  (normalized variance)")
    ax1.set_title("Scree curves: σ_k² / d_block per block\n"
                  "(each line = one block, color depth = layer index)")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3)

    fig1.tight_layout()
    out1 = plots_dir / "sigma_scree_per_block.png"
    fig1.savefig(out1, dpi=250)
    plt.close(fig1)
    print(f"[Plot] {out1}")

    # Figure 2: total and top-1 normalised variance per block
    totals = [sv.sum() for sv in sigmas]
    top1s  = [sv[0]   for sv in sigmas]
    rhos   = [len(sv) for sv in sigmas]
    bid_labels = [
        f"L{b['layers'][0]}\n{'attn' if is_attn[i] else 'MLP'}"
        for i, b in enumerate(blocks)
    ]

    x = np.arange(n_blocks)
    bar_w = 0.4

    fig2, axes = plt.subplots(2, 1, figsize=paper_size("double", aspect=1.8),
                              sharex=True)
    colors = ["#4c8fd6" if a else "#e8834f" for a in is_attn]
    axes[0].bar(x, totals, width=bar_w, color=colors, alpha=0.8)
    axes[0].set_ylabel("Σ σ_k² / d_block  (total norm. var.)")
    axes[0].set_title("Total normalized variance per block  (Σ σ_k² / d_block)")
    axes[0].grid(axis="y", alpha=0.3)
    for i, (v, r) in enumerate(zip(totals, rhos)):
        axes[0].text(i, v * 1.02, f"ρ={r}", ha="center", va="bottom", fontsize=6)

    dominance = [t1 / (tot + 1e-20) for t1, tot in zip(top1s, totals)]
    axes[1].bar(x, dominance, width=bar_w, color=colors, alpha=0.8)
    axes[1].set_ylabel("σ_1² / Σ σ_k²  (top-1 dominance)")
    axes[1].set_title("Top-1 direction dominance per block  (σ_1² / total)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bid_labels, fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)

    attn_p = mpatches.Patch(color="#4c8fd6", label="attn block")
    mlp_p  = mpatches.Patch(color="#e8834f", label="MLP block")
    axes[0].legend(handles=[attn_p, mlp_p], fontsize=9)

    fig2.tight_layout()
    out2 = plots_dir / "sigma_total_per_block.png"
    fig2.savefig(out2, dpi=250)
    plt.close(fig2)
    print(f"[Plot] {out2}")


if __name__ == "__main__":
    default_dir = "results/gpt2_small_ioi_strong_N50_M10"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_dir)
    plot_sigma_distributions(target)
