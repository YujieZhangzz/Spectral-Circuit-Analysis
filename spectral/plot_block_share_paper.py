"""
Same strip plot as plot_block_dist_paper.py, but the y-axis is the unit
subspace share

    share(u) = || U_B[local_idx_u, :] ||_F^2  /  d_u

i.e. how much of the block's truncated SVD basis lies on unit u's parameters,
normalised by d_u. This is consistency_share importance without the
layer-consistency factor and without σ weighting. The model is loaded on CPU
only to rebuild unit_layout.

Usage:
    python spectral/plot_block_share_paper.py [results/<experiment_name>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paper_style import paper_rc, paper_size, TYPE_STYLE
from utils.model_utils import get_param_filter
from spectral.block_finder import build_unit_layout, detect_arch
from transformers import AutoModelForCausalLM


# Per-unit share computation

def compute_shares(results_dir: Path, model_name: str,
                   scope: str, n_layers: int):
    """Returns list of dicts: {layer, unit_type, label, share}."""
    print(f"[load] {model_name} on CPU (layout only)...")
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  torch_dtype=torch.float32)
    arch = detect_arch(model)
    param_filter = get_param_filter(scope, n_layers)
    unit_layout  = build_unit_layout(model, param_filter, arch=arch,
                                     include_mlp=True)
    del model
    uid_to_unit = {u.unit_id: u for u in unit_layout}

    import json
    with open(results_dir / "blocks.json") as f:
        blocks_meta = json.load(f)
    blocks_meta = [b for b in blocks_meta if "unit_ids" in b]

    rows = []
    for b in blocks_meta:
        bid     = b["block_id"]
        U_B     = torch.load(results_dir / f"U_rho_block{bid}.pt").numpy()
        global_idx = torch.load(results_dir / f"global_indices_block{bid}.pt").numpy()
        # global_idx[k] = the flat-grad index that occupies U_B row k
        glob2local = {int(g): k for k, g in enumerate(global_idx)}

        for uid in b["unit_ids"]:
            u = uid_to_unit[uid]
            unit_global = u.get_indices().numpy()
            try:
                local = np.fromiter((glob2local[int(g)] for g in unit_global),
                                    dtype=np.int64, count=len(unit_global))
            except KeyError as e:
                raise RuntimeError(
                    f"unit {u.label} (uid={uid}) has param idx {e} not in "
                    f"block {bid}'s global_indices (layout/blocks mismatch)"
                )
            d_u   = local.shape[0]
            slab  = U_B[local, :]
            share = float((slab ** 2).sum()) / d_u

            rows.append({
                "layer":     u.layer,
                "unit_type": u.unit_type,
                "label":     u.label,
                "share":     share,
                "block_id":  bid,
            })
    return rows


# Plotting (same layout as plot_block_dist_paper.py)

def plot(results_dir: Path,
         model_name: str = "gpt2",
         scope: str = "attn_VO_mlp",
         n_layers: int = 12,
         log_y: bool = True,
         out_name: str = "block_subspace_share_dist.png"):
    results_dir = Path(results_dir)

    rows = compute_shares(results_dir, model_name, scope, n_layers)

    by_layer = {}
    for r in rows:
        d = by_layer.setdefault(r["layer"], {"attn": [], "mlp": []})
        bucket = "attn" if r["unit_type"] == "attn_head" else "mlp"
        d[bucket].append(r["share"])
    layers = sorted(by_layer.keys())

    attn_color = TYPE_STYLE["attn"]["color"]
    mlp_color  = TYPE_STYLE["mlp"]["color"]
    mean_color = "#C0392B"

    pair_offset = 0.22
    jitter_w    = 0.14
    rng         = np.random.default_rng(0)

    with plt.rc_context(paper_rc("double")):
        fig_w, fig_h = paper_size("double", aspect=2.4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.4),
                               constrained_layout=True)

        for L in layers:
            attn_vals = by_layer[L]["attn"]
            mlp_vals  = by_layer[L]["mlp"]

            x_a = L - pair_offset
            if attn_vals:
                jx = rng.uniform(-jitter_w, jitter_w, len(attn_vals))
                ax.scatter(x_a + jx, attn_vals,
                           c=attn_color, s=14, alpha=0.78, linewidths=0,
                           zorder=3)
                ax.hlines(np.mean(attn_vals), x_a - 0.18, x_a + 0.18,
                          color=mean_color, linewidth=1.4, zorder=5)

            x_m = L + pair_offset
            if mlp_vals:
                jx = rng.uniform(-jitter_w, jitter_w, len(mlp_vals))
                ax.scatter(x_m + jx, mlp_vals,
                           c=mlp_color, s=18, alpha=0.85, linewidths=0,
                           marker="s", zorder=3)
                ax.hlines(np.mean(mlp_vals), x_m - 0.18, x_m + 0.18,
                          color=mean_color, linewidth=1.4, zorder=5)

        for L in layers[:-1]:
            ax.axvline(L + 0.5, color="#E0E0E0", linewidth=0.6, zorder=1)

        AX_LABEL_FS = 12
        AX_TICK_FS  = 10
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{L}" for L in layers])
        ax.set_xlabel("Transformer layer  (attn left  $\\bullet$  MLP right)",
                      fontsize=AX_LABEL_FS)
        ax.set_ylabel("Unit subspace share within block", fontsize=AX_LABEL_FS)
        ax.tick_params(axis="both", labelsize=AX_TICK_FS)

        if log_y:
            ax.set_yscale("log")
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
    p.add_argument("--model", default="gpt2")
    p.add_argument("--scope", default="attn_VO_mlp")
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--linear", action="store_true")
    args = p.parse_args()
    plot(Path(args.results_dir),
         model_name=args.model,
         scope=args.scope,
         n_layers=args.n_layers,
         log_y=not args.linear)
