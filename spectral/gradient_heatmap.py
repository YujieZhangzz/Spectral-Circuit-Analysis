"""
Per-layer gradient heatmaps over all paraphrases, shape (NM, n_layers):
  A. per-layer gradient L2 norm (log scale)
  B. per-layer cosine similarity of each gradient to the layer-mean direction

Called from run.py (run_gradient_heatmap) or standalone:
    python spectral/gradient_heatmap.py --config configs/gpt2_small_ioi_strong.yaml
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils.io_utils    import load_config, get_output_dir
from utils.model_utils import load_model_and_tokenizer, get_param_filter, obtain_gradients
from data.loader       import (
    load_records, load_records_cap, load_records_gp,
    load_records_ioi_v3, load_records_ioi_strong, load_records_ioi_sampled,
    make_batch,
)
from spectral.block_finder import detect_arch
from utils.paper_style import (
    PAPER_PRESETS, TYPE_STYLE,
    BG_AXES, GRID_COLOR,
    paper_rc, paper_size, get_preset,
)


# Style for the split (attn vs MLP) consistency plot

DEFAULT_SPLIT_PRESET = "single"

# Error band: "sem" | "std" | "iqr" | "none"
ERROR_KIND  = "sem"
ERROR_MULT  = 1.96      # 1.96·SEM ≈ 95% CI
Y_PAD_FRAC  = 0.18      # vertical padding as a fraction of the data range
BAND_ALPHA  = 0.18


def _compute_band(arr: np.ndarray) -> tuple:
    """For a single (NM,) array, return (mean, low, high) under ERROR_KIND."""
    mean = float(arr.mean())
    if ERROR_KIND == "iqr":
        return mean, float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    if ERROR_KIND == "std":
        e = ERROR_MULT * float(arr.std())
        return mean, mean - e, mean + e
    if ERROR_KIND == "sem":
        e = ERROR_MULT * float(arr.std()) / float(np.sqrt(len(arr)))
        return mean, mean - e, mean + e
    return mean, mean, mean   # "none"


def plot_layer_consistency_split(
    cosim_split: dict,                # {(layer, type) -> (NM,) ndarray}
    layers:      List[int],
    save_dir:    str,
    preset:      str = DEFAULT_SPLIT_PRESET,
) -> None:
    """Paper-quality split (attn vs MLP) consistency plot.

    Writes:
      <save_dir>/layer_consistency_split.pdf
      <save_dir>/layer_consistency_split.png   (300 dpi)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg     = get_preset(preset)
    layers  = sorted(layers)
    types   = sorted({k[1] for k in cosim_split},
                     key=lambda t: list(TYPE_STYLE).index(t)
                     if t in TYPE_STYLE else 99)

    with plt.rc_context(paper_rc(preset)):
        fig, ax = plt.subplots(figsize=paper_size(preset, aspect=1.52))
        fig.patch.set_facecolor("white")
        ax.set_facecolor(BG_AXES)

        all_low, all_high = [], []
        for t in types:
            style = TYPE_STYLE.get(t,
                                   {"color": "gray", "label": t, "marker": "o"})
            color = style["color"]
            ks    = sorted(k for k in cosim_split if k[1] == t)
            ls    = [k[0] for k in ks]
            tri   = [_compute_band(cosim_split[k]) for k in ks]
            means = np.array([m for m, _, _ in tri])
            low   = np.array([l for _, l, _ in tri])
            high  = np.array([h for _, _, h in tri])

            if ERROR_KIND != "none":
                ax.fill_between(ls, low, high,
                                alpha=BAND_ALPHA, color=color, linewidth=0)
            ax.plot(ls, means,
                    color=color,
                    linewidth=cfg["linewidth"],
                    marker=style["marker"],
                    markersize=cfg["marker_sz"],
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label=style["label"],
                    zorder=3)
            all_low.append(low); all_high.append(high)

        # Tight y-limits so that layer-to-layer variation is visible
        data_lo = float(np.concatenate(all_low).min())
        data_hi = float(np.concatenate(all_high).max())
        rng     = max(data_hi - data_lo, 1e-3)
        pad     = rng * Y_PAD_FRAC
        ax.set_ylim(max(-1.0, data_lo - pad),
                    min( 1.0, data_hi + pad))

        ax.set_xlabel("Layer",
                      fontsize=cfg["label_fs"], labelpad=3, fontweight="bold")
        ax.set_ylabel("Cosine sim. to type-mean",
                      fontsize=cfg["label_fs"], labelpad=4, fontweight="bold")

        ax.set_xticks(layers)
        ax.set_xticklabels([str(l) for l in layers], fontsize=cfg["tick_fs"])
        ax.tick_params(axis="y", labelsize=cfg["tick_fs"])

        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        if ax.get_ylim()[0] < 0 < ax.get_ylim()[1]:
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5,
                       alpha=0.6, zorder=1)

        ax.legend(loc="lower right", fontsize=cfg["legend_fs"],
                  ncol=len(types), handlelength=1.4,
                  columnspacing=1.0, borderaxespad=0.4)

        plt.tight_layout(pad=0.4)
        out_pdf = Path(save_dir) / "layer_consistency_split.pdf"
        out_png = Path(save_dir) / "layer_consistency_split.png"
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.savefig(out_png, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Plot] {out_pdf}")
        print(f"[Plot] {out_png}")


# Layer index map

def build_layer_index_map(
    model,
    param_filter: Callable[[str], bool],
    n_layers:     int,
    arch_name:    str,
    split_by_type: bool = False,
) -> Dict:
    """
    Return {key: indices_in_flat_grad}.

    split_by_type=False (default): key = layer (attn and MLP together).
    split_by_type=True:  key = (layer, "attn"|"mlp"), for separate
    attn / MLP consistency.
    """
    import re

    if arch_name == "gpt2":
        layer_pat = r"transformer\.h\.(\d+)\."
    elif arch_name == "pythia":
        layer_pat = r"gpt_neox\.layers\.(\d+)\."
    else:
        layer_pat = r"model\.layers\.(\d+)\."

    raw: Dict = {}
    offset = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if not param_filter(name):
            continue
        m = re.search(layer_pat, name)
        if m:
            layer = int(m.group(1))
            if split_by_type:
                if "attn" in name or "attention" in name:
                    key = (layer, "attn")
                elif "mlp" in name or "ffn" in name:
                    key = (layer, "mlp")
                else:
                    key = (layer, "other")
            else:
                key = layer
            raw.setdefault(key, []).extend(range(offset, offset + n))
        offset += n

    result = {k: torch.tensor(v, dtype=torch.long)
              for k, v in raw.items() if v}

    if split_by_type:
        print(f"[LayerIndex] split_by_type=True: {len(result)} groups")
        for k in sorted(result):
            print(f"  L{k[0]:2d} {k[1]:5s}: {len(result[k]):>10,} params")
    else:
        print(f"[LayerIndex] {len(result)} layers with filtered params")
        for l, idx in result.items():
            print(f"  Layer {l:2d}: {len(idx):>10,} params in flat grad")

    return result


# Gradient collection (per-layer statistics only)

def collect_layer_gradients(
    model:          PreTrainedModel,
    tokenizer:      PreTrainedTokenizer,
    records:        List[dict],
    param_filter:   Callable[[str], bool],
    layer_idx_map:  Dict,
    cfg:            dict,
    split_idx_map:  Dict = None,
    loss_fn:        Optional[Callable] = None,
) -> tuple:
    """
    Compute the gradient of every paraphrase and extract per-layer norms and
    cosine similarities to the layer mean.

    split_idx_map=None (default):
        Returns: (norm_matrix, cosim_matrix, labels, layers)

    split_idx_map={(layer, "attn"|"mlp"): tensor, ...}:
        Additionally compute attn / MLP cosine similarities in the same pass.
        Returns: (norm_matrix, cosim_matrix, labels, layers, cosim_split)
            cosim_split: dict[(layer, type) -> (NM,) ndarray]
    """
    N        = len(records)
    M        = len(records[0]["paraphrase_queries"])
    NM       = N * M
    n_layers = len(layer_idx_map)
    layers   = sorted(layer_idx_map.keys())
    device   = cfg["model"]["device"]
    max_len  = cfg["data"]["max_seq_len"]

    norm_matrix = np.zeros((NM, n_layers), dtype=np.float32)

    # Per-layer norms go into norm_matrix; per-layer unit-norm directions are
    # kept in float16 and compared to the layer mean afterwards.
    layer_mean: Dict[int, torch.Tensor] = {l: None for l in layers}
    layer_count = 0

    layer_grads_norm: Dict[int, List[float]] = {l: [] for l in layers}
    layer_grads_dir: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

    # split mode: also collect (layer, type) directions in the same pass
    split_grads_dir: Dict = {k: [] for k in split_idx_map} if split_idx_map else {}

    labels = []
    idx    = 0

    for i, rec in enumerate(tqdm(records, desc="Collecting layer grads")):
        extras_list = rec.get("loss_extras_per_paraphrase")
        for j, (q, c) in enumerate(zip(
            rec["paraphrase_queries"],
            rec["paraphrase_completions"],
        )):
            batch  = make_batch(q, c, tokenizer, max_len, device)
            extras = extras_list[j] if extras_list is not None else None
            g      = obtain_gradients(
                model, batch, param_filter,
                loss_fn=loss_fn, loss_extras=extras,
            )                                                       # (D,) cpu float32

            for li, l in enumerate(layers):
                g_l   = g[layer_idx_map[l]]                        # (d_l,)
                norm  = g_l.norm().item()
                norm_matrix[idx, li] = norm
                # store the normalised direction in float16
                g_dir = F.normalize(g_l.unsqueeze(0), p=2, dim=1).squeeze(0)
                layer_grads_dir[l].append(g_dir.to(torch.float16))

            # split mode: (layer, type) directions
            if split_idx_map:
                for k, idx_t in split_idx_map.items():
                    g_s   = g[idx_t]
                    g_dir = F.normalize(g_s.unsqueeze(0), p=2, dim=1).squeeze(0)
                    split_grads_dir[k].append(g_dir.to(torch.float16))

            labels.append(f"N{i:02d}_M{j}")
            del g
            idx += 1

    # Cosine sim to the layer mean (attn + MLP together)
    print("\n[Heatmap] Computing cosine similarities to layer mean...")
    cosim_matrix = np.zeros((NM, n_layers), dtype=np.float32)

    for li, l in enumerate(layers):
        G_l    = torch.stack(layer_grads_dir[l], dim=0).float()  # (NM, d_l)
        # mean direction
        mean_l = G_l.mean(dim=0)
        mean_l = F.normalize(mean_l.unsqueeze(0), p=2, dim=1).squeeze(0)
        # rows are already normalised, so a dot product gives cosine sim
        cosim  = (G_l @ mean_l).numpy()   # (NM,)
        cosim_matrix[:, li] = cosim
        del G_l

    if not split_idx_map:
        return norm_matrix, cosim_matrix, labels, layers

    # split mode: cosine sim per (layer, type)
    print("[Heatmap] Computing split cosine similarities (attn | mlp)...")
    cosim_split = {}
    for k, dirs in split_grads_dir.items():
        G_s    = torch.stack(dirs, dim=0).float()           # (NM, d_s)
        mean_s = F.normalize(G_s.mean(dim=0, keepdim=True), p=2, dim=1)
        cosim_split[k] = (G_s @ mean_s.T).squeeze(1).numpy()   # (NM,)
        del G_s

    return norm_matrix, cosim_matrix, labels, layers, cosim_split


# Plotting

def plot_gradient_heatmap(
    norm_matrix:  np.ndarray,    # (NM, n_layers)
    cosim_matrix: np.ndarray,    # (NM, n_layers)
    labels:       List[str],
    layers:       List[int],
    N:            int,
    M:            int,
    save_dir:     str,
    cosim_split:  dict = None,   # optional: {(layer, type) -> (NM,) array}
    split_preset: str  = DEFAULT_SPLIT_PRESET,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[Heatmap] matplotlib not available"); return

    NM = len(labels)

    # double-column figure (7 in wide), height scales with NM
    height = max(2.8, min(NM * 0.04 + 1.2, 6.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, height))
    fig.suptitle(
        f"Gradient Heatmap  ({NM} paraphrases × {len(layers)} layers)\n"
        f"N={N} semantic instances, M={M} paraphrases each",
        fontsize=11
    )

    # Panel 1: grad norm (log scale)
    log_norm = np.log1p(norm_matrix)   # log(1 + norm) to compress the range

    im1 = ax1.imshow(log_norm, aspect="auto", cmap="YlOrRd",
                     interpolation="nearest")
    plt.colorbar(im1, ax=ax1, label="log(1 + grad norm)")
    ax1.set_title("Gradient Norm per Layer\n(brighter = stronger activation)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Paraphrase instance")
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"L{l}" for l in layers], fontsize=8)

    # separator every M rows (one block per instance)
    for ni in range(1, N):
        ax1.axhline(ni * M - 0.5, color="white", linewidth=0.8, alpha=0.6)

    # y ticks: instance ids
    instance_ticks = [i * M + M // 2 for i in range(N)]
    ax1.set_yticks(instance_ticks)
    ax1.set_yticklabels([f"N{i:02d}" for i in range(N)], fontsize=7)

    # Panel 2: cosine sim to layer mean
    im2 = ax2.imshow(cosim_matrix, aspect="auto", cmap="RdBu_r",
                     vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im2, ax=ax2, label="cosine sim to layer mean")
    ax2.set_title("Cosine Sim to Layer Mean\n"
                  "(red = consistent, blue = divergent)")
    ax2.set_xlabel("Layer")
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f"L{l}" for l in layers], fontsize=8)

    for ni in range(1, N):
        ax2.axhline(ni * M - 0.5, color="white", linewidth=0.8, alpha=0.6)

    ax2.set_yticks(instance_ticks)
    ax2.set_yticklabels([f"N{i:02d}" for i in range(N)], fontsize=7)

    plt.tight_layout()
    out = Path(save_dir) / "gradient_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")

    # Per-layer mean cosine sim (line plot)
    _plot_layer_consistency(cosim_matrix, layers, save_dir,
                            cosim_split=cosim_split, split_preset=split_preset)


def _plot_layer_consistency(
    cosim_matrix: np.ndarray,
    layers:       List[int],
    save_dir:     str,
    cosim_split:  dict = None,   # optional: {(layer, type) -> (NM,) array}
    split_preset: str  = DEFAULT_SPLIT_PRESET,
) -> None:
    """
    Plot the mean and std of cosine sim over all paraphrases for each layer.

    cosim_split:  if given, also draw the attn / MLP split plot
                  via plot_layer_consistency_split() (PDF + PNG).
    split_preset: figure size preset ("single" | "double").
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_per_layer = cosim_matrix.mean(axis=0)   # (n_layers,)
    std_per_layer  = cosim_matrix.std(axis=0)    # (n_layers,)

    # single-column figure
    _, ax = plt.subplots(figsize=paper_size(split_preset, aspect=1.5))
    ax.plot(layers, mean_per_layer, "o-", color="#4f86c6",
            linewidth=1.6, markersize=4, markeredgecolor="white",
            markeredgewidth=0.9, label="mean")
    ax.fill_between(layers,
                    mean_per_layer - std_per_layer,
                    mean_per_layer + std_per_layer,
                    alpha=0.18, color="#4f86c6", label="±1 std")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine sim. to layer mean")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "layer_consistency.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out}")

    # attn vs MLP consistency plot
    if cosim_split:
        plot_layer_consistency_split(
            cosim_split, layers, save_dir, preset=split_preset,
        )


# Pipeline entry point (model / tokenizer / records already loaded)

def run_gradient_heatmap(
    model:             PreTrainedModel,
    tokenizer:         PreTrainedTokenizer,
    records:           List[dict],
    param_filter:      Callable[[str], bool],
    cfg:               dict,
    n_layers:          int,
    arch_name:         str,
    save_dir:          str,
    split_consistency: bool = True,
    split_preset:      str  = DEFAULT_SPLIT_PRESET,
    loss_fn:           Optional[Callable] = None,
) -> None:
    """
    Collect per-layer gradients and plot them (called from run.py).

    Plots (save_dir, default <outdir>/plots/):
      - gradient_heatmap.png              (norm + cosine panels, NM x n_layers)
      - layer_consistency.png             (per-layer mean cosine sim)
      - layer_consistency_split.{pdf,png} (attn vs MLP; if split_consistency)

    Data cache (parent of save_dir, next to plots/):
      - gradient_heatmap_data.npz         (used by --replot)
      - gradient_heatmap_summary.csv      (per-layer mean/std/quartiles)

    This re-runs forward+backward per paraphrase (cost similar to Phase 0),
    so it is enabled only via cfg["heatmap"]["enabled"] = true.
    """
    N  = len(records)
    M  = len(records[0]["paraphrase_queries"])
    print(f"[Heatmap] N={N}, M={M}, NM={N*M}, "
          f"split_consistency={split_consistency}")

    layer_idx_map = build_layer_index_map(model, param_filter, n_layers, arch_name)
    split_idx_map = None
    if split_consistency:
        split_idx_map = build_layer_index_map(
            model, param_filter, n_layers, arch_name, split_by_type=True)

    result = collect_layer_gradients(
        model, tokenizer, records, param_filter, layer_idx_map, cfg,
        split_idx_map=split_idx_map,
        loss_fn=loss_fn,
    )
    if split_consistency:
        norm_matrix, cosim_matrix, labels, layers, cosim_split = result
    else:
        norm_matrix, cosim_matrix, labels, layers = result
        cosim_split = None

    print(f"\n[Heatmap] norm_matrix={norm_matrix.shape}  "
          f"cosim_matrix={cosim_matrix.shape}")
    print("  Layer-wise mean cosine sim:")
    for li, l in enumerate(layers):
        print(f"    L{l:2d}: mean={cosim_matrix[:, li].mean():.3f}  "
              f"std={cosim_matrix[:, li].std():.3f}")

    save_heatmap_data(
        norm_matrix, cosim_matrix, labels, layers, N, M, save_dir,
        cosim_split=cosim_split,
    )

    plot_gradient_heatmap(
        norm_matrix, cosim_matrix, labels, layers, N, M, save_dir,
        cosim_split=cosim_split, split_preset=split_preset,
    )


# Data cache: save / load / replot

DATA_FILENAME    = "gradient_heatmap_data.npz"
SUMMARY_FILENAME = "gradient_heatmap_summary.csv"


def save_heatmap_data(
    norm_matrix:  np.ndarray,
    cosim_matrix: np.ndarray,
    labels:       List[str],
    layers:       List[int],
    N:            int,
    M:            int,
    save_dir:     str,
    cosim_split:  dict = None,
) -> None:
    """Save everything needed for plotting so gradients need not be recomputed.

    Files go to the parent of save_dir (<outdir>/, next to plots/):
      - <outdir>/gradient_heatmap_data.npz     (loaded by --replot)
      - <outdir>/gradient_heatmap_summary.csv  (per-layer mean/std/quartiles)
    """
    data_dir = Path(save_dir).parent
    out = data_dir / DATA_FILENAME
    data_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "norm_matrix":  norm_matrix,
        "cosim_matrix": cosim_matrix,
        "labels":       np.array(labels),
        "layers":       np.array(layers, dtype=np.int32),
        "N":            np.int64(N),
        "M":            np.int64(M),
    }
    if cosim_split:
        for (layer, typ), arr in cosim_split.items():
            bundle[f"split_L{layer:02d}_{typ}"] = arr

    np.savez_compressed(out, **bundle)
    print(f"[Save] {out}  ({out.stat().st_size/1024:.1f} KB)")

    # Small human-readable CSV summary
    rows = ["layer,type,mean,std,q25,q50,q75,n"]
    for li, l in enumerate(layers):
        v = cosim_matrix[:, li]
        rows.append(
            f"{l},mixed,{v.mean():.4f},{v.std():.4f},"
            f"{np.percentile(v,25):.4f},{np.percentile(v,50):.4f},"
            f"{np.percentile(v,75):.4f},{len(v)}"
        )
    if cosim_split:
        for (l, typ), arr in sorted(cosim_split.items()):
            rows.append(
                f"{l},{typ},{arr.mean():.4f},{arr.std():.4f},"
                f"{np.percentile(arr,25):.4f},{np.percentile(arr,50):.4f},"
                f"{np.percentile(arr,75):.4f},{len(arr)}"
            )
    csv_out = data_dir / SUMMARY_FILENAME
    csv_out.write_text("\n".join(rows) + "\n")
    print(f"[Save] {csv_out}")


def load_heatmap_data(path: str) -> tuple:
    """Load the .npz written by save_heatmap_data.

    `path` may be:
      - the .npz file itself
      - the directory containing it (usually <outdir>/)
      - the plots/ directory next to it (save_dir in run_gradient_heatmap)
    """
    pp = Path(path)
    if pp.is_file() and pp.suffix == ".npz":
        p = pp
    elif pp.is_dir() and (pp / DATA_FILENAME).exists():
        p = pp / DATA_FILENAME
    else:
        p = pp.parent / DATA_FILENAME           # path is the plots/ directory
    if not p.exists():
        raise FileNotFoundError(
            f"{DATA_FILENAME} not found (tried {pp/DATA_FILENAME} and {pp.parent/DATA_FILENAME}); "
            "run gradient_heatmap once without --replot to collect gradients.")
    z = np.load(p, allow_pickle=False)

    norm_matrix  = z["norm_matrix"]
    cosim_matrix = z["cosim_matrix"]
    labels       = [str(x) for x in z["labels"]]
    layers       = [int(x) for x in z["layers"]]
    N            = int(z["N"])
    M            = int(z["M"])

    cosim_split = {}
    for k in z.files:
        if k.startswith("split_L"):
            # split_L00_attn -> (0, "attn")
            _, layer_part, typ = k.split("_")
            cosim_split[(int(layer_part[1:]), typ)] = z[k]

    print(f"[Load] {p}  NM={len(labels)}  layers={len(layers)}  "
          f"split_keys={len(cosim_split)}")
    return (norm_matrix, cosim_matrix, labels, layers, N, M,
            cosim_split or None)


def replot_from_cache(save_dir: str,
                      split_preset: str = DEFAULT_SPLIT_PRESET) -> None:
    """Redraw the plots from the .npz cache without loading the model."""
    (norm_matrix, cosim_matrix, labels, layers,
     N, M, cosim_split) = load_heatmap_data(save_dir)
    plot_gradient_heatmap(
        norm_matrix, cosim_matrix, labels, layers, N, M, save_dir,
        cosim_split=cosim_split, split_preset=split_preset,
    )


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split_consistency", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compute gradient consistency separately for attn and MLP units. "
                        "Saves an additional layer_consistency_split.png. "
                        "Use --no-split_consistency to disable.")
    p.add_argument("--replot", action="store_true",
                   help=f"Skip gradient collection and replot from <outdir>/{DATA_FILENAME} "
                        "(the parent of plots/).")
    p.add_argument("--save-dir", default=None,
                   help="Override save_dir (default <outdir>/plots). "
                        "With --replot, the .npz is looked up relative to this path.")
    p.add_argument("--split_preset", default=DEFAULT_SPLIT_PRESET,
                   choices=list(PAPER_PRESETS.keys()),
                   help="Figure size preset: single (3.5 in) | double (7 in).")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    outdir = get_output_dir(cfg)
    save_dir = args.save_dir or str(outdir / "plots")

    print(f"\n{'='*60}")
    print(f"  Gradient Heatmap Visualization")
    print(f"  config: {args.config}")
    print(f"  mode:   {'replot (from cache)' if args.replot else 'collect + plot'}")
    print(f"{'='*60}\n")

    if args.replot:
        replot_from_cache(save_dir, split_preset=args.split_preset)
        print(f"\nDone. Outputs saved to {save_dir}/")
        return

    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name_or_path"],
        cfg["model"]["torch_dtype"],
        cfg["model"]["device"],
    )
    arch     = detect_arch(model)
    n_layers = getattr(model.config, "num_hidden_layers",
                       getattr(model.config, "n_layer", None))
    scope    = cfg["params"]["scope"]
    pf       = get_param_filter(scope, n_layers)

    data_format = cfg["data"].get("format", "ioi")
    _M = cfg["data"].get("M", None)
    if data_format == "cap":
        records = load_records_cap(cfg["data"]["path"], cfg["data"]["source"], cfg["data"]["N"])
    elif data_format == "gp":
        records = load_records_gp(cfg["data"]["path"], cfg["data"]["source"], cfg["data"]["N"])
    elif data_format == "ioi_v3":
        records = load_records_ioi_v3(cfg["data"]["path"], N=cfg["data"]["N"], M=_M)
    elif data_format == "ioi_strong":
        records = load_records_ioi_strong(cfg["data"]["path"], N=cfg["data"]["N"], M=_M)
    elif data_format == "ioi_sampled":
        records = load_records_ioi_sampled(cfg["data"]["path"], N=cfg["data"]["N"], M=_M)
    else:
        records = load_records(cfg["data"]["path"], cfg["data"]["source"], cfg["data"]["N"])

    run_gradient_heatmap(
        model, tokenizer, records, pf, cfg,
        n_layers=n_layers, arch_name=arch.name, save_dir=save_dir,
        split_consistency=args.split_consistency,
        split_preset=args.split_preset,
    )
    print(f"\nDone. Outputs saved to {save_dir}/")


if __name__ == "__main__":
    main()