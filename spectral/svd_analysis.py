"""
Phase 4: per-block SVD of the gradient matrix via the Gram trick.

    G_B ∈ R^{d_k × N},  K_B = G_B.T @ G_B ∈ R^{N × N}
    eigh(K_B) = (λ, V),  U_rho = G_B @ V_rho @ diag(1/σ_rho)

Effective rank: ρ = min{k : Σ_{i=1}^k σ_i² / Σ σ_j² ≥ η}
"""

from __future__ import annotations
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from data.loader import make_batch
from utils.model_utils import obtain_gradients
from spectral.block_finder import BlockResult, UnitInfo


# Result container

@dataclass
class BlockSVDResult:
    block_id:         int
    unit_ids:         List[int]         # unit_ids in the block
    unit_labels:      List[str]         # e.g. "L3H5", "L4_MLP_gate"
    unit_types:       List[str]         # unit type of each unit
    layers:           List[int]         # layers covered by the block

    d_k:              int               # block parameter dimension
    N:                int               # number of instances
    singular_values:  torch.Tensor      # (min(d_k,N),)  descending
    explained_var:    torch.Tensor      # (N,)  cumulative explained variance
    effective_rank:   int               # ρ
    eta:              float
    U_rho:            torch.Tensor      # (d_k, ρ)  subspace basis
    redundancy_ratio: float             # 1 - ρ/N
    global_indices:   torch.Tensor      # (d_k,) LongTensor: flat-grad index of
                                        # each row of U_rho; used to map rows
                                        # back to units.

    def summary(self) -> str:
        return (
            f"  Block {self.block_id}: "
            f"d_k={self.d_k:,}, N={self.N}, "
            f"ρ={self.effective_rank}/{len(self.singular_values)} "
            f"(η={self.eta}, redundancy={self.redundancy_ratio:.1%})\n"
            f"    layers={self.layers}, types={sorted(set(self.unit_types))}"
        )


# Building G_B: fast path (grad_avg) or slow path (backward)

def build_augmented_grad_matrix(
    block_indices: torch.Tensor,          # (d_k,) LongTensor, block positions in the flat grad
    records:       List[dict],
    cfg:           dict,
    # Fast path: grad_avg already in memory
    grad_avg:      Optional[torch.Tensor] = None,   # (N, D) float16
    # Slow path: recompute gradients
    model:         Optional[PreTrainedModel]     = None,
    tokenizer:     Optional[PreTrainedTokenizer] = None,
    param_filter:  Optional[Callable]            = None,
    loss_fn:       Optional[Callable]            = None,
) -> torch.Tensor:
    """
    Build G_B ∈ R^{d_k × N}, column i = ḡ_i|_B = (1/M) Σ_j g_{i,j,B}.

    Fast path (grad_avg given): slice grad_avg[:, block_indices], no backward.
    Slow path (grad_avg=None):  recompute the gradient for every (i, j).
    """
    N   = len(records)
    M   = len(records[0]["paraphrase_queries"])
    d_k = len(block_indices)

    # Fast path
    if grad_avg is not None:
        print(f"  [G_B] Fast path via grad_avg  "
              f"(d_k={d_k:,}, N={N}, skipping {M*N} backward passes)")
        G_B = grad_avg[:, block_indices].float().T.contiguous()  # (d_k, N)
        return G_B

    # Slow path
    assert model is not None and tokenizer is not None and param_filter is not None, \
        "Slow path requires model, tokenizer, param_filter"

    device  = cfg["model"]["device"]
    max_len = cfg["data"]["max_seq_len"]

    print(f"  [G_B] Slow path (backward)  "
          f"(d_k={d_k:,}, N={N}, M={M}, total={M*N} passes)")

    G_B = torch.zeros(d_k, N, dtype=torch.float32)

    for i, rec in enumerate(tqdm(records,
                                  desc=f"  Building G_B (d_k={d_k:,})",
                                  leave=False)):
        g_sum = torch.zeros(d_k, dtype=torch.float32)
        extras_list = rec.get("loss_extras_per_paraphrase")
        for j, (q, c) in enumerate(zip(rec["paraphrase_queries"],
                                        rec["paraphrase_completions"])):
            batch  = make_batch(q, c, tokenizer, max_len, device)
            extras = extras_list[j] if extras_list is not None else None
            g      = obtain_gradients(
                model, batch, param_filter,
                loss_fn=loss_fn, loss_extras=extras,
            )                                                   # (D,)
            g_sum += g[block_indices]
            del g

        G_B[:, i] = g_sum / M

    return G_B   # (d_k, N)


# Gram-trick SVD

def gram_svd(
    G_B:         torch.Tensor,    # (d_k, N)
    eta:         float = 0.95,
    chunk:       int   = 5_000_000,
    mean_center: bool  = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    SVD of G_B via the N x N Gram matrix.

    Args:
        G_B:         block gradient matrix, columns = per-instance gradients
        eta:         cumulative-variance threshold for effective rank
        chunk:       row-chunk size for the (d_k, d_k)-avoiding products
        mean_center: if True, subtract cross-instance mean μ = G_B.mean(dim=1)
                     before SVD. The resulting U_rho spans the column space of
                     G_B − μ, i.e., the *deviations* from the shared task
                     direction. importance ‖U_rho[u, :]‖² then measures how
                     much each unit contributes to instance-level differences,
                     not to the common task signal — needed when the common
                     signal dominates (e.g. greater-than), where the ranking
                     would otherwise reduce to the Magnitude baseline.

    Returns:
        sigma    (N,)      singular values, descending (of the centred matrix if mean_center)
        U_rho    (d_k, ρ)  subspace basis
        rho      int       effective rank
        cumvar   (N,)      cumulative explained variance
    """
    d_k, N = G_B.shape

    mu = G_B.mean(dim=1, keepdim=True) if mean_center else None   # (d_k, 1) or None

    # K_B = G_B_c.T @ G_B_c, accumulated in row chunks
    K_B = torch.zeros(N, N, dtype=torch.float32)
    for s in range(0, d_k, chunk):
        e    = min(s + chunk, d_k)
        G_c  = G_B[s:e, :]                       # (chunk, N)
        if mu is not None:
            G_c = G_c - mu[s:e]                  # broadcast subtract
        K_B += G_c.T @ G_c                       # (N, N)

    # eigh: K_B is symmetric PSD
    eigenvalues, V = torch.linalg.eigh(K_B)   # ascending
    idx         = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    V           = V[:, idx]                    # (N, N), descending

    sigma_sq  = eigenvalues.clamp(min=0)
    sigma     = torch.sqrt(sigma_sq)           # (N,)
    total_var = sigma_sq.sum() + 1e-12
    cumvar    = torch.cumsum(sigma_sq, dim=0) / total_var

    # effective rank ρ
    rho_hits = (cumvar >= eta).nonzero(as_tuple=True)[0]
    rho      = int(rho_hits[0].item()) + 1 if len(rho_hits) > 0 else N

    # U_rho = G_B_c @ V_rho @ diag(1/σ_rho), in row chunks
    V_rho   = V[:, :rho]                            # (N, ρ)
    sigma_r = sigma[:rho].clamp(min=1e-8)           # (ρ,)
    U_rho   = torch.zeros(d_k, rho, dtype=torch.float32)
    for s in range(0, d_k, chunk):
        e   = min(s + chunk, d_k)
        G_c = G_B[s:e, :]
        if mu is not None:
            G_c = G_c - mu[s:e]
        U_rho[s:e, :] = (G_c @ V_rho) / sigma_r.unsqueeze(0)

    return sigma, U_rho, rho, cumvar


# Phase 4 entry point

def run_block_svd(
    block_result: BlockResult,
    records:      List[dict],
    cfg:          dict,
    # Fast path
    grad_avg:     Optional[torch.Tensor] = None,   # (N, D) float16
    # Slow path fallback
    model:        Optional[PreTrainedModel]     = None,
    tokenizer:    Optional[PreTrainedTokenizer] = None,
    param_filter: Optional[Callable]            = None,
    loss_fn:      Optional[Callable]            = None,
) -> List[BlockSVDResult]:
    """
    Run the SVD independently on every block in block_result.
    Uses grad_avg if available, otherwise recomputes gradients.

    cfg["svd"]["mean_center"] (default False): if True, each block's gradient
    matrix is mean-centered (subtract cross-instance mean per param) before
    SVD. Used to surface instance-level structure when the common-task
    direction dominates (greater-than). On tasks with diverse instance signal
    (IOI), centering changes little.
    """
    eta         = cfg["svd"]["eta"]
    chunk       = cfg["svd"]["gram_chunk_size"]
    mean_center = cfg["svd"].get("mean_center", False)
    ul          = block_result.unit_layout
    results     = []

    use_fast = grad_avg is not None
    print(f"\n[SVD] {block_result.K} blocks, η={eta}, "
          f"path={'fast(grad_avg)' if use_fast else 'slow(backward)'}, "
          f"mean_center={mean_center}")

    for k, (unit_ids, block_idx) in enumerate(
        zip(block_result.blocks, block_result.block_indices)
    ):
        d_k    = len(block_idx)
        labels = [ul[uid].label      for uid in unit_ids]
        types  = [ul[uid].unit_type  for uid in unit_ids]
        layers = sorted({ul[uid].layer for uid in unit_ids})

        print(f"\n[SVD] Block {k}: {len(unit_ids)} units, "
              f"d_k={d_k:,}, layers={layers}, "
              f"types={sorted(set(types))}")

        # Build G_B
        G_B = build_augmented_grad_matrix(
            block_indices = block_idx,
            records       = records,
            cfg           = cfg,
            grad_avg      = grad_avg,
            model         = model,
            tokenizer     = tokenizer,
            param_filter  = param_filter,
            loss_fn       = loss_fn,
        )   # (d_k, N)

        # Gram-trick SVD
        sigma, U_rho, rho, cumvar = gram_svd(
            G_B, eta=eta, chunk=chunk, mean_center=mean_center,
        )
        redundancy = 1.0 - rho / max(len(sigma), 1)

        result = BlockSVDResult(
            block_id        = k,
            unit_ids        = unit_ids,
            unit_labels     = labels,
            unit_types      = types,
            layers          = layers,
            d_k             = d_k,
            N               = len(records),
            singular_values = sigma.cpu(),
            explained_var   = cumvar.cpu(),
            effective_rank  = rho,
            eta             = eta,
            U_rho           = U_rho.cpu(),
            redundancy_ratio= redundancy,
            global_indices  = block_idx.cpu().long(),
        )
        results.append(result)
        print(result.summary())

        del G_B
        gc.collect()

    return results


# Plotting

def plot_svd_spectra(
    results:  List[BlockSVDResult],
    save_dir: Optional[str] = None,
) -> None:
    """
    Figure 1: singular value spectrum + cumulative variance per block.
    Figure 2: effective rank vs. layer.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[SVD plot] matplotlib not available, skip.")
        return

    # Figure 1: one panel per block, grouped by type (attn rows, then mlp,
    # then mixed/other), sorted by block_id, at most 6 panels per row.
    import math

    def _block_kind(res):
        types = set(res.unit_types)
        if types <= {"attn_head"}:
            return "attn"
        if types and types <= {"mlp_gate", "mlp_up", "mlp_down"}:
            return "mlp"
        return "other"

    group_order = ["attn", "mlp", "other"]
    groups: Dict[str, list] = {g: [] for g in group_order}
    for res in results:
        groups[_block_kind(res)].append(res)
    for g in groups:
        groups[g].sort(key=lambda r: r.block_id)

    K          = len(results)
    ncols      = min(K, 6)
    group_rows = {g: math.ceil(len(groups[g]) / ncols) if groups[g] else 0
                  for g in group_order}
    nrows      = sum(group_rows.values())
    figsize    = (3.0 * ncols, 3.0 * nrows)

    local_rc = {
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Liberation Sans", "Arial"],
        "mathtext.fontset":  "dejavusans",
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.linewidth":    1.4,
        "axes.edgecolor":    "black",
        "axes.labelweight":  "normal",
        "axes.titleweight":  "normal",
        "axes.labelsize":    10,
        "axes.titlesize":    10,
        "axes.labelcolor":   "black",
        "xtick.color":       "black",
        "ytick.color":       "black",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon":    True,
    }

    with plt.rc_context(local_rc):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        row_off  = 0
        drew_any = False
        for g in group_order:
            block_list = groups[g]
            for i, res in enumerate(block_list):
                r = row_off + i // ncols
                c = i % ncols
                ax = axes[r, c]

                sv = res.singular_values.numpy()
                ev = res.explained_var.numpy()
                x  = range(1, len(sv) + 1)

                ax2 = ax.twinx()
                ax.bar(x, sv, color="#4f86c6", alpha=0.7, label="σ_i")
                ax2.plot(x, ev * 100, color="#e05c5c", linewidth=2,
                         label="cum. var")
                ax2.axhline(res.eta * 100, color="gray", linestyle=":",
                            label=f"η={res.eta*100:.0f}%")
                ax2.axvline(res.effective_rank + 0.5, color="#e05c5c",
                            linestyle="--", label="ρ")

                ax.set_xlabel("singular value index")
                ax.set_ylabel("σ_i", color="#4f86c6")
                ax2.set_ylabel("cumulative var (%)", color="#e05c5c")
                ax2.set_ylim(0, 105)
                # legend on the first panel only
                if not drew_any:
                    lines  = [ax.patches[0]] + ax2.lines[:3]
                    labels = ["σ_i", "cum. var",
                              f"η={res.eta*100:.0f}%",
                              f"ρ={res.effective_rank}"]
                    ax2.legend(lines, labels, fontsize=8, loc="lower right")
                    drew_any = True

                _types = set(res.unit_types)
                if _types and _types <= {"mlp_gate", "mlp_up", "mlp_down"}:
                    type_str = "mlp"
                else:
                    type_str = ",".join(sorted(_types))
                ax.set_title(
                    f"Block {res.block_id}  layers={res.layers}\n"
                    f"types={type_str}  ρ={res.effective_rank}  "
                    f"red={res.redundancy_ratio:.0%}",
                    fontsize=9,
                )

            # hide unused panels in the last row of this group
            used_last_row = len(block_list) % ncols
            if used_last_row != 0:
                last_row = row_off + group_rows[g] - 1
                for c in range(used_last_row, ncols):
                    axes[last_row, c].set_visible(False)
            row_off += group_rows[g]

        plt.tight_layout()
        if save_dir:
            out = Path(save_dir) / "svd_spectra.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=250, bbox_inches="tight")
            print(f"  [plot] {out}")
        plt.close()

    # Figure 2: effective rank vs. layer
    _plot_layerwise_rank(results, save_dir)


def _plot_layerwise_rank(
    results:  List[BlockSVDResult],
    save_dir: Optional[str] = None,
) -> None:
    """
    Scatter of effective rank ρ against the block's median layer,
    coloured by unit type.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    type_colors = {
        "attn_head": "#4f86c6",
        "mlp_gate":  "#e05c5c",
        "mlp_up":    "#f5a623",
        "mlp_down":  "#7ed321",
        "mixed":     "#9b59b6",
    }

    # group by unit type ("mixed" if more than one)
    points: Dict[str, Tuple[List[float], List[int]]] = {
        t: ([], []) for t in type_colors
    }

    for res in results:
        type_set = set(res.unit_types)
        if len(type_set) == 1:
            t = list(type_set)[0]
        else:
            t = "mixed"
        median_layer = float(np.median(res.layers))
        points[t][0].append(median_layer)
        points[t][1].append(res.effective_rank)

    local_rc = {
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Liberation Sans", "Arial"],
        "mathtext.fontset":  "dejavusans",
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.linewidth":    1.4,
        "axes.edgecolor":    "black",
        "axes.labelweight":  "normal",
        "axes.titleweight":  "normal",
        "axes.labelsize":    11,
        "axes.titlesize":    11,
        "axes.labelcolor":   "black",
        "xtick.color":       "black",
        "ytick.color":       "black",
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon":    True,
    }

    with plt.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        has_data = False
        for t, (xs, ys) in points.items():
            if not xs:
                continue
            ax.scatter(xs, ys, c=type_colors[t], label=t, alpha=0.8,
                       s=60, zorder=3)
            if len(xs) >= 3:
                z = np.polyfit(xs, ys, 1)
                xline = np.linspace(min(xs), max(xs), 100)
                ax.plot(xline, np.polyval(z, xline),
                        color=type_colors[t], alpha=0.4, linewidth=1.5)
            has_data = True

        if not has_data:
            plt.close(); return

        ax.set_xlabel("Layer (median of block's layers)")
        ax.set_ylabel(f"Effective rank ρ  (η={results[0].eta})")
        ax.set_title(
            "Layer-wise Effective Rank\n"
            "Expected: lower layers -> higher rank  (more lexical noise)\n"
            "         upper layers -> lower rank   (more semantic consistency)"
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "layerwise_rank.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=250, bbox_inches="tight")
        print(f"  [plot] {out}")
    plt.close()