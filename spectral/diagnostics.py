"""
Gradient diagnostics.

Phase 1: NSR and the (NM x NM) similarity heatmap from the Gram matrix.
Phase 2: per-dimension sign consistency SC^(d) from sign_accum.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from spectral.gradient_collector import CollectionResult


# Result container

@dataclass
class DiagnosticsResult:
    # Phase 1
    nsr: float
    d_intra_mean: float
    d_inter_mean: float
    gram_matrix:  torch.Tensor      # (NM, NM)

    # Phase 2
    sc:              torch.Tensor   # (D,)  float32, in [0, 1]
    sc_mean:         float
    sc_threshold:    float
    n_semantic_dims: int
    n_noise_dims:    int
    semantic_mask:   torch.Tensor   # (D,)  bool

    N: int
    M: int
    D: int

    def summary(self) -> str:
        frac = self.n_semantic_dims / max(self.D, 1)
        return (
            f"\n{'='*55}\n"
            f"  Diagnostics Summary\n"
            f"{'='*55}\n"
            f"  N={self.N}, M={self.M}, D={self.D:,}\n\n"
            f"  [B1] NSR          = {self.nsr:.4f}  "
            f"({'PASS' if self.nsr < 0.5 else 'MARGINAL' if self.nsr < 0.8 else 'FAIL'})\n"
            f"       d_intra_mean = {self.d_intra_mean:.4f}\n"
            f"       d_inter_mean = {self.d_inter_mean:.4f}\n\n"
            f"  [B2] SC_mean      = {self.sc_mean:.4f}\n"
            f"       threshold    = {self.sc_threshold}\n"
            f"       semantic dims= {self.n_semantic_dims:,} ({frac:.1%})\n"
            f"       noise dims   = {self.n_noise_dims:,}\n"
            f"{'='*55}"
        )


# Phase 1: NSR

def compute_nsr(
    gram: torch.Tensor,   # (NM, NM) cosine similarity
    N: int,
    M: int,
) -> Tuple[float, float, float]:
    """
    Compute d_intra, d_inter and NSR = d_intra / d_inter from the Gram matrix.

    Gram[i*M+j, i*M+k] = cosine_sim(g_{i,j}, g_{i,k})
    """
    NM = N * M
    assert gram.shape == (NM, NM)

    intra_sims, inter_sims = [], []

    for i in range(N):
        s = i * M
        e = s + M
        # same-instance block
        block = gram[s:e, s:e]   # (M, M)
        mask  = torch.triu(torch.ones(M, M, dtype=torch.bool), diagonal=1)
        intra_sims.extend(block[mask].tolist())

        # cross-instance
        for j in range(N):
            if j == i:
                continue
            sj = j * M
            ej = sj + M
            cross = gram[s:e, sj:ej]   # (M, M)
            inter_sims.extend(cross.flatten().tolist())

    d_intra = 1.0 - torch.tensor(intra_sims).mean().item()  # cosine dist
    d_inter = 1.0 - torch.tensor(inter_sims).mean().item()
    nsr     = d_intra / (d_inter + 1e-8)

    return nsr, d_intra, d_inter


# Phase 2: sign consistency

def compute_sc(
    sign_accum: torch.Tensor,   # (N, D)  int8
    M: int,
    threshold: float = 0.7,
    top_pct: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    SC^(d,i) = |sign_accum[i,d] / M|   ∈ [0, 1]
    SC^(d)   = mean over i

    top_pct: if set, keep the top top_pct% dimensions by SC score.
             e.g. top_pct=30 -> threshold = 70th-percentile of SC.
             Takes priority over threshold.

    Returns:
        sc            (D,) float32
        semantic_mask (D,) bool
        actual_threshold float  (the threshold value actually used)
    """
    sc_per_inst = (sign_accum.float() / M).abs()   # (N, D) ∈ [0,1]
    sc          = sc_per_inst.mean(dim=0)           # (D,)

    if top_pct is not None:
        import numpy as np
        actual_threshold = float(np.percentile(sc.cpu().numpy(), 100.0 - top_pct))
    else:
        actual_threshold = threshold

    mask = sc >= actual_threshold
    return sc, mask, actual_threshold


# Main entry point

def run_diagnostics(
    result:       CollectionResult,
    sc_threshold: float = 0.7,
    sc_top_pct:   Optional[float] = None,
) -> DiagnosticsResult:

    N, M, D = result.N, result.M, result.D
    print(f"\n[Diagnostics] Phase 1: NSR from Gram matrix ({N*M}×{N*M})...")
    nsr, d_intra, d_inter = compute_nsr(result.gram_matrix, N, M)

    pct_str = f"top_pct={sc_top_pct}%" if sc_top_pct is not None else f"threshold={sc_threshold}"
    print(f"[Diagnostics] Phase 2: Sign Consistency (D={D:,}, {pct_str})...")
    sc, semantic_mask, actual_threshold = compute_sc(
        result.sign_accum, M, sc_threshold, sc_top_pct
    )

    n_sem   = int(semantic_mask.sum().item())
    n_noise = D - n_sem

    diag = DiagnosticsResult(
        nsr=nsr,
        d_intra_mean=d_intra,
        d_inter_mean=d_inter,
        gram_matrix=result.gram_matrix,
        sc=sc,
        sc_mean=sc.mean().item(),
        sc_threshold=actual_threshold,
        n_semantic_dims=n_sem,
        n_noise_dims=n_noise,
        semantic_mask=semantic_mask,
        N=N, M=M, D=D,
    )
    print(diag.summary())
    return diag


# Plotting

def plot_all(
    diag:     DiagnosticsResult,
    save_dir: Optional[str] = None,
) -> None:
    """
    Three panels: (NM x NM) gradient similarity heatmap, SC histogram, SC CDF.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Diagnostics] matplotlib not available, skip plots.")
        return

    # Local sans-serif style overriding the global paper_style set in run.py:
    # all four spines on, thicker axis lines.
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
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle(
            f"Gradient Diagnostics  N={diag.N} M={diag.M}  "
            f"NSR={diag.nsr:.3f}  SC_mean={diag.sc_mean:.3f}",
            fontsize=13,
        )

        # Panel 1: Gram heatmap
        ax = axes[0]
        g  = diag.gram_matrix.numpy()
        im = ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        for k in range(1, diag.N):
            ax.axhline(k * diag.M - 0.5, color="white", lw=0.5)
            ax.axvline(k * diag.M - 0.5, color="white", lw=0.5)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Gradient cosine sim ({diag.N*diag.M}×{diag.N*diag.M})\n"
                     f"NSR={diag.nsr:.3f} (block-diagonal = good)")
        ax.set_xlabel("gradient vector index")
        ax.set_ylabel("gradient vector index")

        # Panel 2: SC histogram
        ax = axes[1]
        sc = diag.sc.numpy()
        ax.hist(sc, bins=100, color="#4f86c6", alpha=0.8)
        ax.axvline(diag.sc_threshold, color="#e05c5c", linestyle="--",
                   label=f"threshold={diag.sc_threshold}")
        ax.set_xlabel("SC^(d)")
        ax.set_ylabel("count")
        ax.set_title(f"Dimension-wise SC distribution\n"
                     f"semantic={diag.n_semantic_dims:,} "
                     f"({diag.n_semantic_dims/diag.D:.1%})")
        ax.legend(fontsize=10)

        # Panel 3: SC CDF
        ax = axes[2]
        sorted_sc = np.sort(sc)
        cdf = np.arange(1, len(sorted_sc)+1) / len(sorted_sc)
        ax.plot(sorted_sc, cdf * 100, color="#4f86c6")
        ax.axvline(diag.sc_threshold, color="#e05c5c", linestyle="--",
                   label=f"threshold={diag.sc_threshold}")
        frac = diag.n_semantic_dims / diag.D
        ax.axhline((1 - frac) * 100, color="gray", linestyle=":",
                   label=f"noise frac={1-frac:.1%}")
        ax.set_xlabel("SC^(d)")
        ax.set_ylabel("CDF (%)")
        ax.set_title("SC CDF")
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_dir:
            out = Path(save_dir) / "diagnostics.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=250, bbox_inches="tight")
            print(f"  [plot] {out}")
    return fig