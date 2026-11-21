"""Check why SVD importance and the Magnitude baseline give nearly identical
head rankings on some tasks (e.g. greater-than) but not others (e.g. IOI).
If G_B is close to rank-1, U_B[:, 0] is roughly the mean gradient direction and
the SVD ranking reduces to the Magnitude ranking. Panel A: top-K overlap of the
two rankings (with Kendall τ / Spearman ρ); panel B: per-head direction score
d(u) = ‖ḡ[u]‖² / E_n‖g_n[u]‖² (high d -> near rank-1 gradients).

Usage:
    python -m circuits_discovery.diagnostics.svd_vs_magnitude_collapse \
        --tasks GT:results/gpt2_small_gt_N100_M5 \
                IOI:results/gpt2_small_ioi_strong_N100_M5 \
        --out_dir results/cross_task_diagnostics
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

from utils.paper_style import apply_paper_style_globally, get_preset


# Loading

def load_attn_rankings(task_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (svd, magnitude, direction) DataFrames restricted to attn heads."""
    svd  = pd.read_csv(task_dir / "ranked_units.csv")
    mag  = pd.read_csv(task_dir / "variance_baseline" / "ranked_units_magnitude.csv")
    dirn = pd.read_csv(task_dir / "variance_baseline" / "ranked_units_direction.csv")
    return (svd[svd.unit_type == "attn_head"].copy(),
            mag[mag.unit_type == "attn_head"].copy(),
            dirn[dirn.unit_type == "attn_head"].copy())


# Metrics

def overlap_curve(a: pd.DataFrame, b: pd.DataFrame, k_max: int) -> np.ndarray:
    """Top-K Jaccard-style overlap as a function of K."""
    a_ids = a["unit_id"].tolist()
    b_ids = b["unit_id"].tolist()
    out = np.zeros(k_max)
    a_seen, b_seen = set(), set()
    for k in range(1, k_max + 1):
        a_seen.add(a_ids[k - 1])
        b_seen.add(b_ids[k - 1])
        out[k - 1] = len(a_seen & b_seen) / k
    return out


def rank_correlations(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[float, float]:
    A = a.sort_values("unit_id").reset_index(drop=True)
    B = b.set_index("unit_id").reindex(A["unit_id"]).reset_index()
    tau = kendalltau(A["rank"].values, B["rank"].values)[0]
    rho = spearmanr (A["rank"].values, B["rank"].values)[0]
    return float(tau), float(rho)


# Plot

def plot(curves: List[dict], out_dir: Path) -> None:
    apply_paper_style_globally("double")
    preset = get_preset("double")
    lw = preset["linewidth"]

    fig, axes = plt.subplots(1, 2, figsize=(preset["figwidth"], 2.2))

    # Panel A: top-K overlap
    ax = axes[0]
    k_max = curves[0]["overlap"].size
    ks = np.arange(1, k_max + 1)
    for c in curves:
        ax.plot(ks, c["overlap"],
                label=f"{c['name']}  (τ={c['tau']:+.2f}, ρ={c['rho']:+.2f})",
                linewidth=lw)
    ax.axhline(1.0, color="0.7", linestyle=":", linewidth=lw * 0.6)
    ax.set_xlabel("K (top-K attn heads)")
    ax.set_ylabel("SVD ∩ Magnitude  /  K")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, k_max)
    ax.set_title("(A) Ranking overlap: SVD vs Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=preset["legend_fs"])

    # Panel B: direction distribution
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    for c in curves:
        ax.hist(c["direction"], bins=bins, alpha=0.55, label=c["name"],
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel(r"direction $= \|\bar g\|^2 / \mathbb{E}_n\|g_n\|^2$")
    ax.set_ylabel("# attn heads")
    ax.set_xlim(0, 1)
    ax.set_title("(B) Per-head direction concentration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=preset["legend_fs"])

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"svd_vs_magnitude_collapse.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.close(fig)


# Main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True,
                   help="LABEL:DIR pairs, e.g. GT:results/gpt2_small_gt_N100_M5")
    p.add_argument("--out_dir", type=Path,
                   default=Path("results/cross_task_diagnostics"))
    p.add_argument("--k_max", type=int, default=144)
    args = p.parse_args()

    curves = []
    print(f"\n{'Task':12s} | top-10 ov | top-50 ov | Kendall τ | Spearman ρ "
          f"| dir mean | dir median")
    print("-" * 92)
    for spec in args.tasks:
        if ":" not in spec:
            raise SystemExit(f"--tasks needs LABEL:DIR; got {spec!r}")
        label, d = spec.split(":", 1)
        d = Path(d)
        svd, mag, dirn = load_attn_rankings(d)
        ov   = overlap_curve(svd, mag, args.k_max)
        tau, rho = rank_correlations(svd, mag)
        direction = dirn["importance"].to_numpy()

        curves.append({
            "name":      label,
            "overlap":   ov,
            "tau":       tau,
            "rho":       rho,
            "direction": direction,
        })
        print(f"{label:12s} | {ov[9]:>9.2%} | {ov[49]:>9.2%} "
              f"| {tau:>+.3f}    | {rho:>+.3f}     "
              f"| {direction.mean():>.3f}    | {np.median(direction):>.3f}")

    plot(curves, args.out_dir)


if __name__ == "__main__":
    main()
