"""Paper plots for head-level faithfulness (symlog / linear recovery vs K,
recovery@K bars, and a combined figure), read from head_faithfulness.json.

Usage:
    python -m circuits_discovery.faithfulness.plot_head_final
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from utils.paper_style import paper_rc, paper_size  # noqa: E402


METHOD_COLORS = {
    "SVD (ours)":            "#1f77b4",   # head patching
    "SVD subspace (ours)":   "#1f77b4",   # weight-subspace knockout
    "SVD z-subspace (ours)": "#1f77b4",   # z-subspace activation patching
    "EAP-IG":                "#9467bd",
    "EAP-IG-KL":             "#8c564b",
    "EAP":                   "#d62728",
    "Magnitude (raw)":       "#ff7f0e",
    "Direction (raw)":       "#2ca02c",
    "Random":                "gray",
}

KEY_KS = [5, 10, 20, 26, 30, 40, 50]


def auc(ks: np.ndarray, ys: np.ndarray) -> float:
    return float(np.trapz(ys, ks) / (ks[-1] - ks[0]))


# Legend label for a method, shared by all plots
def legend_label(method: str, recovery: np.ndarray, ks: np.ndarray) -> str:
    return f"{method} (AUC={auc(ks, recovery):+.3f}, min={recovery.min():+.3f})"


# Bottom legend, 4 columns x 2 rows
def add_grouped_legend(fig, axes_for_handles):
    handles, labels = axes_for_handles.get_legend_handles_labels()
    # Legend fills column-major, so rows read:
    #   row 0: SVD            Magnitude  EAP     EAP-IG-KL
    #   row 1: SVD z-subspace Direction  EAP-IG  Random
    desired = [
        "SVD (ours)", "SVD z-subspace (ours)",
        "Magnitude (raw)", "Direction (raw)",
        "EAP", "EAP-IG",
        "EAP-IG-KL", "Random",
    ]
    # split on " (AUC" so parens like "(ours)" / "(raw)" stay in the method name
    label_to_idx = {l.split(" (AUC")[0]: i for i, l in enumerate(labels)}
    order = [label_to_idx[d] for d in desired if d in label_to_idx]
    h2 = [handles[i] for i in order]
    l2 = [labels[i]  for i in order]
    fig.legend(h2, l2, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04),
               frameon=False, columnspacing=1.6, handlelength=2.0)


def _draw_recovery_curves(ax, ks, recovery, rand_std, span,
                          sub_ks, sub_recovery, subz_ks, subz_recovery):
    for label, y in recovery.items():
        c = METHOD_COLORS.get(label, "black")
        leg = legend_label(label, y, ks)
        if label == "Random":
            ax.plot(ks, y, color=c, ls="--", lw=0.9, label=leg)
            ax.fill_between(ks, y - rand_std/span, y + rand_std/span,
                            color=c, alpha=0.15)
        else:
            ax.plot(ks, y, color=c, lw=1.0, label=leg)
    if sub_recovery is not None:
        ax.plot(sub_ks, sub_recovery,
                color=METHOD_COLORS["SVD subspace (ours)"],
                ls=":", lw=1.2,
                label=legend_label("SVD subspace (ours)", sub_recovery, sub_ks))
    if subz_recovery is not None:
        ax.plot(subz_ks, subz_recovery,
                color=METHOD_COLORS["SVD z-subspace (ours)"],
                ls="-.", lw=1.2,
                label=legend_label("SVD z-subspace (ours)", subz_recovery, subz_ks))
    ax.axhline(0.0, color="black", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(1.0, color="black", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("K (top-K attention units affected)")
    ax.set_ylabel("recovery score")
    ax.grid(alpha=0.3)


def plot_symlog_panel(ax, ks, recovery, rand_std, span, linthresh,
                      sub_ks=None, sub_recovery=None,
                      subz_ks=None, subz_recovery=None):
    _draw_recovery_curves(ax, ks, recovery, rand_std, span,
                          sub_ks, sub_recovery, subz_ks, subz_recovery)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_title(f"recovery vs K  (symlog, linthresh={linthresh})")


def plot_linear_panel(ax, ks, recovery, rand_std, span,
                      sub_ks=None, sub_recovery=None,
                      subz_ks=None, subz_recovery=None):
    _draw_recovery_curves(ax, ks, recovery, rand_std, span,
                          sub_ks, sub_recovery, subz_ks, subz_recovery)
    ax.set_title("recovery vs K  (linear)")


def plot_bars_panel(ax, ks, recovery, sub_ks=None, sub_recovery=None,
                    subz_ks=None, subz_recovery=None):
    # Bar order: ours, EAP family, raw baselines, Random
    bar_order = [
        "SVD (ours)", "SVD z-subspace (ours)", "SVD subspace (ours)",
        "EAP", "EAP-IG", "EAP-IG-KL",
        "Magnitude (raw)", "Direction (raw)",
        "Random",
    ]
    available = set(recovery.keys())
    if sub_recovery is not None:  available.add("SVD subspace (ours)")
    if subz_recovery is not None: available.add("SVD z-subspace (ours)")
    methods = [m for m in bar_order if m in available]
    n_methods = len(methods)
    width = 0.8 / n_methods
    x_pos = np.arange(len(KEY_KS))
    for i, m in enumerate(methods):
        if m == "SVD subspace (ours)":
            rec_curve = sub_recovery; curve_ks = sub_ks
        elif m == "SVD z-subspace (ours)":
            rec_curve = subz_recovery; curve_ks = subz_ks
        else:
            rec_curve = recovery[m]; curve_ks = ks
        vals = [rec_curve[int(np.where(curve_ks == k)[0][0])] for k in KEY_KS]
        c = METHOD_COLORS.get(m, "black")
        offset = (i - n_methods / 2 + 0.5) * width
        kwargs = dict(color=c, edgecolor="white", linewidth=0.5)
        if m == "SVD subspace (ours)":
            kwargs.update(hatch="///")
        elif m == "SVD z-subspace (ours)":
            kwargs.update(hatch="...")
        ax.bar(x_pos + offset, vals, width=width,
               label=legend_label(m, rec_curve, curve_ks), **kwargs)
    ax.axhline(0.0, color="black", lw=0.7)
    ax.axhline(1.0, color="black", ls=":", lw=0.7, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={k}" for k in KEY_KS], rotation=30, ha="right")
    ax.set_ylabel("recovery score")
    ax.set_title("Recovery @ K  (lower = faster damage = better)")
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/faithfulness_head"))
    ap.add_argument("--sub_dir", type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/faithfulness_subspace"),
                    help="Subspace knockout output dir; z-subspace results are read from "
                         "its sibling faithfulness_subspace_z/")
    ap.add_argument("--linthresh", type=float, default=0.05)
    args = ap.parse_args()

    # Head-level activation patching
    data = json.loads((args.in_dir / "head_faithfulness.json").read_text())
    ks = np.array(data["ks"])
    ld_clean = data["ld_clean"]; ld_corr = data["ld_corrupted"]
    span = ld_clean - ld_corr
    curves = {k: np.array(v) for k, v in data["curves"].items()}
    rand_std = np.array(data["random_std"])
    recovery = {k: (v - ld_corr) / span for k, v in curves.items()}

    # The weight-subspace knockout curve is not plotted (its damage is shallow).
    sub_ks = sub_recovery = None

    # z-subspace activation patching (optional)
    subz_ks = subz_recovery = None
    subz_path = args.sub_dir.parent / "faithfulness_subspace_z" / "subspace_z_faithfulness.json"
    if subz_path.exists():
        subz = json.loads(subz_path.read_text())
        subz_ks = np.array(subz["ks"])
        subz_clean = subz["ld_clean"]; subz_corr = subz["ld_corrupted"]
        subz_recovery = (np.array(subz["logit_diff"]) - subz_corr) \
                        / (subz_clean - subz_corr)
        print(f"[loaded z-subspace] clean={subz_clean:.3f} corrupted={subz_corr:.3f}  "
              f"K range {subz_ks[0]}..{subz_ks[-1]}")
    else:
        print(f"[no z-subspace data] {subz_path} not found, skipping")

    # Single-column (3.5in) figures; the combined figure uses double column (7in)
    def _save_both(fig, base: Path):
        """Save as .pdf and 300-dpi .png."""
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {base.with_suffix('.pdf')}")
        print(f"[saved] {base.with_suffix('.png')}")

    with plt.rc_context(paper_rc("single")):
        # (1) symlog
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.45))
        plot_symlog_panel(ax, ks, recovery, rand_std, span, args.linthresh,
                          sub_ks=sub_ks, sub_recovery=sub_recovery,
                          subz_ks=subz_ks, subz_recovery=subz_recovery)
        add_grouped_legend(fig, ax)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        _save_both(fig, args.in_dir / "head_faithfulness_symlog")

        # (1b) linear y
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.45))
        plot_linear_panel(ax, ks, recovery, rand_std, span,
                          sub_ks=sub_ks, sub_recovery=sub_recovery,
                          subz_ks=subz_ks, subz_recovery=subz_recovery)
        add_grouped_legend(fig, ax)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        _save_both(fig, args.in_dir / "head_faithfulness_linear")

        # (2) bars
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.45))
        plot_bars_panel(ax, ks, recovery,
                        sub_ks=sub_ks, sub_recovery=sub_recovery,
                        subz_ks=subz_ks, subz_recovery=subz_recovery)
        add_grouped_legend(fig, ax)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        _save_both(fig, args.in_dir / "head_faithfulness_bars")

    # (3) combined, double column
    with plt.rc_context(paper_rc("double")):
        fig, axes = plt.subplots(1, 2, figsize=paper_size("double", aspect=2.4))
        plot_symlog_panel(axes[0], ks, recovery, rand_std, span, args.linthresh,
                          sub_ks=sub_ks, sub_recovery=sub_recovery,
                          subz_ks=subz_ks, subz_recovery=subz_recovery)
        plot_bars_panel(axes[1], ks, recovery,
                        sub_ks=sub_ks, sub_recovery=sub_recovery,
                        subz_ks=subz_ks, subz_recovery=subz_recovery)
        axes[0].set_title("(a) " + axes[0].get_title())
        axes[1].set_title("(b) " + axes[1].get_title())
        add_grouped_legend(fig, axes[0])
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        _save_both(fig, args.in_dir / "head_faithfulness_combined")


if __name__ == "__main__":
    main()
