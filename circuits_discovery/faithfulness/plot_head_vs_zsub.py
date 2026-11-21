"""Compare full head patching (solid) with patching only the SVD subspace U_z
of each head (dashed) for the top-K heads of SVD, EAP, EAP-IG and EAP-IG-KL.
A small gap means U_z captures most of the task signal in those heads.

Usage:
    python -m circuits_discovery.faithfulness.plot_head_vs_zsub
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
    "SVD (ours)": "#1f77b4",
    "EAP":        "#d62728",
    "EAP-IG":     "#9467bd",
    "EAP-IG-KL":  "#8c564b",
}

# (display name in legend, head-curves key in head_faithfulness.json,
#  zsub JSON filename suffix)
PAIRS = [
    ("SVD (ours)", "SVD (ours)", "svd"),
    ("EAP",        "EAP",        "eap"),
    ("EAP-IG",     "EAP-IG",     "eap_ig"),
    ("EAP-IG-KL",  "EAP-IG-KL",  "eap_ig_kl"),
]


def auc(ks, ys):
    return float(np.trapz(ys, ks) / (ks[-1] - ks[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/faithfulness_head"))
    ap.add_argument("--zsub_dir", type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/faithfulness_subspace_z"))
    ap.add_argument("--k_max", type=int, default=50,
                    help="Crop x-axis to [0, k_max]")
    args = ap.parse_args()

    head = json.loads((args.head_dir / "head_faithfulness.json").read_text())
    head_ks = np.array(head["ks"])
    span_h = head["ld_clean"] - head["ld_corrupted"]
    rand_std = np.array(head["random_std"]) / span_h
    head_recovery = {k: (np.array(v) - head["ld_corrupted"]) / span_h
                     for k, v in head["curves"].items()}

    zsub = {}
    for _, _, tag in PAIRS:
        path = args.zsub_dir / f"subspace_z_{tag}.json"
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        d = json.loads(path.read_text())
        ks = np.array(d["ks"])
        span = d["ld_clean"] - d["ld_corrupted"]
        rec = (np.array(d["logit_diff"]) - d["ld_corrupted"]) / span
        zsub[tag] = (ks, rec)

    # single-column paper style (3.5in)
    rc = paper_rc("single")
    plt.rcParams.update(rc)
    fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.20))

    # crop helper
    def crop(x, y, kmax):
        m = x <= kmax
        return x[m], y[m]

    for display, head_key, tag in PAIRS:
        c = METHOD_COLORS.get(display, "black")
        # head patching curve
        if head_key in head_recovery:
            x, y = crop(head_ks, head_recovery[head_key], args.k_max)
            ax.plot(x, y, color=c, ls="-", lw=1.1,
                    label=f"{display} head     (AUC={auc(x,y):+.2f}, "
                          f"min={y.min():+.2f})")
        # z-subspace curve
        if tag in zsub:
            zks, zrec = zsub[tag]
            x, y = crop(zks, zrec, args.k_max)
            ax.plot(x, y, color=c, ls="--", lw=1.1,
                    label=f"{display} @ z-sub  (AUC={auc(x,y):+.2f}, "
                          f"min={y.min():+.2f})")

    # Random (reference)
    if "Random" in head_recovery:
        x, y = crop(head_ks, head_recovery["Random"], args.k_max)
        rs = rand_std[head_ks <= args.k_max]
        ax.plot(x, y, color="gray", ls=":", lw=0.9,
                label=f"Random (AUC={auc(x,y):+.2f})")
        ax.fill_between(x, y - rs, y + rs, color="gray", alpha=0.15)

    ax.axhline(0.0, color="black", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(1.0, color="black", ls=":", lw=0.5, alpha=0.5)
    ax.text(args.k_max, 1.0, " clean    ",     fontsize=9, va="bottom",
            ha="right", alpha=0.6)
    ax.text(args.k_max, 0.0, " corrupted   ",  fontsize=9, va="bottom",
            ha="right", alpha=0.6)
    ax.set_xlim(0, args.k_max)
    ax.set_ylim(-0.6, 1.1)
    ax.set_xlabel("K (top-K heads patched cumulatively)")
    ax.set_ylabel("recovery score")
    ax.set_title("head patching (solid) vs z-subspace patching (dashed)\n"
                 "small gap -> U_z captures most IOI signal of that ranking's heads")
    ax.grid(alpha=0.3)
    ax.minorticks_on()
    ax.grid(which="minor", alpha=0.12, linestyle=":")

    # bottom legend, 3 cols x 3 rows
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.04),
               frameon=False, columnspacing=1.6, handlelength=2.4)
    fig.tight_layout(rect=(0, 0.22, 1, 1))

    base = args.head_dir / f"head_vs_zsub_K{args.k_max}"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {base.with_suffix('.pdf')}")
    print(f"[saved] {base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
