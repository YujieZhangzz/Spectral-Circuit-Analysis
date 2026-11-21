"""
Compare top-K metrics (Recall, Precision, Hits) of multiple ranking CSVs
against the IOI canonical 26 attention heads. Random baseline (100 trials)
is overlaid as mean ± 1 std band.

The rankings to compare are listed in METHODS below.

Usage: python spectral/plot_topk_compare.py
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from utils.paper_style import paper_rc, paper_size  # noqa: E402

# Each entry is (display_label, path_to_ranking_csv).
# CSVs must have columns: rank, label, unit_type. Only rows with
# unit_type == "attn_head" are used.
# Legend order: ours, then strong baselines (descending mAP), then weak baselines.
# Random is added separately and always rendered last in the legend.
METHODS = [
    # Ours
    ("SCA (ours)",        "results/gpt2_small_ioi_strong_N100_M5_ori/ranked_units.csv"),
    # Strong baselines (EAP family)
    ("EAP-IG",            "eap/ioi_task_head_ranking.csv"),
    ("EAP-IG-KL",         "eap/ioi_kl_head_ranking.csv"),
    ("EAP",               "eap/ioi_vanilla_head_ranking.csv"),
    # Weak baselines (raw gradient)
    ("Magnitude (raw)",   "results/gpt2_small_ioi_strong_N100_M5_ori/variance_baseline/ranked_units_magnitude.csv"),
    ("Direction (raw)",   "results/gpt2_small_ioi_strong_N100_M5_ori/variance_baseline/ranked_units_direction.csv"),
]

# Explicit color map per method, so reordering METHODS doesn't reshuffle colors.
# SVD = blue (ours). EAP family = warm reds/purple. Raw baselines = orange/green.
METHOD_COLORS = {
    "SCA (ours)":      "#1f77b4",   # blue
    "EAP-IG":          "#9467bd",   # purple
    "EAP-IG-KL":       "#8c564b",   # brown
    "EAP":             "#d62728",   # red
    "Magnitude (raw)": "#ff7f0e",   # orange
    "Direction (raw)": "#2ca02c",   # green
}

K_MAX_LIST = [50, 60, 80, 100, 120, 144]   # one figure per K_max
N_RANDOM_TRIALS = 100
RANDOM_SEED = 42
N_LAYERS, N_HEADS = 12, 12             # GPT-2 small head universe
OUT_DIR = Path("results/gpt2_small_ioi_strong_N100_M5_ori/topk_compare/")

IOI_GT = {
    "L2H2", "L4H11", "L0H1", "L0H10", "L3H0",
    "L5H5", "L5H8", "L5H9", "L6H9",
    "L7H3", "L7H9", "L8H6", "L8H10",
    "L9H6", "L9H9", "L10H0",
    "L10H7", "L11H10",
    "L9H0", "L9H7", "L10H1", "L10H2", "L10H6", "L10H10", "L11H2", "L11H9",
}


def load_attn_ranking(csv_path: Path) -> list:
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("unit_type") == "attn_head":
                rows.append((int(row["rank"]), row["label"]))
    rows.sort()
    return [lab for _, lab in rows]


def compute_curves(ranking: list, gt: set, ks: list):
    n_gt = len(gt)
    hits, prec, rec = [], [], []
    cum = 0; aps = []
    for i, lab in enumerate(ranking, start=1):
        if lab in gt:
            cum += 1
            aps.append(cum / i)
    aps += [0.0] * (n_gt - cum)
    map_score = sum(aps) / n_gt

    for k in ks:
        h = len(set(ranking[:k]) & gt)
        hits.append(h)
        prec.append(h / k if k > 0 else 0.0)
        rec.append(h / n_gt if n_gt > 0 else 0.0)
    return np.array(hits), np.array(prec), np.array(rec), map_score


def random_curves(ks: list, n_trials: int, seed: int):
    all_heads = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
    H = np.zeros((n_trials, len(ks)))
    P = np.zeros((n_trials, len(ks)))
    R = np.zeros((n_trials, len(ks)))
    maps = []
    for t in range(n_trials):
        rng = random.Random(seed + t)
        ranked = all_heads[:]
        rng.shuffle(ranked)
        h, p, r, m = compute_curves(ranked, IOI_GT, ks)
        H[t], P[t], R[t] = h, p, r
        maps.append(m)
    return (H.mean(0), H.std(0),
            P.mean(0), P.std(0),
            R.mean(0), R.std(0),
            float(np.mean(maps)), float(np.std(maps)))


def plot_one(k_max: int, results: dict, rand_stats, out_path: Path):
    ks = np.arange(1, k_max + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # (title, index into (hits, prec, rec))
    panels = [("Recall@K", 2), ("Precision@K", 1)]

    (_, _, rand_p_m, rand_p_s, rand_r_m, rand_r_s,
     rand_map_m, rand_map_s) = rand_stats

    for ax, (title, idx) in zip(axes, panels):
        for label, _, arrs, m in results["methods"]:
            ax.plot(ks, arrs[idx], label=f"{label} (mAP={m:.3f})",
                    color=METHOD_COLORS.get(label), lw=1.6)
        if title == "Recall@K":
            rm, rs = rand_r_m, rand_r_s
        else:
            rm, rs = rand_p_m, rand_p_s
        ax.plot(ks, rm, color="gray", lw=1.4, ls="--",
                label=f"Random (n={N_RANDOM_TRIALS}, mAP={rand_map_m:.3f}±{rand_map_s:.3f})")
        ax.fill_between(ks, rm - rs, rm + rs, color="gray", alpha=0.18)

        ax.axvline(len(IOI_GT), ls=":", c="black", lw=0.8, alpha=0.6)
        ax.text(len(IOI_GT), 0.96, " |GT|=26",
                transform=ax.get_xaxis_transform(),
                fontsize=8, va="top", ha="left",
                color="black", alpha=0.7)

        ax.set_xlabel("K (top-K attention heads)")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("metric value")
    # shared legend below the panels to avoid covering curves
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(4, len(labels)), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"Top-K performance vs IOI canonical 26 heads  (K_max={k_max})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_one_paper(k_max: int, results: dict, rand_stats, out_path: Path):
    """Paper version: double-column 1x2 panels, paper_rc style. Writes PDF + PNG."""
    ks = np.arange(1, k_max + 1)

    with plt.rc_context(paper_rc("double")):
        fig, axes = plt.subplots(1, 2, figsize=paper_size("double", aspect=2.4))

        panels = [("Recall@K", 2), ("Precision@K", 1)]
        (_, _, rand_p_m, rand_p_s, rand_r_m, rand_r_s,
         rand_map_m, rand_map_s) = rand_stats

        for ax, (title, idx) in zip(axes, panels):
            for label, _, arrs, m in results["methods"]:
                ax.plot(ks, arrs[idx], label=f"{label} (mAP={m:.3f})",
                        color=METHOD_COLORS.get(label), lw=1.0)
            if title == "Recall@K":
                rm, rs = rand_r_m, rand_r_s
            else:
                rm, rs = rand_p_m, rand_p_s
            ax.plot(ks, rm, color="gray", lw=0.9, ls="--",
                    label=f"Random (mAP={rand_map_m:.3f}±{rand_map_s:.2f})")
            ax.fill_between(ks, rm - rs, rm + rs, color="gray", alpha=0.18)

            ax.axvline(len(IOI_GT), ls=":", c="black", lw=0.5, alpha=0.6)
            ax.text(len(IOI_GT), 0.96, " |GT|=26",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7.5, va="top", ha="left",
                    color="black", alpha=0.7)

            ax.set_xlabel("K (top-K attention heads)")
            ax.set_title(title)
            ax.grid(alpha=0.3)

        axes[0].set_ylabel("Metric value")
        handles, labels = axes[0].get_legend_handles_labels()
        # Two-row legend, each row centered independently:
        #   row 0 (4 items):  SCA (ours) | Magnitude (raw) | EAP        | EAP-IG-KL
        #   row 1 (3 items):  Random     | Direction (raw) | EAP-IG     (centered)
        row_top = ["SCA (ours)", "Magnitude (raw)", "EAP ", "EAP-IG-KL"]
        row_bot = ["Random", "Direction (raw)", "EAP-IG "]

        def pick(prefixes):
            label_to_idx = {l: i for i, l in enumerate(labels)}
            out = []
            for p in prefixes:
                for l, i in label_to_idx.items():
                    if l.startswith(p):
                        out.append(i); break
            return [handles[i] for i in out], [labels[i] for i in out]

        h_top, l_top = pick(row_top)
        h_bot, l_bot = pick(row_bot)

        leg1 = fig.legend(h_top, l_top, loc="lower center",
                          ncol=4, bbox_to_anchor=(0.5, 0.06),
                          frameon=False, columnspacing=1.4, handlelength=1.8)
        fig.add_artist(leg1)
        fig.legend(h_bot, l_bot, loc="lower center",
                   ncol=3, bbox_to_anchor=(0.5, -0.02),
                   frameon=False, columnspacing=1.4, handlelength=1.8)
        fig.tight_layout(rect=(0, 0.18, 1, 1))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # PDF + 300 dpi PNG
        out_pdf = out_path.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_pdf}")
        print(f"[saved] {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load all method rankings once (full ranking list each)
    method_rankings = []
    for label, path in METHODS:
        path = Path(path)
        if not path.exists():
            print(f"[WARN] missing: {path}, skipping {label}")
            continue
        method_rankings.append((label, load_attn_ranking(path)))

    if not method_rankings:
        print("[ERR] no valid ranking CSVs in METHODS list.")
        return

    for k_max in K_MAX_LIST:
        ks = list(range(1, k_max + 1))
        results = {"methods": []}
        for label, ranking in method_rankings:
            arrs = compute_curves(ranking, IOI_GT, ks)  # (hits, prec, rec, map)
            results["methods"].append((label, ranking, arrs[:3], arrs[3]))

        rand_stats = random_curves(ks, N_RANDOM_TRIALS, RANDOM_SEED)

        out_path = OUT_DIR / f"topk_compare_K{k_max}.png"
        plot_one(k_max, results, rand_stats, out_path)

        # paper-ready variant only at K=144
        if k_max == 144:
            paper_path = OUT_DIR / f"topk_compare_K{k_max}_paper.png"
            plot_one_paper(k_max, results, rand_stats, paper_path)


if __name__ == "__main__":
    main()
