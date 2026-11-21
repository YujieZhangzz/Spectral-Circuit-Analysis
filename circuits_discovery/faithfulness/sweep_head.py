"""Head-level faithfulness sweep: for each ranking, patch the top-K attention
heads with corrupted z for K = 0..k_max and record the task metric. Random
rankings are averaged over --n_random_trials shuffles.

Usage:
    python -m circuits_discovery.faithfulness.sweep_head \
        --device cuda:0 --n 100 --k_max 144 --k_step 1
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from circuits_discovery.faithfulness.data     import load_ioi
from circuits_discovery.faithfulness.patching import (
    tokenize_pair, cache_corrupted_z, run_clean_with_patches,
    make_metric_fn,
)


# Methods are built from the CLI arguments (see build_methods)
METHOD_COLORS = {
    "SVD (ours)":      "#1f77b4",
    "EAP-IG":          "#9467bd",
    "EAP-IG-KL":       "#8c564b",
    "EAP":             "#d62728",
    "Magnitude (raw)": "#ff7f0e",
    "Direction (raw)": "#2ca02c",
    "Random":          "gray",
}


def parse_label(label: str) -> Tuple[int, int]:
    """'L9H6' -> (9, 6).  Returns None if not an attn head label."""
    if label[0] != "L" or "H" not in label:
        return None  # type: ignore
    l, h = label[1:].split("H")
    return int(l), int(h)


def load_attn_ranking(csv_path: Path) -> List[Tuple[int, int]]:
    """Read a ranked_units CSV, filter to attn_head, return (layer, head) list
    in ascending rank order."""
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("unit_type") != "attn_head":
                continue
            lh = parse_label(row["label"])
            if lh is None:
                continue
            rows.append((int(row["rank"]), lh))
    rows.sort(key=lambda x: x[0])
    return [lh for _, lh in rows]


def sweep_one_ranking(
    model:          HookedTransformer,
    clean_tokens:   torch.Tensor,
    corrupted_z:    dict,
    metric_fn,
    ranked_heads:   List[Tuple[int, int]],
    ks:             List[int],
) -> List[float]:
    """Run model with cumulative top-K patching for each K in ks; return
    list of mean metric values, one per K."""
    out = []
    for k in ks:
        head_set = ranked_heads[:k]
        logits = run_clean_with_patches(model, clean_tokens, corrupted_z, head_set)
        m = metric_fn(logits).item()
        del logits
        out.append(m)
    return out


def random_curve(
    model, clean_tokens, corrupted_z, metric_fn,
    ks, n_layers, n_heads, n_trials, seed=42,
) -> Tuple[np.ndarray, np.ndarray]:
    all_lh = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    Y = np.zeros((n_trials, len(ks)))
    for t in range(n_trials):
        rng = random.Random(seed + t)
        shuffled = all_lh[:]
        rng.shuffle(shuffled)
        Y[t] = sweep_one_ranking(
            model, clean_tokens, corrupted_z, metric_fn, shuffled, ks,
        )
    return Y.mean(0), Y.std(0)


def build_methods(args) -> List[Tuple[str, str]]:
    """Build the (label, csv_path) list from CLI arguments, in the order SVD,
    EAP family, raw baselines. Missing files are skipped with a warning; at
    least one ranking is required."""
    candidates = [
        ("SVD (ours)",      args.svd_csv),
        ("EAP",             args.eap_csv),
        ("EAP-IG",          args.eap_ig_csv),
        ("EAP-IG-KL",       args.eap_ig_kl_csv),
        ("Magnitude (raw)", args.magnitude_csv),
        ("Direction (raw)", args.direction_csv),
    ]
    methods: List[Tuple[str, str]] = []
    for name, p in candidates:
        if p is None:
            continue
        if not Path(p).exists():
            print(f"[WARN] {name}: {p} not found, skipping")
            continue
        methods.append((name, str(p)))
    if not methods:
        raise SystemExit(
            "[ERROR] no method csv supplied; need at least one of "
            "--svd_csv / --eap_csv / --eap_ig_csv / --eap_ig_kl_csv / "
            "--magnitude_csv / --direction_csv")
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="prompts dataset csv (clean+corrupted pairs)")
    ap.add_argument("--n",      type=int, default=100)
    ap.add_argument("--start",  type=int, default=0,
                    help="Skip first --start rows of csv before taking --n "
                         "(disjoint subsets, e.g. front-100 vs back-100).")
    ap.add_argument("--model",  default="gpt2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--k_max",  type=int, default=144)
    ap.add_argument("--k_step", type=int, default=1)
    ap.add_argument("--n_random_trials", type=int, default=5)
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="output dir for head_faithfulness.json")

    # Ranking CSVs (all optional, at least one required)
    ap.add_argument("--svd_csv",       type=Path,
                    help="our model's ranked_units.csv")
    ap.add_argument("--magnitude_csv", type=Path,
                    help="ranked_units_magnitude.csv from variance_baseline/")
    ap.add_argument("--direction_csv", type=Path,
                    help="ranked_units_direction.csv from variance_baseline/")
    ap.add_argument("--eap_csv",       type=Path,
                    help="EAP (vanilla) head_ranking csv")
    ap.add_argument("--eap_ig_csv",    type=Path,
                    help="EAP-IG head_ranking csv")
    ap.add_argument("--eap_ig_kl_csv", type=Path,
                    help="EAP-IG-KL head_ranking csv")
    ap.add_argument("--gt_mode", default="prob_diff",
                    choices=["prob_diff", "boundary_logit_diff"],
                    help="Greater-than metric: prob_diff (default) or "
                         "boundary_logit_diff (= logit(YY₀+1) - logit(YY₀))")
    ap.add_argument("--verb_list_csv", type=Path, default=Path(
        "eap_data/sva/combined_verb_list.csv"),
        help="Required for task='sva' verb_set variant: sing/plur verb pair csv")
    ap.add_argument("--sva_metric_variant", default="verb_set",
                    choices=["verb_set", "is_are"],
                    help="SVA metric: 'verb_set' (default, prob_diff over "
                         "verb_list_csv) or 'is_are' (logit_diff on ' is'/' are')")

    args = ap.parse_args()
    METHODS = build_methods(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(range(0, args.k_max + 1, args.k_step))
    if 0 not in ks:
        ks = [0] + ks
    if args.k_max not in ks:
        ks.append(args.k_max)

    print(f"[1/5] Loading {args.model} on {args.device}")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    print(f"[2/5] Loading {args.n} prompt pairs from {args.csv} "
          f"(start={args.start}, slice=[{args.start}, {args.start + args.n}))")
    batch = load_ioi(args.csv, n=args.n, start=args.start,
                     gt_mode=args.gt_mode, tokenizer=model.tokenizer)
    print(f"      task = {batch.task!r}  (gt_mode={args.gt_mode!r})")
    clean_tokens, corrupted_tokens, input_lengths, n_pos = tokenize_pair(
        model, batch.clean, batch.corrupted,
    )
    clean_tokens     = clean_tokens.to(args.device)
    corrupted_tokens = corrupted_tokens.to(args.device)
    input_lengths    = input_lengths.to(args.device)
    labels           = batch.labels.to(args.device)
    metric_fn        = make_metric_fn(batch.task, model, input_lengths, labels,
                                       verb_list_csv=args.verb_list_csv,
                                       sva_metric_variant=args.sva_metric_variant)
    if batch.task == "sva":
        print(f"      sva_metric_variant = {args.sva_metric_variant!r}  "
              f"({metric_fn.name})")

    print(f"[3/5] Caching corrupted z (B={batch.n}, n_pos={n_pos})")
    corrupted_z = cache_corrupted_z(model, corrupted_tokens)

    # Baseline metrics (no patching for clean; full corrupted forward)
    with torch.no_grad():
        clean_logits = model(clean_tokens)
        ld_clean = metric_fn(clean_logits).item()
        del clean_logits
        corr_logits = model(corrupted_tokens)
        ld_corr = metric_fn(corr_logits).item()
        del corr_logits
    print(f"      clean baseline     = {ld_clean:+.4f}  ({metric_fn.name})")
    print(f"      corrupted baseline = {ld_corr:+.4f}  ({metric_fn.name})")

    # Per-method sweeps
    print(f"[4/5] Sweeping {len(METHODS)} methods + Random over K={ks[0]}..{ks[-1]} "
          f"(step={args.k_step}, {len(ks)} points)")
    curves: Dict[str, List[float]] = {}
    for label, path in METHODS:
        path = Path(path)
        if not path.exists():
            print(f"      [WARN] missing {path} — skipping {label}")
            continue
        ranking = load_attn_ranking(path)
        print(f"      - {label}: {len(ranking)} attn heads ranked")
        curve = sweep_one_ranking(
            model, clean_tokens, corrupted_z, metric_fn, ranking, ks,
        )
        curves[label] = curve

    print(f"      - Random ({args.n_random_trials} trials)")
    rand_mean, rand_std = random_curve(
        model, clean_tokens, corrupted_z, metric_fn,
        ks, n_layers, n_heads, args.n_random_trials,
    )
    curves["Random"] = rand_mean.tolist()

    # Save raw curves
    out_json = args.out_dir / "head_faithfulness.json"
    out_json.write_text(json.dumps({
        "ks":     ks,
        "ld_clean":     ld_clean,
        "ld_corrupted": ld_corr,
        "curves":       curves,
        "random_std":   rand_std.tolist(),
        "n":            batch.n,
        "n_random_trials": args.n_random_trials,
        "task":           batch.task,
        "metric":         metric_fn.name,
    }, indent=2))
    print(f"      [saved] {out_json}")

    # Plot
    print(f"[5/5] Plotting")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for label, curve in curves.items():
        c = METHOD_COLORS.get(label, "black")
        if label == "Random":
            ax.plot(ks, curve, color=c, ls="--", lw=1.4, label=f"{label}")
            ax.fill_between(ks, np.array(curve) - rand_std,
                            np.array(curve) + rand_std,
                            color=c, alpha=0.15)
        else:
            ax.plot(ks, curve, color=c, lw=1.6, label=f"{label}")
    ax.axhline(ld_clean, color="black", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(ld_corr,  color="black", ls=":", lw=0.8, alpha=0.6)
    ax.text(0.0, ld_clean, " clean",     fontsize=9, va="bottom", alpha=0.6)
    ax.text(0.0, ld_corr,  " corrupted", fontsize=9, va="bottom", alpha=0.6)
    ax.set_xlabel("K (top-K heads patched cumulatively)")
    ax.set_ylabel(metric_fn.label)
    ax.set_title(f"Head-level faithfulness — task={batch.task} (N={batch.n})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    out_png = args.out_dir / "head_faithfulness.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      [saved] {out_png}")


if __name__ == "__main__":
    main()
