"""
Two raw-gradient baselines (no SVD), one score per unit:

    magnitude(u) = mean_n ‖g_n‖²                            (signal strength)
    direction(u) = ‖mean_n g_n‖² / mean_n ‖g_n‖²    ∈[0,1]  (directional concentration)

Both are ranked in descending order. Magnitude favours noisy heads and
direction favours near-dead heads, and the two pick largely disjoint units.

Outputs (results_dir/variance_baseline/):
    unit_metrics.csv                 one row per unit, both metrics
    ranked_units_magnitude.csv       ranking by magnitude, descending
    ranked_units_direction.csv       ranking by direction,  descending
                                      (same schema as ranked_units.csv, for eval_recall_at_k.py)

Usage:
    python -m spectral.variance_baseline \
        --results_dir results/<experiment_name>
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import List, Dict

import torch

from utils.model_utils import load_model_and_tokenizer, get_param_filter, count_params
from spectral.block_finder import detect_arch, build_unit_layout


METRICS = ["magnitude", "direction"]


def compute_unit_metrics(grad_avg: torch.Tensor, unit_layout) -> List[Dict]:
    rows: List[Dict] = []
    for u in unit_layout:
        idx = u.get_indices()
        G = grad_avg.index_select(1, idx).float()           # (N, d_u)
        d_u = G.shape[1]

        magnitude = float((G * G).sum(dim=1).mean().item())  # mean_n ‖g_n‖²
        mean_g    = G.mean(dim=0)                            # (d_u,)
        signal_sq = float((mean_g * mean_g).sum().item())    # ‖mean_n g_n‖²
        direction = signal_sq / (magnitude + 1e-30)          # ∈ [0, 1]

        rows.append({
            "unit_id":   u.unit_id,
            "label":     u.label,
            "unit_type": u.unit_type,
            "layer":     u.layer,
            "d_u":       d_u,
            "magnitude": magnitude,
            "direction": direction,
        })
        del G, mean_g
    rows.sort(key=lambda r: r["unit_id"])
    return rows


def write_full_csv(rows: List[Dict], path: Path) -> None:
    cols = ["unit_id", "label", "unit_type", "layer", "d_u"] + METRICS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def write_ranked_csv(rows: List[Dict], metric: str, path: Path) -> None:
    """Same schema as results/.../ranked_units.csv, for eval_recall_at_k.py."""
    ordered = sorted(rows, key=lambda r: r[metric], reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "unit_id", "label", "unit_type", "layer",
                    "importance", "rho"])
        for rank, r in enumerate(ordered):
            w.writerow([rank, r["unit_id"], r["label"], r["unit_type"],
                        r["layer"], f"{r[metric]:.6g}", ""])


def print_topk(rows: List[Dict], metric: str, k: int = 30) -> None:
    ordered = sorted(rows, key=lambda r: r[metric], reverse=True)[:k]
    print(f"\n  Top-{k} by {metric}:")
    print(f"    {'rank':>4}  {'label':<14}  {'type':<10}  {'layer':>5}  "
          f"{'d_u':>10}  {metric:>14}")
    for i, r in enumerate(ordered):
        print(f"    {i:>4}  {r['label']:<14}  {r['unit_type']:<10}  "
              f"{r['layer']:>5}  {r['d_u']:>10,}  {r[metric]:>14.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path,
                    default=Path("results/gpt2_small_ioi_strong_N100_M5"))
    ap.add_argument("--model",       default="gpt2")
    ap.add_argument("--dtype",       default="bfloat16")
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--scope",       default="attn_VO_mlp")
    ap.add_argument("--include_mlp", action="store_true", default=True)
    ap.add_argument("--out_subdir",  default="variance_baseline")
    args = ap.parse_args()

    grad_path = args.results_dir / "grad_avg.pt"
    if not grad_path.exists():
        raise SystemExit(f"[ERR] not found: {grad_path}")
    out_dir = args.results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading grad_avg from {grad_path}")
    grad_avg = torch.load(grad_path, map_location="cpu", weights_only=False)
    print(f"      shape={tuple(grad_avg.shape)} dtype={grad_avg.dtype}")
    N, D = grad_avg.shape

    print(f"[2/3] Rebuilding unit layout from {args.model!r}")
    model, _ = load_model_and_tokenizer(args.model, args.dtype, args.device)
    arch = detect_arch(model)
    n_layers = getattr(model.config, "num_hidden_layers",
                       getattr(model.config, "n_layer", None))
    param_filter = get_param_filter(args.scope, n_layers)
    pinfo = count_params(model, param_filter)
    if pinfo["total_D"] != D:
        raise SystemExit(
            f"[ERR] D mismatch: model+scope D={pinfo['total_D']:,}, "
            f"grad_avg has D={D:,}")
    unit_layout = build_unit_layout(
        model, param_filter=param_filter, arch=arch, include_mlp=args.include_mlp,
    )
    del model
    torch.cuda.empty_cache()

    print(f"[3/3] Computing per-unit metrics  ({len(unit_layout)} units, N={N})")
    rows = compute_unit_metrics(grad_avg, unit_layout)

    write_full_csv(rows, out_dir / "unit_metrics.csv")
    print(f"      [saved] unit_metrics.csv")
    for m in METRICS:
        write_ranked_csv(rows, m, out_dir / f"ranked_units_{m}.csv")
        print(f"      [saved] ranked_units_{m}.csv")

    print("\n" + "=" * 60)
    print(f"  variance baseline summary  (results_dir={args.results_dir})")
    print("=" * 60)
    print(f"  units: {len(rows)}  "
          f"(attn={sum(1 for r in rows if r['unit_type']=='attn_head')}, "
          f"mlp={sum(1 for r in rows if r['unit_type']!='attn_head')})")
    for m in METRICS:
        print_topk(rows, m, k=30)

    print("\n  Ranked CSVs written. To compute recall@K, run eval_recall_at_k.py "
          "on a ranking whose task has a ground-truth circuit (currently only IOI):")
    for m in METRICS:
        print(f"      python eval_recall_at_k.py "
              f"{out_dir}/ranked_units_{m}.csv")


if __name__ == "__main__":
    main()
