"""Evaluate a ranked_units CSV against the IOI circuit of Wang et al. (26 heads).

On the attention-only ranking, reports precision@K and recall@K
(K in {10, 20, 26, 30, 50}), mean average precision, and the rank of each
ground-truth head.

Usage:
    python eval_recall_at_k.py results/gpt2_small_ioi_strong_N50_M10/ranked_units.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Set


IOI_GT: Set[str] = {
    # Previous Token Heads
    "L2H2", "L4H11",
    # Duplicate Token Heads
    "L0H1", "L0H10", "L3H0",
    # Induction Heads
    "L5H5", "L5H8", "L5H9", "L6H9",
    # S-Inhibition Heads
    "L7H3", "L7H9", "L8H6", "L8H10",
    # Name Mover Heads
    "L9H6", "L9H9", "L10H0",
    # Negative Name Mover Heads
    "L10H7", "L11H10",
    # Backup Name Mover Heads
    "L9H0", "L9H7", "L10H1", "L10H2", "L10H6", "L10H10", "L11H2", "L11H9",
}
N_GT = len(IOI_GT)   # 26
DEFAULT_KS = [10, 20, 26, 30, 50]


def load_ranking(csv_path: Path) -> List[dict]:
    """Read ranked_units CSV, return list of row dicts sorted by rank ascending."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["rank"]))
    return rows


def evaluate(rows: List[dict], gt: Set[str], ks: List[int]) -> dict:
    """
    Filter to attn_head, then compute precision@K, recall@K, and mAP against gt.
    Assumes rows already sorted by ascending rank (rank 0 = top).
    """
    attn_rows = [r for r in rows if r["unit_type"] == "attn_head"]
    n_attn = len(attn_rows)
    n_gt   = len(gt)

    # precision@K and recall@K
    at_k = {}
    for k in ks:
        topk = {r["label"] for r in attn_rows[:k]}
        hits = topk & gt
        n_hits = len(hits)
        at_k[k] = {
            "precision": n_hits / k        if k > 0     else 0.0,
            "recall":    n_hits / n_gt     if n_gt > 0  else 0.0,
            "n_hits":    n_hits,
            "n_gt":      n_gt,
            "hits":      sorted(hits),
        }

    # mAP: walk attn_rows, accumulate precision at each GT hit; missed GT contributes 0
    cum_hits = 0
    aps = []
    label_to_rank: dict = {}
    for i, r in enumerate(attn_rows, start=1):
        if r["label"] in gt:
            cum_hits += 1
            aps.append(cum_hits / i)
            label_to_rank[r["label"]] = i
    aps += [0.0] * (n_gt - cum_hits)
    map_score = sum(aps) / n_gt if n_gt > 0 else 0.0

    gt_positions = [
        {"label": label, "attn_rank": label_to_rank.get(label, None)}
        for label in sorted(gt)
    ]

    return {
        "n_total":      len(rows),
        "n_attn":       n_attn,
        "n_gt":         n_gt,
        "at_k":         at_k,
        "map":          map_score,
        "gt_positions": gt_positions,
    }


def fmt_report(name: str, res: dict) -> str:
    lines = [f"\n=== {name} ===",
             f"  total units in ranking: {res['n_total']}  (attn={res['n_attn']})",
             f"  ground-truth size:      {res['n_gt']}",
             ""]
    lines.append(f"  {'K':>4}  {'precision':>10}  {'recall':>8}  {'hits':>7}  hit list")
    for k, r in res["at_k"].items():
        hit_str = ", ".join(r["hits"]) if r["hits"] else "-"
        lines.append(
            f"  {k:>4}  {r['precision']:>9.1%}   {r['recall']:>7.1%}  "
            f"{r['n_hits']:>3}/{r['n_gt']}    {hit_str[:80]}"
        )
    lines.append("")
    lines.append(f"  mAP: {res['map']:.4f}")
    lines.append("")
    lines.append("  GT head positions in attn-only ranking (1-indexed; - = not in ranking):")
    for entry in res["gt_positions"]:
        rk = entry["attn_rank"]
        lines.append(f"    {entry['label']:>6}  ->  "
                     + (f"#{rk}" if rk is not None else "-"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path,
                    help="Path to ranked_units.csv (or ranked_units_direct.csv)")
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                    help=f"K values for recall@K (default: {DEFAULT_KS})")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path prefix. Writes <out>.txt and <out>.json. "
                         "Default: <csv_path's dir>/<csv_stem>_recall")
    ap.add_argument("--no-save", action="store_true",
                    help="Print only, do not write any files.")
    args = ap.parse_args()

    if not args.csv_path.exists():
        sys.exit(f"[ERR] file not found: {args.csv_path}")

    rows   = load_ranking(args.csv_path)
    res    = evaluate(rows, IOI_GT, args.ks)
    report = fmt_report(str(args.csv_path), res)
    print(report)

    if not args.no_save:
        out = args.out if args.out is not None else \
              args.csv_path.parent / f"{args.csv_path.stem}_recall"
        out.parent.mkdir(parents=True, exist_ok=True)
        txt_path  = out.with_suffix(".txt")
        json_path = out.with_suffix(".json")
        txt_path.write_text(report + "\n")
        json_path.write_text(json.dumps(
            {"csv_path": str(args.csv_path), "ks": args.ks, "metrics": res},
            indent=2, ensure_ascii=False,
        ))
        print(f"\n[saved] {txt_path}\n[saved] {json_path}")


if __name__ == "__main__":
    main()
