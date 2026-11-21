"""
Write enriched universal-unit tables from an existing cross_task_result.pkl.

For each importance variant under cross_task/<CT_RUN>/ (default run_gpt_12tasks),
writes universal_units_full.csv (all units, metadata and both rankings),
universal_units_top20_by_{min,mean}.csv and layer_distribution.txt.

Usage:
    python cross_task/enrich_universal_units.py
"""
import csv
import os
import pickle
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")
VARIANTS = ["consistency_share", "sigma_weighted"]


def build_rows(result):
    unit_meta = result["tensor"]["unit_meta"]
    u = result["universality"]
    n = len(unit_meta)
    rows = []
    for i, m in enumerate(unit_meta):
        rows.append({
            "idx": i,
            "unit_id": m.get("unit_id"),
            "unit_type": m.get("unit_type"),
            "layer": m.get("layer"),
            "parent_block": m.get("parent_block"),
            "importance": float(m.get("importance", float("nan"))),
            "d_k": m.get("d_k"),
            "score_min":  float(u["min"][i]),
            "score_mean": float(u["mean"][i]),
            "score_max":  float(u["max"][i]),
            "n_significant": int(u["n_significant"][i]),
        })
    by_min  = sorted(range(n), key=lambda i: -rows[i]["score_min"])
    by_mean = sorted(range(n), key=lambda i: -rows[i]["score_mean"])
    rank_by_min  = {i: r + 1 for r, i in enumerate(by_min)}
    rank_by_mean = {i: r + 1 for r, i in enumerate(by_mean)}
    for r in rows:
        r["rank_by_min"]  = rank_by_min[r["idx"]]
        r["rank_by_mean"] = rank_by_mean[r["idx"]]
    return rows


def write_full(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "unit_id", "unit_type", "layer", "parent_block",
            "importance", "d_k",
            "score_min", "score_mean", "score_max", "n_significant",
            "rank_by_min", "rank_by_mean",
        ])
        for r in rows:
            w.writerow([
                r["unit_id"], r["unit_type"], r["layer"], r["parent_block"],
                f"{r['importance']:.6e}", r["d_k"],
                f"{r['score_min']:.4f}", f"{r['score_mean']:.4f}",
                f"{r['score_max']:.4f}", r["n_significant"],
                r["rank_by_min"], r["rank_by_mean"],
            ])


def write_top(rows, path, sort_key, k=20):
    top = sorted(rows, key=lambda r: -r[sort_key])[:k]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "unit_id", "unit_type", "layer", "parent_block",
            "importance",
            "score_min", "score_mean", "score_max", "n_significant",
        ])
        for rank, r in enumerate(top, 1):
            w.writerow([
                rank, r["unit_id"], r["unit_type"], r["layer"], r["parent_block"],
                f"{r['importance']:.6e}",
                f"{r['score_min']:.4f}", f"{r['score_mean']:.4f}",
                f"{r['score_max']:.4f}", r["n_significant"],
            ])


def write_distribution(variant, rows, path):
    lines = [f"=== {variant} ===", f"Total units: {len(rows)}"]
    type_c  = Counter(r["unit_type"] for r in rows)
    layer_c = Counter(r["layer"] for r in rows)
    lines.append(f"Type breakdown: {dict(type_c)}")
    lines.append(f"Layer breakdown (all units): {sorted(layer_c.items())}")
    for k in (10, 20, 50):
        for sort_key in ("score_min", "score_mean"):
            top = sorted(rows, key=lambda r: -r[sort_key])[:k]
            tc = Counter(r["unit_type"] for r in top)
            lc = Counter(r["layer"] for r in top)
            lines.append(f"\nTop-{k} by {sort_key}:")
            lines.append(f"  type:           {dict(tc)}")
            lines.append(f"  layer hist:     {sorted(lc.items())}")
            lines.append(f"  unit_ids:       {[r['unit_id'] for r in top]}")
            lines.append(f"  (id, layer, type, score_min, score_mean):")
            for r in top:
                lines.append(
                    f"    {r['unit_id']:>4}  L{r['layer']:<2}  {r['unit_type']:<10}  "
                    f"min={r['score_min']:.3f}  mean={r['score_mean']:.3f}"
                )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def main():
    for variant in VARIANTS:
        vdir = ROOT / variant
        pkl  = vdir / "cross_task_result.pkl"
        if not pkl.exists():
            print(f"[skip] {pkl} not found")
            continue
        result = pickle.loads(pkl.read_bytes())
        rows = build_rows(result)

        write_full(rows, vdir / "universal_units_full.csv")
        write_top(rows, vdir / "universal_units_top20_by_min.csv",  "score_min")
        write_top(rows, vdir / "universal_units_top20_by_mean.csv", "score_mean")
        lines = write_distribution(variant, rows, vdir / "layer_distribution.txt")

        print("\n".join(lines))
        print(f"\n[wrote] {vdir}/universal_units_full.csv")
        print(f"[wrote] {vdir}/universal_units_top20_by_min.csv")
        print(f"[wrote] {vdir}/universal_units_top20_by_mean.csv")
        print(f"[wrote] {vdir}/layer_distribution.txt\n")


if __name__ == "__main__":
    main()