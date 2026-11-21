"""
Compare aggregations of per-task importance across the 12 tasks.

unit_meta['importance'] holds task 0 (col_obj) importance only. This script
recomputes mean / geomean / max / min / sum aggregations, prints the top-10
attention heads and their layer histogram for each, and computes IOI
recall@K against the 26 canonical heads.

Writes cross_task/run_gpt_12tasks/consistency_share/importance_aggregation_recall.csv.
Usage:
    python cross_task/importance_aggregation_check.py
"""
import csv
import pickle
from pathlib import Path

ROOT = Path("cross_task/run_gpt_12tasks")
VARIANT = "consistency_share"  # both variants share unit_meta and tensor

# 26 canonical IOI heads (Wang et al.)
IOI_HEADS = [
    (2, 2), (4, 11),
    (0, 1), (0, 10), (3, 0),
    (5, 5), (5, 8), (5, 9), (6, 9),
    (7, 3), (7, 9), (8, 6), (8, 10),
    (9, 6), (9, 9), (10, 0),
    (10, 7), (11, 10),
    (9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9),
]
IOI_GT = {14 * L + h for L, h in IOI_HEADS}


def label(uid):
    L, rem = divmod(uid, 14)
    return f"L{L}H{rem}" if rem < 12 else f"L{L}_MLP"


def main():
    pkl = ROOT / VARIANT / "cross_task_result.pkl"
    r = pickle.loads(pkl.read_bytes())
    unit_meta = r["tensor"]["unit_meta"]
    imp_per_task = r["tensor"]["importance"]   # shape (168, 12)
    n_units, n_tasks = imp_per_task.shape

    # Check: task 0 (col_obj) importance should match unit_meta['importance']
    uid_to_meta_imp = [m["importance"] for m in unit_meta]
    diffs = [abs(uid_to_meta_imp[i] - float(imp_per_task[i, 0])) for i in range(n_units)]
    print(f"Sanity: max |unit_meta.importance - imp_per_task[:,0]| = {max(diffs):.3e}")
    print(f"  (~0 means unit_meta importance comes from task 0 only)\n")

    import math

    def agg_mean(row):
        return float(sum(row) / len(row))

    def agg_geomean(row):
        # robust to zeros
        return float(math.exp(sum(math.log(max(v, 1e-12)) for v in row) / len(row)))

    def agg_max(row):
        return float(max(row))

    def agg_min(row):
        return float(min(row))

    def agg_sum(row):
        return float(sum(row))

    aggs = {
        "task0_only (current bug)": [float(imp_per_task[i, 0]) for i in range(n_units)],
        "mean_over_tasks":   [agg_mean(imp_per_task[i].tolist())    for i in range(n_units)],
        "geomean":           [agg_geomean(imp_per_task[i].tolist()) for i in range(n_units)],
        "max_over_tasks":    [agg_max(imp_per_task[i].tolist())     for i in range(n_units)],
        "min_over_tasks":    [agg_min(imp_per_task[i].tolist())     for i in range(n_units)],
        "sum_over_tasks":    [agg_sum(imp_per_task[i].tolist())     for i in range(n_units)],
    }

    # Filter to attn_head only for IOI recall
    attn_idxs = [i for i, m in enumerate(unit_meta) if m["unit_type"] == "attn_head"]

    K_LIST = [10, 20, 26, 30, 50]
    rows_csv = []

    for agg_name, vals in aggs.items():
        ranked = sorted(attn_idxs, key=lambda i: -vals[i])
        ranked_uids = [unit_meta[i]["unit_id"] for i in ranked]
        # Layer histogram of top-10
        from collections import Counter
        top10_layers = [unit_meta[i]["layer"] for i in ranked[:10]]
        layer_hist = sorted(Counter(top10_layers).items())
        print(f"--- {agg_name} ---")
        print(f"  top-10 unit_ids:  {ranked_uids[:10]}")
        print(f"  top-10 labels:    {[label(u) for u in ranked_uids[:10]]}")
        print(f"  top-10 layer hist: {layer_hist}")
        for K in K_LIST:
            hits = len(set(ranked_uids[:K]) & IOI_GT)
            recall = hits / 26
            base   = K / 144
            lift   = recall / base if base > 0 else float("nan")
            print(f"  recall@{K:>2}: {recall:.3f} ({hits}/26)  base={base:.3f}  lift={lift:.2f}x")
            rows_csv.append({
                "aggregation": agg_name,
                "K": K,
                "recall": f"{recall:.4f}",
                "n_hits": hits,
                "baseline": f"{base:.4f}",
                "lift": f"{lift:.3f}",
            })
        print()

    out = ROOT / VARIANT / "importance_aggregation_recall.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_csv[0].keys())
        w.writeheader()
        w.writerows(rows_csv)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()