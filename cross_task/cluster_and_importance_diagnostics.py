"""
Task-clustering and importance-distribution diagnostics.

1. Flat clusters from task_linkage at several cut heights, all task pairs
   sorted by similarity, and within/cross similarity for semantic groupings.
2. Importance distributions per unit type and per-unit aggregates, used to
   choose an importance threshold tau.

Reads cross_task/run_gpt_12tasks/consistency_share/cross_task_result.pkl and
writes results/cross_task_diagnostics/cluster_and_importance_diagnostics.txt.

Usage:
    python cross_task/cluster_and_importance_diagnostics.py
"""
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster

ROOT = Path("cross_task/run_gpt_12tasks")
VARIANT = "consistency_share"
OUT_DIR = Path("results/cross_task_diagnostics")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    r = pickle.loads((ROOT / VARIANT / "cross_task_result.pkl").read_bytes())
    names    = r["task_names"]
    sim      = r["task_similarity"]
    linkage  = r["task_linkage"]
    unit_meta = r["tensor"]["unit_meta"]
    imp      = r["tensor"]["importance"]
    n_units, n_tasks = imp.shape

    out = []

    out.append("=" * 70)
    out.append("PART 1: TASK CLUSTERING")
    out.append("=" * 70)

    out.append("\nAll 66 task pairs sorted by similarity (consistency_share):")
    pairs = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            pairs.append((float(sim[i, j]), names[i], names[j]))
    pairs.sort(reverse=True)
    for v, a, b in pairs:
        out.append(f"  {v:.4f}   {a:<14} <-> {b}")

    out.append("\n--- Flat clusters from scipy.fcluster at various cut heights ---")
    for height in (0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        labels = fcluster(linkage, t=height, criterion="distance")
        groups = defaultdict(list)
        for name, lab in zip(names, labels):
            groups[int(lab)].append(name)
        out.append(f"\n  cut height={height}:  ({len(groups)} clusters)")
        for cid in sorted(groups):
            out.append(f"    C{cid}: {groups[cid]}")

    semantic = {
        "SVA (syntactic agreement)":      ["sva", "sva_blimp", "sva_isare"],
        "Numerical (greater-than)":       ["gt", "gt_price"],
        "Factual recall (entity->fact)":  ["country_cap", "person_country"],
        "Reference / pronoun":             ["ioi_strong", "person_pro", "gender_bias"],
        "Other":                           ["col_obj", "gp"],
    }
    out.append("\n--- Proposed semantic clusters (within vs cross similarity) ---")
    for label, group in semantic.items():
        in_pairs = []
        for a in group:
            for b in group:
                if a == b: continue
                in_pairs.append(sim[names.index(a), names.index(b)])
        avg_in = float(np.mean(in_pairs)) if in_pairs else float("nan")
        out_pairs = []
        for a in group:
            for b in names:
                if b in group: continue
                out_pairs.append(sim[names.index(a), names.index(b)])
        avg_out = float(np.mean(out_pairs)) if out_pairs else float("nan")
        sep = avg_in - avg_out
        out.append(f"  [{label}]   {group}")
        out.append(f"      mean within = {avg_in:.4f}   "
                   f"mean cross = {avg_out:.4f}   sep = {sep:+.4f}")

    out.append("\n" + "=" * 70)
    out.append("PART 2: IMPORTANCE DISTRIBUTION")
    out.append("=" * 70)

    by_type = defaultdict(list)
    for i, m in enumerate(unit_meta):
        by_type[m["unit_type"]].extend(imp[i].tolist())
    out.append("\n--- Importance distribution by unit_type (flattened across 12 tasks) ---")
    for t, vals in by_type.items():
        a = np.array(vals)
        out.append(f"  {t:<10}  n={len(vals):>4}  "
                   f"min={a.min():.2e}  p25={np.percentile(a,25):.2e}  "
                   f"p50={np.percentile(a,50):.2e}  p75={np.percentile(a,75):.2e}  "
                   f"max={a.max():.2e}")

    out.append("\n--- Per-unit aggregations (across 12 tasks) ---")
    per_unit = {
        "min_imp_over_tasks":     imp.min(axis=1),
        "geomean_imp_over_tasks": np.exp(np.log(np.maximum(imp, 1e-12)).mean(axis=1)),
        "mean_imp_over_tasks":    imp.mean(axis=1),
        "max_imp_over_tasks":     imp.max(axis=1),
    }
    for stat, vals in per_unit.items():
        a = np.array(vals)
        out.append(f"\n  {stat}:")
        for q in (10, 25, 50, 75, 90, 95, 99):
            out.append(f"    p{q:>2} = {np.percentile(a, q):.2e}")

    out.append("\n--- Per-unit min-importance, by unit_type ---")
    by_type_min = defaultdict(list)
    for i, m in enumerate(unit_meta):
        by_type_min[m["unit_type"]].append(float(imp[i].min()))
    for t, vals in by_type_min.items():
        a = np.array(vals)
        out.append(f"  {t:<10}  n={len(vals):>3}  "
                   f"min={a.min():.2e}  p25={np.percentile(a,25):.2e}  "
                   f"p50={np.percentile(a,50):.2e}  p75={np.percentile(a,75):.2e}  "
                   f"max={a.max():.2e}")

    attn_min = np.array([float(imp[i].min())
                         for i, m in enumerate(unit_meta)
                         if m["unit_type"] == "attn_head"])
    out.append("\n--- # attn_heads (out of 144) passing min-importance threshold tau ---")
    for tau in (1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4):
        n_pass = int((attn_min >= tau).sum())
        out.append(f"  tau = {tau:.1e}   ->  n_pass = {n_pass:>3}/144  "
                   f"({100*n_pass/144:.1f}%)")

    u = r["universality"]
    score_min  = u["min"]
    score_mean = u["mean"]
    out.append("\n--- Cross check: top 15 attn by score_min, with importance stats ---")
    attn_idxs = [i for i, m in enumerate(unit_meta) if m["unit_type"] == "attn_head"]
    sorted_min = sorted(attn_idxs, key=lambda i: -float(score_min[i]))
    out.append(f"  rank   uid   layer   score_min   imp_min    imp_mean   imp_max")
    for rank, i in enumerate(sorted_min[:15], 1):
        m = unit_meta[i]
        out.append(f"  {rank:>3}   {m['unit_id']:>3}    L{m['layer']:<3}  "
                   f"{float(score_min[i]):.4f}    "
                   f"{float(imp[i].min()):.2e}  {float(imp[i].mean()):.2e}  "
                   f"{float(imp[i].max()):.2e}")

    out.append("\n--- Cross check: top 15 attn by score_mean, with importance stats ---")
    sorted_mean = sorted(attn_idxs, key=lambda i: -float(score_mean[i]))
    out.append(f"  rank   uid   layer   score_mean  imp_min    imp_mean   imp_max")
    for rank, i in enumerate(sorted_mean[:15], 1):
        m = unit_meta[i]
        out.append(f"  {rank:>3}   {m['unit_id']:>3}    L{m['layer']:<3}  "
                   f"{float(score_mean[i]):.4f}    "
                   f"{float(imp[i].min()):.2e}  {float(imp[i].mean()):.2e}  "
                   f"{float(imp[i].max()):.2e}")

    out_path = OUT_DIR / "cluster_and_importance_diagnostics.txt"
    out_path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
