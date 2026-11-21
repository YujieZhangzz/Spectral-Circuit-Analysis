"""Rank correlation (Spearman and Pearson) between the cross-task universality
scores (score_min, score_mean, score_max, n_significant) and per-unit
importance, to check whether the cross-task scores carry signal beyond raw
importance.

Reads cross_task/run_gpt_12tasks/<variant>/cross_task_result.pkl and writes
<variant>/rank_vs_importance.txt.

Usage: python cross_task/rank_correlation.py
"""
import pickle
from pathlib import Path

ROOT = Path("cross_task/run_gpt_12tasks")
VARIANTS = ["consistency_share", "sigma_weighted"]


def spearman(x, y):
    """Plain Spearman: Pearson on rank-transforms."""
    n = len(x)
    rx = _ranks(x)
    ry = _ranks(y)
    return _pearson(rx, ry)


def _ranks(values):
    # average rank for ties
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = (sum((xi - mx) ** 2 for xi in x)) ** 0.5
    dy = (sum((yi - my) ** 2 for yi in y)) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def evaluate(variant):
    vdir = ROOT / variant
    result = pickle.loads((vdir / "cross_task_result.pkl").read_bytes())
    unit_meta = result["tensor"]["unit_meta"]
    u = result["universality"]

    # Three slices: all, attn-only, mlp-only
    slices = {
        "all (168)":  list(range(len(unit_meta))),
        "attn (144)": [i for i, m in enumerate(unit_meta) if m["unit_type"] == "attn_head"],
        "mlp  (24)":  [i for i, m in enumerate(unit_meta) if m["unit_type"] != "attn_head"],
    }

    metrics = {
        "score_min":     [float(u["min"][i])  for i in range(len(unit_meta))],
        "score_mean":    [float(u["mean"][i]) for i in range(len(unit_meta))],
        "score_max":     [float(u["max"][i])  for i in range(len(unit_meta))],
        "n_significant": [int(u["n_significant"][i]) for i in range(len(unit_meta))],
    }
    importance = [float(unit_meta[i].get("importance", 0.0)) for i in range(len(unit_meta))]

    lines = [f"=== {variant} ===", ""]
    for slice_name, idxs in slices.items():
        lines.append(f"--- slice: {slice_name} ---")
        imp_slice = [importance[i] for i in idxs]
        for mname, mvals in metrics.items():
            slice_vals = [mvals[i] for i in idxs]
            sp = spearman(slice_vals, imp_slice)
            pe = _pearson(slice_vals, imp_slice)
            lines.append(f"  {mname:<14} vs importance:  Spearman={sp:+.4f}   Pearson={pe:+.4f}")
        # Pairwise correlations between the cross-task metrics
        lines.append("  -- pairwise between cross-task metrics --")
        names = list(metrics.keys())
        for a_idx, a in enumerate(names):
            for b in names[a_idx + 1:]:
                va = [metrics[a][i] for i in idxs]
                vb = [metrics[b][i] for i in idxs]
                sp = spearman(va, vb)
                lines.append(f"  {a:<14} vs {b:<14}  Spearman={sp:+.4f}")
        lines.append("")
    return lines


def main():
    all_lines = []
    for v in VARIANTS:
        if not (ROOT / v / "cross_task_result.pkl").exists():
            continue
        lines = evaluate(v)
        all_lines.extend(lines)
        (ROOT / v / "rank_vs_importance.txt").write_text("\n".join(lines) + "\n")
        print(f"[wrote] {ROOT / v}/rank_vs_importance.txt")
    print()
    print("\n".join(all_lines))


if __name__ == "__main__":
    main()