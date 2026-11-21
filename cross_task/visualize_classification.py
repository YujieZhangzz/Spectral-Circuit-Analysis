"""Annotate the cross-task classification CSVs with standard unit names and
IOI circuit roles (Wang et al., 2022), and plot per-layer stacked bars and
category pie charts.

For each variant and tag (v1_min, v1_mean, ...) under
cross_task/run_gpt_12tasks/<variant>/classify/, reads <tag>.csv and writes
<tag>_annotated.csv, <tag>_stacked_bar.{pdf,png} and <tag>_pie.{pdf,png}.

unit_id layout: layer = unit_id // 14; unit_id % 14 is the head index
(0-11), 12 = mlp_gate, 13 = mlp_down.

Usage: python cross_task/visualize_classification.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.paper_style import paper_rc, paper_size

# Wang et al. (2022) IOI circuit (26 attn heads)
IOI_CANONICAL: dict[tuple[int, int], str] = {
    # Duplicate Token
    (0, 1): "Duplicate Token", (0, 10): "Duplicate Token", (3, 0): "Duplicate Token",
    # Previous Token
    (2, 2): "Previous Token",  (4, 11): "Previous Token",
    # Induction
    (5, 5): "Induction", (5, 8): "Induction", (5, 9): "Induction", (6, 9): "Induction",
    # S-Inhibition
    (7, 3): "S-Inhibition", (7, 9): "S-Inhibition",
    (8, 6): "S-Inhibition", (8, 10): "S-Inhibition",
    # Name Movers
    (9, 6): "Name Mover", (9, 9): "Name Mover", (10, 0): "Name Mover",
    # Negative Name Movers
    (10, 7): "Negative Name Mover", (11, 10): "Negative Name Mover",
    # Backup Name Movers
    (9, 0): "Backup Name Mover", (9, 7): "Backup Name Mover",
    (10, 1): "Backup Name Mover", (10, 2): "Backup Name Mover",
    (10, 6): "Backup Name Mover", (10, 10): "Backup Name Mover",
    (11, 2): "Backup Name Mover", (11, 9): "Backup Name Mover",
}


def unit_id_to_name(uid: int) -> tuple[str, int, int]:
    """unit_id -> (standard_name, layer, head or -1)."""
    layer = uid // 14
    intra = uid %  14
    if intra < 12:
        return f"L{layer}H{intra}", layer, intra
    elif intra == 12:
        return f"L{layer}.mlp_gate", layer, -1
    else:
        return f"L{layer}.mlp_down", layer, -1


# Category colors
CAT_ORDER = [
    "universal",
    "family_GT_specialized", "family_GT_broad",
    "family_SVA_specialized", "family_SVA_broad",
    "family_COUNTRY_broad",
    "family_PRONOUN_broad",     # may be empty in min
    "family_IOI_COL_broad",     # ditto
    "task_specific",
    "mixed",
]
CAT_COLORS = {
    "universal":              "#1f3a93",   # deep navy
    "family_GT_specialized":  "#c0392b",   # deep red
    "family_GT_broad":        "#f5b7b1",   # pale red
    "family_SVA_specialized": "#1e8449",   # deep green
    "family_SVA_broad":       "#abebc6",   # pale green
    "family_COUNTRY_broad":   "#ca6f1e",   # orange
    "family_PRONOUN_broad":   "#7d6608",   # olive
    "family_IOI_COL_broad":   "#5d6d7e",   # slate
    "task_specific":          "#7d3c98",   # purple
    "mixed":                  "#bdc3c7",   # light gray
}
CAT_DISPLAY = {
    "universal":              "Universal",
    "family_GT_specialized":  "GT specialized",
    "family_GT_broad":        "GT broad",
    "family_SVA_specialized": "SVA specialized",
    "family_SVA_broad":       "SVA broad",
    "family_COUNTRY_broad":   "COUNTRY broad",
    "family_PRONOUN_broad":   "PRONOUN broad",
    "family_IOI_COL_broad":   "IOI_COL broad",
    "task_specific":          "Task-specific",
    "mixed":                  "Mixed",
}


def derive_category_key(category: str, primary_label: str, family_tier: str) -> str:
    """Map (category, primary_label, family_tier) -> a single CAT_ORDER key."""
    if category == "universal":
        return "universal"
    if category == "task":
        return "task_specific"
    if category == "mixed":
        return "mixed"
    if category == "family":
        fname = primary_label.split("_", 1)[1]      # "GT", "SVA", ...
        return f"family_{fname}_{family_tier}"
    return "mixed"


def annotate(in_csv: Path, out_csv: Path) -> list[dict]:
    rows = list(csv.DictReader(open(in_csv)))
    extra_cols = ["standard_name", "ioi_role", "category_key"]
    fieldnames = list(rows[0].keys()) + extra_cols
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            uid = int(r["unit_id"])
            name, layer, head = unit_id_to_name(uid)
            ioi_role = IOI_CANONICAL.get((layer, head), "") if head >= 0 else ""
            cat_key = derive_category_key(r["category"], r["primary_label"], r["family_tier"])
            r["standard_name"] = name
            r["ioi_role"]      = ioi_role
            r["category_key"]  = cat_key
            w.writerow(r)
    return rows


def plot_stacked_bar(rows: list[dict], out_base: Path):
    layers = list(range(12))
    by_layer_cat = {l: {c: 0 for c in CAT_ORDER} for l in layers}
    for r in rows:
        layer = int(r["layer"])
        cat = derive_category_key(r["category"], r["primary_label"], r["family_tier"])
        by_layer_cat[layer][cat] += 1

    # Drop empty categories from the legend
    used_cats = [c for c in CAT_ORDER
                 if any(by_layer_cat[l][c] > 0 for l in layers)]

    with plt.rc_context(paper_rc("double")):
        _, ax = plt.subplots(figsize=paper_size("double", aspect=0.55))
        bottom = np.zeros(len(layers))
        for cat in used_cats:
            heights = np.array([by_layer_cat[l][cat] for l in layers])
            ax.bar(layers, heights, bottom=bottom,
                   color=CAT_COLORS[cat], label=CAT_DISPLAY[cat],
                   edgecolor="white", linewidth=0.4)
            bottom += heights
        ax.set_xlabel("Layer")
        ax.set_ylabel("Number of units")
        ax.set_xticks(layers)
        ax.set_xlim(-0.5, 11.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=4, fontsize=7, frameon=False)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(out_base.with_suffix(f".{ext}"),
                        bbox_inches="tight", dpi=300 if ext == "png" else None)
        plt.close()
    print(f"  [saved] {out_base.with_suffix('.pdf')}")


def plot_pie(rows: list[dict], out_base: Path):
    counts = {c: 0 for c in CAT_ORDER}
    for r in rows:
        cat = derive_category_key(r["category"], r["primary_label"], r["family_tier"])
        counts[cat] += 1
    used = [(c, counts[c]) for c in CAT_ORDER if counts[c] > 0]
    labels = [f"{CAT_DISPLAY[c]}  ({n})" for c, n in used]
    sizes  = [n for _, n in used]
    colors = [CAT_COLORS[c] for c, _ in used]

    with plt.rc_context(paper_rc("single")):
        _, ax = plt.subplots(figsize=paper_size("single", aspect=1.0))
        wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                            wedgeprops=dict(edgecolor="white", linewidth=0.7))
        ax.legend(wedges, labels, loc="center left",
                  bbox_to_anchor=(1.05, 0.5), fontsize=7, frameon=False)
        ax.set_aspect("equal")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(out_base.with_suffix(f".{ext}"),
                        bbox_inches="tight", dpi=300 if ext == "png" else None)
        plt.close()
    print(f"  [saved] {out_base.with_suffix('.pdf')}")


def main():
    base = REPO_ROOT / "cross_task" / "run_gpt_12tasks"
    for variant in ("consistency_share", "sigma_weighted"):
        for tag in ("v1_min", "v1_mean", "v2_min", "v2_mean"):
            in_csv  = base / variant / "classify" / f"{tag}.csv"
            if not in_csv.exists():
                print(f"[skip] {in_csv}")
                continue
            out_csv = base / variant / "classify" / f"{tag}_annotated.csv"
            print(f"\n[{variant} / {tag}]")
            rows = annotate(in_csv, out_csv)
            print(f"  [saved] {out_csv}")

            stacked = base / variant / "classify" / f"{tag}_stacked_bar"
            piecake = base / variant / "classify" / f"{tag}_pie"
            plot_stacked_bar(rows, stacked)
            plot_pie(rows, piecake)

    # Print top universal / specialized heads with IOI overlap
    print("\n" + "="*72)
    print("  Specialized heads vs Wang IOI canonical (consistency_share v1_min)")
    print("="*72)
    rows = list(csv.DictReader(open(
        base / "consistency_share" / "classify" / "v1_min_annotated.csv")))
    for cat_filter, title in [
        ("universal", "Universal (8)"),
        ("family_GT_specialized", "GT specialized (22)"),
        ("family_SVA_specialized", "SVA specialized (4)"),
    ]:
        sub = [r for r in rows if r["category_key"] == cat_filter]
        n_ioi = sum(1 for r in sub if r["ioi_role"])
        print(f"\n{title}  ({n_ioi}/{len(sub)} on Wang's IOI list)")
        for r in sorted(sub, key=lambda x: int(x["unit_id"])):
            tag = f"  ({r['ioi_role']})" if r["ioi_role"] else ""
            print(f"  uid={int(r['unit_id']):>3}  {r['standard_name']:<14s}{tag}")


if __name__ == "__main__":
    main()
