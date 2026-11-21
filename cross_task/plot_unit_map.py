"""Layer x position map of all 168 units, coloured by classification
(universal / family / task-specific / mixed).

Reads <variant>/classify/v1_mean.csv and, if present,
<variant>/task_specific_units.csv under cross_task/<CT_RUN>/; writes
<variant>/unit_map.{pdf,png}.

Usage:
    python cross_task/plot_unit_map.py                    # consistency_share
    python cross_task/plot_unit_map.py consistency_share
"""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from utils.paper_style import paper_rc

ROOT = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")

# Family display names and colors
FAM_NAME = {
    "GT":      "Numeric Comparison",
    "SVA":     "Subject-Verb Agreement",
    "COUNTRY": "Factual Recall",
    "PRONOUN": "Gendered Coreference",
}
FAM_COLOR = {
    "GT":      "#3D5A80",
    "SVA":     "#E08545",
    "COUNTRY": "#4FA68D",
    "PRONOUN": "#A678B5",
}
UNIVERSAL_COLOR = "#1A2845"
TASK_COLOR      = "#C0436E"
MIXED_COLOR     = "#DDDDDD"
EMPTY_COLOR     = "#FFFFFF"

N_LAYERS = 12
N_INTRA  = 14   # 12 attn heads + mlp_gate + mlp_down

INTRA_LABELS = [f"H{i}" for i in range(12)] + ["MLP\ngate", "MLP\ndown"]

# Short tags for task-specific cell annotations
TASK_TAG = {
    "ioi_strong":     "IOI",
    "country_cap":    "Cap",
    "person_pro":     "Pro",
    "gt":             "GT",
    "gt_price":       "GT$",
    "person_country": "PCo",
    "col_obj":        "CO",
    "gender_bias":    "GB",
    "gp":             "GP",
    "sva":            "SV",
    "sva_blimp":      "SVb",
    "sva_isare":      "SVi",
}


def load_classification(variant_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(variant_dir / "classify" / "v1_mean.csv")
    ts_path = variant_dir / "task_specific_units.csv"
    df["final"]      = df["primary_label"]
    df["task_name"]  = ""
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        ts_map = dict(zip(ts["unit_id"], ts["task"]))
        for uid, task in ts_map.items():
            df.loc[df.unit_id == uid, "final"]     = "task"
            df.loc[df.unit_id == uid, "task_name"] = task
    return df


def cell_color(label: str) -> str:
    if label == "universal":
        return UNIVERSAL_COLOR
    if label == "task":
        return TASK_COLOR
    if label == "mixed":
        return MIXED_COLOR
    if label.startswith("family_"):
        fam = label.split("_", 1)[1]
        return FAM_COLOR.get(fam, "#888888")
    return EMPTY_COLOR


def plot_one(variant_dir: Path, out_base: Path) -> None:
    df = load_classification(variant_dir)

    grid_label = np.full((N_LAYERS, N_INTRA), "", dtype=object)
    grid_task  = np.full((N_LAYERS, N_INTRA), "", dtype=object)
    for _, r in df.iterrows():
        L = int(r["layer"])
        intra = int(r["unit_id"]) - L * N_INTRA
        if 0 <= intra < N_INTRA and 0 <= L < N_LAYERS:
            grid_label[L, intra] = r["final"]
            grid_task[L, intra]  = r["task_name"]

    rc = paper_rc("double")
    rc["axes.spines.top"]    = False
    rc["axes.spines.right"]  = False
    rc["axes.spines.left"]   = False
    rc["axes.spines.bottom"] = False
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10
    rc["axes.labelsize"]  = 11
    rc["legend.fontsize"] = 10
    rc["font.size"]       = 10

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(9.0, 6.4))

        for L in range(N_LAYERS):
            for i in range(N_INTRA):
                lbl = grid_label[L, i]
                ax.add_patch(Rectangle(
                    (i, L), 1, 1,
                    facecolor=cell_color(lbl), edgecolor="white", linewidth=0.7,
                ))
                if lbl == "task":
                    tag = TASK_TAG.get(grid_task[L, i], grid_task[L, i][:5])
                    ax.text(i + 0.5, L + 0.5, tag,
                            ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")

        ax.set_xlim(0, N_INTRA)
        ax.set_ylim(0, N_LAYERS)
        ax.set_aspect("equal")

        # Separator between attention heads (cols 0-11) and MLP slots
        ax.axvline(12, color="#222", lw=0.9, ls=":")

        ax.set_xticks(np.arange(N_INTRA) + 0.5)
        ax.set_xticklabels(INTRA_LABELS)
        ax.set_yticks(np.arange(N_LAYERS) + 0.5)
        ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
        ax.set_xlabel("Unit position")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="both", length=0, pad=2)

        # Legend: universal, 4 families, task-specific, mixed
        handles = [Patch(facecolor=UNIVERSAL_COLOR, label="Universal")]
        for fam in ("GT", "SVA", "COUNTRY", "PRONOUN"):
            handles.append(
                Patch(facecolor=FAM_COLOR[fam], label=FAM_NAME[fam])
            )
        handles.append(Patch(facecolor=TASK_COLOR, label="Task-specific"))
        handles.append(Patch(facecolor=MIXED_COLOR, label="Mixed"))

        ax.legend(
            handles=handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False,
            handletextpad=0.5, columnspacing=1.5,
        )

        plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_base.with_suffix('.pdf')}")
        print(f"[saved] {out_base.with_suffix('.png')}")


def main() -> None:
    variants = sys.argv[1:] or ["consistency_share"]
    for v in variants:
        d = ROOT / v
        if not d.exists():
            print(f"[skip] {d} not found")
            continue
        plot_one(d, d / "unit_map")


if __name__ == "__main__":
    main()
