"""Circuit-style diagram of unit classifications.

One horizontal track per layer: 12 attention heads (circles) and 2 MLP slots
(squares), coloured by category. Units whose knockout set passes the |z| bar
are outlined. Reads <variant>/classify_weighted/v1_min.csv and, if present,
<variant>/knockout_validation_v1_typematched/lift_matrix.csv under
cross_task/<CT_RUN>/; writes <variant>/unit_circuit.{pdf,png}.

Usage:
    python cross_task/plot_unit_circuit.py                    # consistency_share
    python cross_task/plot_unit_circuit.py consistency_share
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
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.lines import Line2D

from utils.paper_style import paper_rc

ROOT = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")

FAM_NAME = {
    "GT":      "Numeric Comparison",
    "SVA":     "Subject-Verb Agreement",
    "COUNTRY": "Factual Recall",
    "PRONOUN": "Gendered Coreference",
}
FAM_COLOR = {
    "GT":      "#3D6FA3",  # light slate blue
    "SVA":     "#6B9E4F",  # forest green
    "COUNTRY": "#4FA68D",  # sage teal
    "PRONOUN": "#A678B5",  # soft purple
}
UNIVERSAL_COLOR = "#5C2C6E"
TASK_COLOR      = "#C0436E"
MIXED_COLOR     = "#E2E2E2"

N_LAYERS = 12
N_INTRA  = 14

TASK_TAG = {
    "ioi_strong":     "IOI",
    "country_cap":    "CC",   # Country-Capital
    "person_pro":     "PP",   # Person-Profession
    "gt":             "GT",   # Greater-Than
    "gt_price":       "GT$",
    "person_country": "PC",   # Person-Country
    "col_obj":        "CO",   # Colored-Objects
    "gender_bias":    "GB",
    "gp":             "GP",
    "sva":            "SV",
    "sva_blimp":      "SB",
    "sva_isare":      "SI",
}


def load(variant_dir: Path) -> pd.DataFrame:
    """Load the classification used by the knockout (classify_weighted/v1_min)."""
    df = pd.read_csv(variant_dir / "classify_weighted" / "v1_min.csv")
    df["family"]    = df["primary_label"]   # priority-exclusive label
    df["final"]     = df["primary_label"]
    df["task_name"] = ""
    is_task = df["primary_label"].str.startswith("task_")
    df.loc[is_task, "final"]     = "task"
    df.loc[is_task, "task_name"] = df.loc[is_task, "primary_label"].str[len("task_"):]
    return df


_TASK_TO_FAMILY = {
    "country_cap":    "COUNTRY", "person_country": "COUNTRY",
    "gt":             "GT",      "gt_price":       "GT",
    "sva":            "SVA",     "sva_blimp":      "SVA",      "sva_isare":  "SVA",
    "gender_bias":    "PRONOUN", "gp":             "PRONOUN",  "person_pro": "PRONOUN",
    # col_obj, ioi_strong are singletons with no family
}
_ALL_TASKS = ["col_obj","country_cap","gender_bias","gp","gt","gt_price",
              "ioi_strong","person_country","person_pro","sva","sva_blimp","sva_isare"]


def _set_target_tasks(set_name: str) -> list:
    """Tasks a unit set is expected to disrupt; |z| is tested on these."""
    if set_name == "Universal":             return _ALL_TASKS
    if set_name == "Mixed_22_control":      return _ALL_TASKS  # control: never validate
    if set_name.startswith("TaskSpec_"):
        # TaskSpec_<task>_<n>; <task> may itself contain '_'.
        return [set_name[len("TaskSpec_"):].rsplit("_", 1)[0]]
    if set_name.startswith("GT"):           return ["gt", "gt_price"]
    if set_name.startswith("SVA"):          return ["sva", "sva_blimp", "sva_isare"]
    if set_name.startswith("COUNTRY"):      return ["country_cap", "person_country"]
    if set_name.startswith("PRONOUN"):      return ["gender_bias", "gp", "person_pro"]
    if set_name.startswith("IOI_COL"):      return ["ioi_strong", "col_obj"]
    return _ALL_TASKS


def get_validated_units(variant_dir: Path,
                        knockout_subdir: str = "knockout_validation_v1_typematched",
                        z_strict: float = 3.0,
                        z_marginal: float | None = None) -> tuple[set, set]:
    """Return (strict_validated, marginal_validated) unit-id sets.

    A unit set passes if max |z| over its expected target tasks meets the bar.
    Mixed_22_control is excluded. If z_marginal is None it defaults to
    max(2.0, control_max + 0.5), so the marginal tier stays above the control
    set's noise level (on Pythia the control max can be ~1.9).
    Falls back to a fixed category rule when lift_matrix.csv is absent."""
    classify_csv = variant_dir / "classify_weighted" / "v1_min.csv"
    df = pd.read_csv(classify_csv)
    lift_csv = variant_dir / knockout_subdir / "lift_matrix.csv"
    if not lift_csv.exists():
        validated = set()
        validated |= set(df.loc[df.category == "universal", "unit_id"].astype(int))
        validated |= set(df.loc[df.primary_label == "family_GT", "unit_id"].astype(int))
        m = (df.primary_label == "family_SVA") & (df.family_tier == "specialized")
        validated |= set(df.loc[m, "unit_id"].astype(int))
        validated |= set(df.loc[df.primary_label == "family_COUNTRY", "unit_id"].astype(int))
        is_task = df["primary_label"].str.startswith("task_")
        validated |= set(df.loc[is_task, "unit_id"].astype(int))
        return validated, set()

    lifts = pd.read_csv(lift_csv)
    set_uids = _parse_classify_into_sets(df)

    # Marginal floor: above the per-task control max in this run.
    ctrl_row = lifts[lifts["set"] == "Mixed_22_control"]
    control_max = 0.0
    if not ctrl_row.empty:
        zs = [abs(float(ctrl_row.iloc[0][f"z_{t}"]))
              for t in _ALL_TASKS if f"z_{t}" in ctrl_row.columns
              and not pd.isna(ctrl_row.iloc[0][f"z_{t}"])]
        if zs: control_max = max(zs)
    if z_marginal is None:
        # ~0.5 above the control noise floor, never below 2.0.
        z_marginal = max(2.0, round(control_max + 0.5, 1))

    strict, marginal = set(), set()
    for _, row in lifts.iterrows():
        nm = row["set"]
        if nm == "Mixed_22_control":
            continue
        targets = _set_target_tasks(nm)
        zs = [abs(float(row[f"z_{t}"])) for t in targets
              if f"z_{t}" in row and not pd.isna(row[f"z_{t}"])]
        if not zs: continue
        m = max(zs)
        if m >= z_strict:
            strict |= set(set_uids.get(nm, []))
        elif m >= z_marginal:
            marginal |= set(set_uids.get(nm, []))
    return strict, marginal


def _parse_classify_into_sets(df: pd.DataFrame) -> dict:
    """Pandas port of knockout_validate_actpatch.parse_classification: rebuild
    the same unit_id sets from v1_min.csv."""
    sets = {}
    sets["Universal"] = df.loc[df.category == "universal", "unit_id"].astype(int).tolist()

    gt_spec = df[df.primary_label.eq("family_GT") & df.family_tier.eq("specialized")] \
                .sort_values("within_GT_min", ascending=False)
    gt_broad = df[df.primary_label.eq("family_GT") & df.family_tier.eq("broad")] \
                .sort_values("within_GT_min", ascending=False)
    sets[f"GT_specialized_{len(gt_spec)}"] = gt_spec.unit_id.astype(int).tolist()
    sets["GT_specialized_top4"]            = gt_spec.head(4).unit_id.astype(int).tolist()
    sets["GT_broad_top4"]                  = sorted(gt_broad.head(4).unit_id.astype(int).tolist())
    sets["GT_broad_top9"]                  = sorted(gt_broad.head(9).unit_id.astype(int).tolist())
    sets["GT_broad_all"]                   = sorted(gt_broad.unit_id.astype(int).tolist())
    gt_broad_mlps = gt_broad[gt_broad.unit_type.isin(["mlp_gate","mlp_down"])]
    sets["GT_broad_mlps_top4"]             = sorted(gt_broad_mlps.head(4).unit_id.astype(int).tolist())
    sets["GT_broad_mlps_all"]              = sorted(gt_broad_mlps.unit_id.astype(int).tolist())
    sets["GT_all"] = sorted(set(sets[f"GT_specialized_{len(gt_spec)}"]) | set(sets["GT_broad_all"]))

    sets["SVA_specialized_4"] = df[df.primary_label.eq("family_SVA") & df.family_tier.eq("specialized")] \
                                  .unit_id.astype(int).tolist()
    sva_broad = df[df.primary_label.eq("family_SVA") & df.family_tier.eq("broad")]
    sets["SVA_broad_all"] = sorted(sva_broad.unit_id.astype(int).tolist())
    sets["SVA_all"]       = sorted(set(sets["SVA_specialized_4"]) | set(sets["SVA_broad_all"]))
    sets["COUNTRY_broad_2"] = df[df.primary_label.str.startswith("family_COUNTRY")] \
                                .unit_id.astype(int).tolist()
    sets["PRONOUN_all"]     = df[df.primary_label.str.startswith("family_PRONOUN")] \
                                .unit_id.astype(int).tolist()
    sets["IOI_COL_all"]     = df[df.primary_label.str.startswith("family_IOI_COL")] \
                                .unit_id.astype(int).tolist()

    for tname, group in df[df.category.eq("task")].groupby(
            df["primary_label"].str.replace("^task_", "", regex=True)):
        sets[f"TaskSpec_{tname}_{len(group)}"] = sorted(group.unit_id.astype(int).tolist())

    return {k: v for k, v in sets.items() if v}


def cell_color(label: str) -> str:
    if label == "universal": return UNIVERSAL_COLOR
    if label == "task":      return TASK_COLOR
    if label == "mixed":     return MIXED_COLOR
    if label.startswith("family_"):
        fam = label.split("_", 1)[1]
        return FAM_COLOR.get(fam, "#888888")
    return "#FFFFFF"


def plot_one(variant_dir: Path, out_base: Path) -> None:
    df = load(variant_dir)
    validated_strict, validated_marginal = get_validated_units(variant_dir)
    grid_label    = np.full((N_LAYERS, N_INTRA), "", dtype=object)
    grid_family   = np.full((N_LAYERS, N_INTRA), "", dtype=object)
    grid_task     = np.full((N_LAYERS, N_INTRA), "", dtype=object)
    grid_valid    = np.zeros((N_LAYERS, N_INTRA), dtype=bool)
    grid_marginal = np.zeros((N_LAYERS, N_INTRA), dtype=bool)
    for _, r in df.iterrows():
        L = int(r["layer"])
        intra = int(r["unit_id"]) - L * N_INTRA
        if 0 <= intra < N_INTRA and 0 <= L < N_LAYERS:
            grid_label[L, intra]    = r["final"]
            grid_family[L, intra]   = r["family"]
            grid_task[L, intra]     = r["task_name"]
            grid_valid[L, intra]    = int(r["unit_id"]) in validated_strict
            grid_marginal[L, intra] = int(r["unit_id"]) in validated_marginal

    # Layout (data units)
    head_r        = 0.34
    mlp_w         = 0.78
    mlp_h         = 0.78
    head_spacing  = 1.0
    gap_attn_mlp  = 1.4
    mlp_spacing   = 1.0
    active_h      = 1.25
    compressed_h  = 0.50    # height of empty (Unassigned-only) layers

    attn_xs = [i * head_spacing for i in range(12)]
    mlp_x0  = attn_xs[-1] + gap_attn_mlp
    mlp_xs  = [mlp_x0 + i * mlp_spacing for i in range(2)]

    # Identify "empty" layers (no non-Unassigned units) and compute
    # per-layer heights / y-centers so empty layers take less space.
    is_empty = np.array([
        all(grid_label[L, i] in ("mixed", "") for i in range(N_INTRA))
        for L in range(N_LAYERS)
    ])
    layer_h = np.where(is_empty, compressed_h, active_h)
    layer_yc = np.empty(N_LAYERS)
    cum = 0.0
    for L in range(N_LAYERS):
        layer_yc[L] = cum + layer_h[L] / 2
        cum += layer_h[L]
    total_h = cum

    rc = paper_rc("double")
    rc["axes.spines.top"]    = False
    rc["axes.spines.right"]  = False
    rc["axes.spines.left"]   = False
    rc["axes.spines.bottom"] = False
    rc["font.size"]       = 10
    rc["legend.fontsize"] = 10

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(13.5, 7.6))

        x_lo, x_hi = -1.3, mlp_xs[-1] + mlp_w
        y_top_layer = layer_yc[-1] + layer_h[-1] / 2
        y_bot_layer = layer_yc[0] - layer_h[0] / 2
        y_lo = y_bot_layer - 0.7
        y_hi = y_top_layer + 0.3

        # Geometry of the attn vs MLP block boundary
        attn_block_x_hi = (attn_xs[-1] + mlp_xs[0]) / 2 - 0.05
        mlp_block_x_lo  = (attn_xs[-1] + mlp_xs[0]) / 2 + 0.05

        for L in range(N_LAYERS):
            y = layer_yc[L]
            h = layer_h[L]

            # Alternating background; attn and MLP get slightly different
            # tints so the two regions read as separate blocks.
            attn_band = "#FBFBFD" if L % 2 == 0 else "#F2F3F6"
            mlp_band  = "#F4F1EC" if L % 2 == 0 else "#EBE7DF"
            ax.add_patch(Rectangle(
                (x_lo, y - h / 2),
                attn_block_x_hi - x_lo, h,
                facecolor=attn_band, edgecolor="none", zorder=0,
            ))
            ax.add_patch(Rectangle(
                (mlp_block_x_lo, y - h / 2),
                x_hi - mlp_block_x_lo, h,
                facecolor=mlp_band, edgecolor="none", zorder=0,
            ))

            # Layer label
            ax.text(-0.7, y, f"L{L}",
                    ha="right", va="center",
                    fontsize=10.5, fontweight="bold",
                    color="#333", zorder=4)

            # Attention heads. Mixed cells are drawn as small dots so the
            # classified units stand out.
            for i in range(12):
                x = attn_xs[i]
                lbl = grid_label[L, i]
                vld = grid_valid[L, i]
                marg = grid_marginal[L, i]
                if lbl == "mixed":
                    ax.add_patch(Circle(
                        (x, y), head_r * 0.22,
                        facecolor="#C8C8C8", edgecolor="none", zorder=2,
                    ))
                    continue
                if vld:
                    ec, lw, ls = "#F5C518", 2.0, "solid"
                elif marg:
                    ec, lw, ls = "#E89A2B", 1.6, "dashed"
                else:
                    ec, lw, ls = "white", 0.9, "solid"
                ax.add_patch(Circle(
                    (x, y), head_r,
                    facecolor=cell_color(lbl),
                    edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2,
                ))
                if lbl == "task":
                    tag = TASK_TAG.get(grid_task[L, i], "")
                    # Smaller font for 3-char tags so they fit.
                    fs = 9.0 if len(tag) >= 3 else 10.0
                    ax.text(x, y, tag, ha="center", va="center",
                            fontsize=fs, color="white",
                            fontweight="bold", zorder=4)

            # MLP slots; mixed slots are small squares
            for i in range(2):
                x = mlp_xs[i]
                lbl = grid_label[L, 12 + i]
                vld = grid_valid[L, 12 + i]
                marg = grid_marginal[L, 12 + i]
                if lbl == "mixed":
                    ax.add_patch(Rectangle(
                        (x - mlp_w * 0.13, y - mlp_h * 0.13),
                        mlp_w * 0.26, mlp_h * 0.26,
                        facecolor="#C8C8C8", edgecolor="none", zorder=2,
                    ))
                    continue
                if vld:
                    ec, lw, ls = "#F5C518", 2.0, "solid"
                elif marg:
                    ec, lw, ls = "#E89A2B", 1.6, "dashed"
                else:
                    ec, lw, ls = "white", 0.9, "solid"
                ax.add_patch(Rectangle(
                    (x - mlp_w / 2, y - mlp_h / 2), mlp_w, mlp_h,
                    facecolor=cell_color(lbl),
                    edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2,
                ))

        # Vertical divider between attn and MLP blocks
        y_top = y_top_layer
        y_bot = y_bot_layer
        divider_x = (attn_block_x_hi + mlp_block_x_lo) / 2
        ax.plot([divider_x, divider_x], [y_bot, y_top],
                color="#888", lw=0.8, ls=(0, (3, 3)), zorder=1)

        # Header brackets above each block
        bracket_y = y_top + 0.15
        bracket_drop = 0.12
        for x0, x1 in [(x_lo + 0.5, attn_block_x_hi - 0.1),
                        (mlp_block_x_lo + 0.1, x_hi - 0.1)]:
            ax.plot([x0, x0, x1, x1],
                    [bracket_y - bracket_drop, bracket_y,
                     bracket_y, bracket_y - bracket_drop],
                    color="#444", lw=0.9, zorder=4, clip_on=False)

        ax.text(np.mean(attn_xs),
                bracket_y + 0.10,
                "Attention heads",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#222",
                clip_on=False)
        ax.text(np.mean(mlp_xs),
                bracket_y + 0.10,
                "MLP", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#222",
                clip_on=False)

        # Column labels (bottom), same size as layer labels
        for i in range(12):
            ax.text(attn_xs[i], -0.7, f"H{i}",
                    ha="center", va="top",
                    fontsize=10.5, color="#333")
        for x, name in zip(mlp_xs, ["up", "down"]):
            ax.text(x, -0.7, name,
                    ha="center", va="top",
                    fontsize=10.5, color="#333")

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.axis("off")

        # Legend (categories + validation marker)
        handles = [Patch(facecolor=UNIVERSAL_COLOR, label="Universal")]
        for fam in ("GT", "SVA", "COUNTRY", "PRONOUN"):
            handles.append(Patch(facecolor=FAM_COLOR[fam], label=FAM_NAME[fam]))
        handles.append(Patch(facecolor=TASK_COLOR, label="Task-specific"))
        handles.append(Line2D(
            [0], [0], marker="o", color="white",
            markerfacecolor="#C8C8C8", markersize=5, label="Unassigned",
        ))
        handles.append(Patch(
            facecolor="white", edgecolor="#F5C518", linewidth=2.0,
            label="Ablation-validated  (|z|≥3)",
        ))
        if validated_marginal:
            handles.append(Patch(
                facecolor="white", edgecolor="#E89A2B",
                linewidth=1.6, linestyle="dashed",
                label="Marginal  (above control max)",
            ))
        ax.legend(
            handles=handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.03), ncol=4,
            frameon=False, handletextpad=0.5, columnspacing=1.5,
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
        plot_one(d, d / "unit_circuit")


if __name__ == "__main__":
    main()
