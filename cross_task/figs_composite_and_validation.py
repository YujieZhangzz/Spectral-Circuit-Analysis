"""
Figures and tables for the cross-task classification and knockout validation.

Writes to cross_task/run_gpt_12tasks/consistency_share/:
    composite_task_distribution.{pdf,png}  category of each task's top-K importance units
    composite_task_topK.csv                per-task top-K unit ids + categories
    task_specific_units.{csv,md}           units classified task-specific in any layout
    family_lift_matrix.{csv,md}            v1 knockout lift / z-score / concentration
    family_lift_heatmap.{pdf,png}          v1 lift matrix heatmap

Inputs: cross_task_result.pkl, classify_weighted/v{1,2}_{min,mean}.csv and
knockout_validation_v1_typematched/summary.txt in the same directory.
Usage:
    python cross_task/figs_composite_and_validation.py
"""
from __future__ import annotations
import csv, pickle, re, sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils.paper_style import paper_rc, paper_size

VARIANT_DIR = REPO_ROOT / "cross_task" / "run_gpt_12tasks" / "consistency_share"

# Category palette
CAT_ORDER = [
    "Universal",
    "family_GT",
    "family_SVA",
    "family_COUNTRY",
    "family_PRONOUN",
    "family_IOI_COL",
    "task_specific",
    "Mixed",
]
CAT_COLORS = {
    "Universal":       "#1f3a93",
    "family_GT":       "#c0392b",
    "family_SVA":      "#1e8449",
    "family_COUNTRY":  "#ca6f1e",
    "family_PRONOUN":  "#7d6608",
    "family_IOI_COL":  "#5d6d7e",
    "task_specific":   "#7d3c98",
    "Mixed":           "#bdc3c7",
}
CAT_DISPLAY = {
    "Universal":       "Universal",
    "family_GT":       "GT family",
    "family_SVA":      "SVA family",
    "family_COUNTRY":  "COUNTRY family",
    "family_PRONOUN":  "PRONOUN family",
    "family_IOI_COL":  "IOI/COL family",
    "task_specific":   "Task-specific",
    "Mixed":           "Mixed",
}

TASK_DISPLAY = {
    "col_obj":         "colored-obj",
    "country_cap":     "country-cap",
    "gender_bias":     "gender-bias",
    "gp":              "gender-pronoun",
    "gt":              "gt",
    "gt_price":        "gt-price",
    "ioi_strong":      "IOI",
    "person_country":  "person-country",
    "person_pro":      "person-pronoun",
    "sva":             "sva",
    "sva_blimp":       "sva-blimp",
    "sva_isare":       "sva-isare",
}

# Task order: composite (PRONOUN) | families (GT, SVA, COUNTRY) | singletons
TASK_ORDER = [
    "gender_bias", "gp", "person_pro",          # PRONOUN (composite)
    "gt", "gt_price",                            # GT family
    "sva", "sva_blimp", "sva_isare",             # SVA family
    "country_cap", "person_country",             # COUNTRY family
    "ioi_strong", "col_obj",                     # singletons / v2 IOI_COL
]


def cat_key_from_row(r) -> str:
    """Map a classify CSV row to a coarse category for grouping."""
    c, p, _ft = r["category"], r["primary_label"], r["family_tier"]
    if c == "universal": return "Universal"
    if c == "task":      return "task_specific"
    if c == "mixed":     return "Mixed"
    if c == "family":
        return f"family_{p.split('_', 1)[1]}"     # family_GT, family_SVA, ...
    return "Mixed"


# Figure 1: composite-task distribution

def make_composite_fig(K: int = 10) -> dict:
    pkl  = pickle.load(open(VARIANT_DIR / "cross_task_result.pkl", "rb"))
    imp  = np.asarray(pkl["tensor"]["importance"])           # (n_units, n_tasks)
    uids = list(pkl["tensor"]["unit_ids"])
    tnames = pkl["task_names"]

    rows = list(csv.DictReader(open(VARIANT_DIR / "classify_weighted" / "v2_min.csv")))
    uid_to_cat  = {int(r["unit_id"]): cat_key_from_row(r) for r in rows}
    uid_to_full = {int(r["unit_id"]): (r["category"], r["primary_label"], r["family_tier"])
                   for r in rows}

    # Per-task top-K counts
    counts = {t: {c: 0 for c in CAT_ORDER} for t in TASK_ORDER}
    detail = {t: [] for t in TASK_ORDER}
    for t in TASK_ORDER:
        ti = tnames.index(t)
        order = np.argsort(-imp[:, ti])[:K]
        for rank_, idx in enumerate(order):
            uid = uids[idx]
            cat = uid_to_cat.get(uid, "Mixed")
            counts[t][cat] += 1
            full_cat, primary, tier = uid_to_full.get(uid, ("?", "?", "?"))
            layer = uid // 14; intra = uid % 14
            name = (f"L{layer}H{intra}" if intra < 12
                    else (f"L{layer}.mlp_gate" if intra == 12 else f"L{layer}.mlp_down"))
            detail[t].append({
                "rank":     rank_ + 1,
                "task":     t,
                "unit_id":  uid,
                "name":     name,
                "imp":      float(imp[idx, ti]),
                "category": full_cat,
                "primary_label": primary,
                "family_tier":   tier,
                "cat_key":  cat,
            })

    # Save supporting CSV
    csv_out = VARIANT_DIR / "composite_task_topK.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "task", "unit_id", "name", "imp",
                                          "category", "primary_label", "family_tier",
                                          "cat_key"])
        w.writeheader()
        for t in TASK_ORDER:
            for d in detail[t]:
                w.writerow(d)
    print(f"  [saved] {csv_out}")

    # Plot stacked bar
    fig_path = VARIANT_DIR / "composite_task_distribution"
    # Keep all family categories in the legend even if empty (an empty PRONOUN family is informative).
    used_cats = [c for c in CAT_ORDER
                 if c.startswith("family_")
                 or any(counts[t][c] > 0 for t in TASK_ORDER)]
    with plt.rc_context(paper_rc("double")):
        _, ax = plt.subplots(figsize=paper_size("double", aspect=1.5))
        x = np.arange(len(TASK_ORDER))
        bottom = np.zeros(len(TASK_ORDER))
        for c in used_cats:
            h = np.array([counts[t][c] for t in TASK_ORDER])
            label = CAT_DISPLAY[c]
            if h.sum() == 0:
                label = label + "  (empty — no units classified)"
            # Always add to the legend; a zero-height bar still yields a swatch.
            ax.bar(x, h, bottom=bottom,
                   color=CAT_COLORS[c], label=label,
                   edgecolor="white", linewidth=0.6, width=0.78)
            bottom += h
        # Annotate PRONOUN task columns that have no PRONOUN-family units
        for ti, t in enumerate(TASK_ORDER):
            if t in ("gender_bias", "gp", "person_pro") and counts[t]["family_PRONOUN"] == 0:
                ax.annotate("0 PRONOUN", xy=(ti, K + 0.15),
                            ha="center", va="bottom", fontsize=6.5,
                            color=CAT_COLORS["family_PRONOUN"], fontweight="bold")

        # group separators (PRONOUN | GT | SVA | COUNTRY | singletons)
        for sep in (2.5, 4.5, 7.5, 9.5):
            ax.axvline(sep, color="#888", linewidth=0.5, linestyle="--", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_DISPLAY[t] for t in TASK_ORDER],
                           rotation=35, ha="right")
        ax.set_ylabel(f"Top-{K} importance units")
        ax.set_ylim(0, K + 1.2)
        ax.set_yticks(range(0, K + 1, 2))

        # group labels above the figure
        for label, xc in [("PRONOUN\n(composite)", 1.0),
                          ("GT family",            3.5),
                          ("SVA family",           6.5),
                          ("COUNTRY family",       8.5),
                          ("Singletons",          10.5)]:
            ax.text(xc, K + 1.4, label, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#333")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32),
                  ncol=4, fontsize=7.5, frameon=False)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(fig_path.with_suffix(f".{ext}"),
                        bbox_inches="tight",
                        dpi=300 if ext == "png" else None)
        plt.close()
    print(f"  [saved] {fig_path.with_suffix('.pdf')}")

    return counts


# Figure 2: family-validation lift heatmap (v1)

def parse_lift_summary(path: Path):
    """Extract the (lift, z) matrices from a knockout summary.txt.

    Layout:
        Lift matrix ...
        set | ...
        ---- separator ----
        <rows>
        <blank>
        z-score (...)
        set | ...
        ---- separator ----
        <rows>
        <blank>
        Concentration ...

    Returns (set_names, task_cols, lift_matrix, z_matrix). z_matrix is None for
    older summaries without z-score block.
    """
    text = path.read_text().splitlines()
    # First "set ... |" is the lift block header
    i_hdrs = [i for i, l in enumerate(text) if l.startswith("set ") and "|" in l]
    if not i_hdrs:
        raise RuntimeError(f"no header found in {path}")
    cols = re.split(r"\s+", text[i_hdrs[0]].split("|", 1)[1].strip())

    def parse_block(start_idx: int):
        rows, names = [], []
        for line in text[start_idx + 2:]:
            if not line.strip(): break
            if line.startswith(("Concentration", "z-score", "set ")): break
            if "|" not in line: continue
            name, vals_s = line.split("|", 1)
            vals = []
            for v in re.split(r"\s+", vals_s.strip()):
                try:    vals.append(float(v))
                except ValueError: vals.append(float("nan"))
            rows.append(vals); names.append(name.strip())
        return names, np.array(rows)

    set_names, lift_M = parse_block(i_hdrs[0])
    z_M = None
    if len(i_hdrs) >= 2:
        _, z_M = parse_block(i_hdrs[1])
    return set_names, cols, lift_M, z_M


def make_lift_heatmap():
    sumtxt = VARIANT_DIR / "knockout_validation_v1_typematched" / "summary.txt"
    set_names, task_cols, M, _Z = parse_lift_summary(sumtxt)
    print(f"  parsed {len(set_names)} sets x {len(task_cols)} tasks from v1")

    # Map summary set names to display labels
    set_disp = {
        "Universal":            "Universal",
        "GT_specialized_2":     "GT specialized (2)",
        "GT_specialized_top4":  "GT specialized top-4",
        "GT_broad_top4":        "GT broad top-4",
        "GT_broad_top9":        "GT broad top-9",
        "GT_broad_all":         "GT broad all",
        "GT_broad_mlps_top4":   "GT broad +MLPs top-4",
        "GT_broad_mlps_all":    "GT broad +MLPs all",
        "GT_all":               "GT all",
        "SVA_specialized_4":    "SVA specialized (4)",
        "SVA_broad_all":        "SVA broad",
        "SVA_all":              "SVA all",
        "COUNTRY_broad_2":      "COUNTRY broad (2)",
        "Mixed_22_control":     "Mixed (control)",
    }
    # Map task short names to display
    task_disp_full = {"col_obj": "col-obj", "country_": "country", "gender_b": "gender-b",
                      "gp": "gp", "gt": "gt", "gt_price": "gt-price",
                      "ioi_stro": "IOI", "person_c": "person-c", "person_p": "person-p",
                      "sva": "sva", "sva_blim": "sva-blim", "sva_isar": "sva-isare"}
    xlabels = [task_disp_full.get(c, c) for c in task_cols]
    ylabels = [set_disp.get(n, n) for n in set_names]

    fig_path = VARIANT_DIR / "family_lift_heatmap"
    with plt.rc_context(paper_rc("double")):
        fig, ax = plt.subplots(figsize=paper_size("double", aspect=0.95))
        # symmetric around 0
        vmax = max(abs(M.min()), abs(M.max()))
        im = ax.imshow(M, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        # annotate cells
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                color = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.4, color=color)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Evaluation task")
        ax.set_ylabel("Knockout set")
        cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
        cb.set_label("Lift (random-corrected)")
        cb.ax.tick_params(labelsize=7)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(fig_path.with_suffix(f".{ext}"),
                        bbox_inches="tight",
                        dpi=300 if ext == "png" else None)
        plt.close()
    print(f"  [saved] {fig_path.with_suffix('.pdf')}")


def make_task_specific_table():
    """List every unit classified as task-specific (across v1/v2 x min/mean)."""
    base = VARIANT_DIR / "classify_weighted"
    out_md = VARIANT_DIR / "task_specific_units.md"
    out_csv = VARIANT_DIR / "task_specific_units.csv"

    # Aggregate across all 4 (layout, criterion) configs
    agg = {}    # uid -> dict of which configs flagged it as task-specific, and to which task
    for tag in ("v1_min", "v2_min", "v1_mean", "v2_mean"):
        rows = list(csv.DictReader(open(base / f"{tag}.csv")))
        for r in rows:
            if r["category"] == "task":
                uid = int(r["unit_id"])
                t   = r["primary_label"].split("_", 1)[1]   # task name
                d = agg.setdefault(uid, {"task": t, "configs": [], "imp_max": float(r["importance_max"]),
                                          "imp_dom": float(r["importance_dominance"]),
                                          "imp_ratio": float(r["importance_ratio"]),
                                          "outside_overlap": float(r["task_outside_overlap_max"]),
                                          "unit_type": r["unit_type"], "layer": int(r["layer"])})
                d["configs"].append(tag)

    # Sort by task then by uid
    sorted_uids = sorted(agg.keys(), key=lambda u: (agg[u]["task"], u))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "name", "task", "unit_type", "layer",
                    "imp_max", "imp_dominance", "imp_ratio",
                    "outside_overlap_max", "configs_flagged"])
        for uid in sorted_uids:
            d = agg[uid]
            layer = uid // 14; intra = uid % 14
            name = (f"L{layer}H{intra}" if intra < 12
                    else (f"L{layer}.mlp_gate" if intra == 12 else f"L{layer}.mlp_down"))
            w.writerow([uid, name, d["task"], d["unit_type"], d["layer"],
                        f"{d['imp_max']:.3e}", f"{d['imp_dom']:.3f}",
                        f"{d['imp_ratio']:.3f}", f"{d['outside_overlap']:.3f}",
                        "|".join(d["configs"])])

    # Markdown for direct reading
    lines = ["# Task-specific units across v1/v2 × min/mean classifications\n"]
    by_task = {}
    for uid in sorted_uids:
        by_task.setdefault(agg[uid]["task"], []).append(uid)
    for t, uids in by_task.items():
        lines.append(f"## task = `{t}`  ({len(uids)} unit{'s' if len(uids)!=1 else ''})\n")
        lines.append("| unit | type | layer | importance_max | dominance | ratio | outside_overlap | flagged in |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for uid in uids:
            d = agg[uid]
            layer = uid // 14; intra = uid % 14
            name = (f"L{layer}H{intra}" if intra < 12
                    else (f"L{layer}.mlp_gate" if intra == 12 else f"L{layer}.mlp_down"))
            cfgs = ",".join(c.replace("_", "-") for c in d["configs"])
            lines.append(f"| `{name}` (uid={uid}) | {d['unit_type']} | {d['layer']} "
                         f"| {d['imp_max']:.2e} | {d['imp_dom']:.2f} | {d['imp_ratio']:.2f} "
                         f"| {d['outside_overlap']:.3f} | {cfgs} |")
        lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"  [saved] {out_md}")
    print(f"  [saved] {out_csv}")
    return agg, sorted_uids


def make_lift_table():
    """Render the v1 (type-matched) lift + z-score matrices as markdown + CSV."""
    sumtxt = VARIANT_DIR / "knockout_validation_v1_typematched" / "summary.txt"
    set_names, task_cols, M, Z = parse_lift_summary(sumtxt)

    # Display labels
    set_disp = {
        "Universal":            "Universal",
        "GT_specialized_2":     "GT specialized (2)",
        "GT_specialized_top4":  "GT specialized top-4",
        "GT_broad_top4":        "GT broad top-4",
        "GT_broad_top9":        "GT broad top-9",
        "GT_broad_all":         "GT broad all",
        "GT_broad_mlps_top4":   "GT broad +MLPs top-4",
        "GT_broad_mlps_all":    "GT broad +MLPs all",
        "GT_all":               "GT all",
        "SVA_specialized_4":    "SVA specialized (4)",
        "SVA_broad_all":        "SVA broad",
        "SVA_all":              "SVA all",
        "COUNTRY_broad_2":      "COUNTRY broad (2)",
        "TaskSpec_ioi_strong_7":  "IOI cohort (7)",
        "TaskSpec_country_cap_1": "L8H11 (country)",
        "TaskSpec_person_pro_1":  "L10H11 (person-pro)",
        "Mixed_22_control":     "Mixed (control)",
    }
    task_disp_full = {"col_obj": "col-obj", "country_": "country-cap",
                      "gender_b": "gender-bias", "gp": "gp", "gt": "gt",
                      "gt_price": "gt-price", "ioi_stro": "IOI",
                      "person_c": "person-c", "person_p": "person-p",
                      "sva": "sva", "sva_blim": "sva-blim", "sva_isar": "sva-isare"}
    xlabels = [task_disp_full.get(c, c) for c in task_cols]
    ylabels = [set_disp.get(n, n) for n in set_names]

    # Family / task membership for each set (for in/out concentration)
    set_fam = {
        "GT_specialized_2": "GT", "GT_specialized_top4": "GT",
        "GT_broad_top4": "GT", "GT_broad_top9": "GT", "GT_broad_all": "GT",
        "GT_broad_mlps_top4": "GT", "GT_broad_mlps_all": "GT", "GT_all": "GT",
        "SVA_specialized_4": "SVA", "SVA_broad_all": "SVA", "SVA_all": "SVA",
        "COUNTRY_broad_2": "COUNTRY",
    }
    set_target_task = {  # for TaskSpec_* sets, in = single target task col-name
        "TaskSpec_ioi_strong_7":  "ioi_stro",
        "TaskSpec_country_cap_1": "country_",
        "TaskSpec_person_pro_1":  "person_p",
    }
    fam_tasks = {
        "GT":      {"gt", "gt_price"},
        "SVA":     {"sva", "sva_blim", "sva_isar"},
        "COUNTRY": {"country_", "person_c"},      # raw column names from summary
    }

    def in_out_idxs(sname):
        if sname in set_target_task:
            in_  = [j for j, c in enumerate(task_cols) if c == set_target_task[sname]]
            out_ = [j for j in range(len(task_cols)) if j not in in_]
            return in_, out_, f"task:{set_target_task[sname]}"
        fam = set_fam.get(sname)
        if not fam:
            return [], list(range(len(task_cols))), "—"
        in_  = [j for j, c in enumerate(task_cols) if c in fam_tasks[fam]]
        out_ = [j for j in range(len(task_cols)) if j not in in_]
        return in_, out_, fam

    # CSV: full lift + z + concentration
    out_csv = VARIANT_DIR / "family_lift_matrix.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["set"] + [f"lift_{c}" for c in xlabels]
        if Z is not None:
            header += [f"z_{c}" for c in xlabels]
        header += ["fam_or_task", "in_fam_mean", "out_fam_mean",
                   "ratio_signed", "abs_ratio"]
        w.writerow(header)
        for i, sname in enumerate(set_names):
            row = [ylabels[i]] + [f"{v:.3f}" for v in M[i]]
            if Z is not None:
                row += [f"{v:.2f}" if not np.isnan(v) else "" for v in Z[i]]
            in_, out_, fam_label = in_out_idxs(sname)
            if in_:
                in_m  = float(np.mean(M[i, in_]))
                out_m = float(np.mean(M[i, out_]))
                signed = (in_m / out_m) if abs(out_m) > 1e-9 else float("nan")
                absrat = (abs(in_m) / abs(out_m)) if abs(out_m) > 1e-9 else float("nan")
                row += [fam_label, f"{in_m:.3f}", f"{out_m:.3f}",
                        f"{signed:.2f}", f"{absrat:.2f}"]
            else:
                row += [fam_label, "", "", "", ""]
            w.writerow(row)
    print(f"  [saved] {out_csv}")

    # Markdown
    lines = ["# v1 knockout-validation lift matrix (type-matched random baseline)\n"]
    lines.append(
        "Knockout sets (rows) ablated via activation patching from corrupted prompt. "
        "**lift** = `F_S − F_R_mean` (excess faithfulness drop above same-composition random); "
        "**z** = `(F_S − F_R_mean) / F_R_std` (significance vs same-composition random). "
        "Random baseline is **type-matched**: random sets preserve each set's "
        "(n_attn, n_mlp) composition. Treat |z| ≥ 3 as confident, |z| < 1 as random-like. "
        "Concentration ratio reported with both signed and |·| values "
        "because the importance metric is direction-agnostic.\n")

    # Table 1: lift matrix
    lines.append("## Lift matrix\n")
    header = "| set | " + " | ".join(xlabels) + " |"
    sep    = "|---|" + "|".join(["---:"] * len(xlabels)) + "|"
    lines += [header, sep]
    for i, sname in enumerate(set_names):
        cells = " | ".join(f"{v:+.2f}" for v in M[i])
        lines.append(f"| **{ylabels[i]}** | {cells} |")
    lines.append("")

    # Table 2: z-score matrix
    if Z is not None:
        lines.append("## z-score matrix\n")
        lines += [header, sep]
        for i, sname in enumerate(set_names):
            cells = " | ".join(f"{v:+.1f}" if not np.isnan(v) else "—" for v in Z[i])
            lines.append(f"| **{ylabels[i]}** | {cells} |")
        lines.append("")

    # Table 3: concentration
    lines.append("## Concentration (in-family / in-task vs out)\n")
    lines.append("| set | scope | in mean | out mean | signed ratio | |ratio| |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for i, sname in enumerate(set_names):
        in_, out_, fam_label = in_out_idxs(sname)
        if not in_:
            lines.append(f"| **{ylabels[i]}** | — | — "
                         f"| {np.mean(M[i]):+.3f} (all-task mean) | — | — |")
            continue
        in_m  = float(np.mean(M[i, in_]))
        out_m = float(np.mean(M[i, out_]))
        signed = (in_m / out_m) if abs(out_m) > 1e-9 else float("nan")
        absrat = (abs(in_m) / abs(out_m)) if abs(out_m) > 1e-9 else float("nan")
        lines.append(f"| **{ylabels[i]}** | {fam_label} "
                     f"| {in_m:+.3f} | {out_m:+.3f} "
                     f"| {signed:+.2f} | {absrat:.2f} |")
    lines.append("")

    out_md = VARIANT_DIR / "family_lift_matrix.md"
    out_md.write_text("\n".join(lines))
    print(f"  [saved] {out_md}")


def main():
    print("[1] composite-task distribution")
    counts = make_composite_fig(K=10)
    for t in ("gender_bias", "gp", "person_pro"):
        n_own = counts[t]["family_PRONOUN"]
        print(f"    {t:<14s}  family_PRONOUN in top-10 = {n_own}")

    print("\n[2] task-specific units table")
    make_task_specific_table()

    print("\n[3] family-validation lift table (v1)")
    make_lift_table()

    print("\n[4] lift heatmap")
    make_lift_heatmap()


if __name__ == "__main__":
    main()
