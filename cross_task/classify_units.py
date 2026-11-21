"""
Classify cross-task units into universal / family-specific / task-specific / mixed.

Reads <CT_RUN>/{consistency_share,sigma_weighted}/cross_task_result.pkl and,
for each family layout (v1: orphans as singletons, v2: IOI_COL grouped,
v3: orphans as singleton families) and universality criterion (min, mean),
writes classify[_weighted]/<layout>_<crit>.csv plus summary.csv and legend.txt.

Classification rule (priority order):
    1. universal:  score_<crit> >= θ_universal
    2. family F:   within_F_<crit> >= within_floor[F] AND
                   within_F_<crit> - cross_F_max >= θ_margin
                   (largest margin wins if several families qualify)
    3. task t*:    importance_dominance >= θ_task_dominance AND
                   task_outside_overlap_max <= θ_task_outside_overlap
    4. mixed:      otherwise

Usage:
    python cross_task/classify_units.py --use_weighted
"""
from __future__ import annotations
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT  = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")

# Family layouts
FAMILIES_V1 = {
    "SVA":     ["sva", "sva_blimp", "sva_isare"],
    "PRONOUN": ["gender_bias", "gp", "person_pro"],
    "COUNTRY": ["country_cap", "person_country"],
    "GT":      ["gt", "gt_price"],
}
FAMILIES_V2 = {
    **FAMILIES_V1,
    "IOI_COL": ["ioi_strong", "col_obj"],
}

# v3: each orphan task as its own singleton "family". Singletons have no
# within-pairs and are skipped by the family rule, so v3 should match v1;
# kept as a sanity check.
FAMILIES_V3 = {
    **FAMILIES_V1,
    "IOI":     ["ioi_strong"],
    "COL_OBJ": ["col_obj"],
}

# Thresholds
TH = {
    "universal_min":        0.08,   # score_min >= this  -> universal (criterion=min)
    "universal_mean":       0.20,   # score_mean >= this -> universal (criterion=mean)
    "within_default":       0.10,   # default within-family score floor
    "margin":               0.05,   # within - cross_max margin
    "task_dominance":       0.30,   # imp[u, t*] / sum(imp[u, :])  (>= 30% of total)
    "task_outside_overlap": 0.15,   # max overlap from t* to any other task
}

# Per-family within floors for families whose cross-pair overlap is
# naturally low (GT, IOI_COL), to filter units that are weak everywhere.
WITHIN_OVERRIDES = {
    "GT":      0.20,    # filters units that are weak everywhere
    "IOI_COL": 0.15,    # likewise
}

# Optional strict tier per family: units with within_F_<crit> >= this are
# labelled "specialized", the rest "broad". Families without an entry
# have only the "broad" tier.
WITHIN_STRICT = {
    "GT":   0.50,   # specialized GT (>=0.50) vs broad GT [0.20, 0.50)
    "SVA":  0.45,   # specialized SVA (>=0.45) vs broad SVA [0.10, 0.45)
}


def compute_auto_thresholds(
    overlap_for_thresh: np.ndarray,            # (U, T, T)
    family_indices:     Dict[str, List[int]],
    quantile_floor:     float = 0.95,           # top 5% triggers family/universal floor
    quantile_strict:    float = 0.99,           # top 1% triggers specialized tier
):
    """
    Data-driven thresholds from quantiles of the per-unit overlap scores.

    Returns a dict with:
      universal_min / universal_mean : q_floor of score_min / score_mean
      within_min[F] / within_mean[F] : q_floor of within_F_min / within_F_mean
      strict_min[F] / strict_mean[F] : q_strict of within_F_min / within_F_mean
    """
    U, T, _ = overlap_for_thresh.shape
    triu_i, triu_j = np.triu_indices(T, k=1)

    # Per-unit score_min, score_mean over all off-diagonal pairs
    pairs_all = overlap_for_thresh[:, triu_i, triu_j]              # (U, n_pairs)
    score_min  = pairs_all.min(axis=1)
    score_mean = pairs_all.mean(axis=1)
    universal_min_floor  = float(np.quantile(score_min,  quantile_floor))
    universal_mean_floor = float(np.quantile(score_mean, quantile_floor))

    # Per-family within distributions
    within_min_floor: Dict[str, float]  = {}
    within_mean_floor: Dict[str, float] = {}
    within_min_strict: Dict[str, float]  = {}
    within_mean_strict: Dict[str, float] = {}
    for fname, idx_list in family_indices.items():
        if len(idx_list) < 2:
            continue
        # within-pair indices for family F
        ii_list, jj_list = [], []
        for ii, i in enumerate(idx_list):
            for j in idx_list[ii + 1:]:
                ii_list.append(i); jj_list.append(j)
        within_pairs = overlap_for_thresh[:, ii_list, jj_list]      # (U, n_within)
        wmin  = within_pairs.min(axis=1)
        wmean = within_pairs.mean(axis=1)
        within_min_floor[fname]   = float(np.quantile(wmin,  quantile_floor))
        within_mean_floor[fname]  = float(np.quantile(wmean, quantile_floor))
        within_min_strict[fname]  = float(np.quantile(wmin,  quantile_strict))
        within_mean_strict[fname] = float(np.quantile(wmean, quantile_strict))

    return {
        "universal_min":  universal_min_floor,
        "universal_mean": universal_mean_floor,
        "within_min":     within_min_floor,
        "within_mean":    within_mean_floor,
        "strict_min":     within_min_strict,
        "strict_mean":    within_mean_strict,
    }


def compute_weighted_overlap(overlap: np.ndarray, importance: np.ndarray,
                              unit_meta: list = None) -> np.ndarray:
    """
    Importance-weighted overlap with type-stratified max-normalization.

    weighted[u, s, t]  =  sqrt(norm_I[u, s] · norm_I[u, t]) · overlap[u, s, t]
    where norm_I[u, t]  =  I[u, t] / max_{v of same type as u}  I[v, t]

    attn and MLP units are normalized separately so the 1/d_u bias of
    consistency_share (lower MLP importance) does not suppress MLPs.
    The result stays in [0, 1].
    """
    if unit_meta is None:
        # No type info: fall back to non-stratified (per-task) max-normalization
        max_per_task = importance.max(axis=0, keepdims=True)
        max_per_task = np.maximum(max_per_task, 1e-12)
        norm_imp = importance / max_per_task
    else:
        # Group unit indices by type
        types = [m.get("unit_type", "unknown") for m in unit_meta]
        type_groups: Dict[str, List[int]] = {}
        for i, t in enumerate(types):
            type_groups.setdefault(t, []).append(i)

        # For each (type, task), compute max importance, then normalize
        norm_imp = np.zeros_like(importance, dtype=np.float64)
        for t_name, idxs in type_groups.items():
            idx_arr = np.asarray(idxs, dtype=np.int64)
            sub = importance[idx_arr]                                  # (n_t, T)
            max_per_task = sub.max(axis=0, keepdims=True)              # (1, T)
            max_per_task = np.maximum(max_per_task, 1e-12)
            norm_imp[idx_arr] = sub / max_per_task

    # Outer product over tasks (per unit): (U, T, 1) x (U, 1, T) -> (U, T, T)
    imp_outer = norm_imp[:, :, None] * norm_imp[:, None, :]
    return np.sqrt(np.maximum(imp_outer, 0.0)) * overlap


def compute_features(
    overlap:    np.ndarray,      # (U, T, T), raw or weighted overlap
    importance: np.ndarray,      # (U, T)
    unit_meta:  list,
    unit_ids:   list,
    task_names: list,
    families:   Dict[str, List[str]],
) -> List[dict]:
    U, T, _ = overlap.shape
    name_to_idx = {n: i for i, n in enumerate(task_names)}
    fam_indices = {
        fname: [name_to_idx[n] for n in members]
        for fname, members in families.items()
    }
    triu = np.triu_indices(T, k=1)

    rows = []
    for u in range(U):
        meta = unit_meta[u]
        ov   = overlap[u]               # (T, T)
        pairs = ov[triu]                # length T*(T-1)/2

        feat = {
            "unit_id":      unit_ids[u],
            "unit_type":    meta.get("unit_type", "?"),
            "layer":        meta.get("layer", -1),
            "parent_block": meta.get("parent_block", "?"),
            "score_min":    float(pairs.min()),
            "score_mean":   float(pairs.mean()),
            "score_max":    float(pairs.max()),
        }

        # Per-family scores
        for fname, idx_list in fam_indices.items():
            within = []
            for ii, i in enumerate(idx_list):
                for j in idx_list[ii + 1:]:
                    within.append(ov[i, j])
            cross = []
            outside = [k for k in range(T) if k not in idx_list]
            for i in idx_list:
                for j in outside:
                    cross.append(ov[i, j])
            within = np.asarray(within, dtype=np.float32)
            cross  = np.asarray(cross,  dtype=np.float32)

            feat[f"within_{fname}_min"]  = float(within.min())  if len(within) else float("nan")
            feat[f"within_{fname}_mean"] = float(within.mean()) if len(within) else float("nan")
            feat[f"cross_{fname}_max"]   = float(cross.max())   if len(cross)  else float("nan")
            feat[f"cross_{fname}_mean"]  = float(cross.mean())  if len(cross)  else float("nan")

        # Importance concentration
        imp     = importance[u].astype(np.float64)
        imp_idx = int(np.argmax(imp))
        imp_max = float(imp[imp_idx])
        others  = np.delete(imp, imp_idx)
        mean_others = float(others.mean()) if len(others) else 1e-12
        feat["importance_max"]           = imp_max
        feat["importance_argmax_task"]   = task_names[imp_idx]
        feat["importance_ratio"]         = imp_max / max(mean_others, 1e-12)
        feat["importance_dominance"]     = imp_max / max(imp.sum(), 1e-12)

        # max overlap from t* (importance argmax task) to any OTHER task
        t_star = imp_idx
        out_overlaps = [ov[t_star, j] for j in range(T) if j != t_star]
        feat["task_outside_overlap_max"] = float(max(out_overlaps)) if out_overlaps else 0.0

        rows.append(feat)
    return rows


def classify(feat: dict, families: Dict, criterion: str, th: dict,
              within_overrides: Dict[str, float] = None,
              within_strict:    Dict[str, float] = None) -> Tuple[str, str, str]:
    """
    Returns (category, primary_label, family_tier).
        category      in {"universal", "family", "task", "mixed"}
        primary_label = "universal" / "family_<F>" / "task_<t>" / "mixed"
        family_tier   in {"specialized", "broad", "-"}  (only for category=family)
    """
    if within_overrides is None:
        within_overrides = WITHIN_OVERRIDES
    if within_strict is None:
        within_strict = WITHIN_STRICT
    score_key = f"score_{criterion}"
    score_uni = feat[score_key]
    th_uni    = th[f"universal_{criterion}"]

    # 1. universal: high score across all pairs
    if score_uni >= th_uni:
        return "universal", "universal", "-"

    # 2. family-specific: within signal above the family floor and clearly
    #    above the cross-family signal
    candidates = []
    for fname, members in families.items():
        if len(members) < 2:
            continue
        within = feat.get(f"within_{fname}_{criterion}", float("nan"))
        cross  = feat.get(f"cross_{fname}_max", float("nan"))
        margin = within - cross
        within_floor = within_overrides.get(fname, th["within_default"])
        if within >= within_floor and margin >= th["margin"]:
            candidates.append((fname, within, cross, margin))
    if candidates:
        # Pick the family with the largest margin
        candidates.sort(key=lambda x: x[3], reverse=True)
        best_fname = candidates[0][0]
        # Sub-tier: specialized vs broad
        within_best  = feat[f"within_{best_fname}_{criterion}"]
        strict_floor = within_strict.get(best_fname, None)
        tier = "specialized" if (strict_floor is not None and within_best >= strict_floor) else "broad"
        return "family", f"family_{best_fname}", tier

    # 3. task-specific (importance dominance + low t*-vs-others subspace overlap)
    if feat["importance_dominance"]      >= th["task_dominance"] and \
       feat["task_outside_overlap_max"]  <= th["task_outside_overlap"]:
        return "task", f"task_{feat['importance_argmax_task']}", "-"

    # 4. mixed
    return "mixed", "mixed", "-"


def write_classified(
    rows:      List[dict],
    out_csv:   Path,
    families:  Dict,
    criterion: str,
    th:        dict,
    within_overrides: Dict[str, float] = None,
    within_strict:    Dict[str, float] = None,
) -> Dict[str, int]:
    counts = {"universal": 0, "family": 0, "task": 0, "mixed": 0}
    family_counts: Dict[str, int] = {}
    task_counts:   Dict[str, int] = {}

    base_cols = ["unit_id", "unit_type", "layer", "parent_block"]
    score_cols = ["score_min", "score_mean", "score_max"]
    imp_cols  = ["importance_max", "importance_argmax_task", "importance_ratio",
                 "importance_dominance", "task_outside_overlap_max"]
    fam_cols  = []
    for f in families:
        for k in ("within_min", "within_mean", "cross_max", "cross_mean"):
            fam_cols.append(f"{k.replace('within', f'within_{f}').replace('cross', f'cross_{f}')}")
    # Re-derive cleanly
    fam_cols = []
    for f in families:
        fam_cols.extend([f"within_{f}_min", f"within_{f}_mean",
                         f"cross_{f}_max", f"cross_{f}_mean"])

    cls_cols = ["category", "primary_label", "family_tier"]
    all_cols = base_cols + score_cols + imp_cols + fam_cols + cls_cols

    family_tier_counts: Dict[str, Dict[str, int]] = {}   # fname -> {specialized: n, broad: n}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(all_cols)
        for r in rows:
            cat, label, tier = classify(r, families, criterion, th,
                                         within_overrides=within_overrides,
                                         within_strict=within_strict)
            counts[cat] += 1
            if cat == "family":
                fname = label.split("_", 1)[1]
                family_counts[fname] = family_counts.get(fname, 0) + 1
                family_tier_counts.setdefault(fname, {"specialized": 0, "broad": 0})
                family_tier_counts[fname][tier] += 1
            elif cat == "task":
                tname = label.split("_", 1)[1]
                task_counts[tname] = task_counts.get(tname, 0) + 1
            row_out = [r.get(c, "") for c in all_cols[:-3]] + [cat, label, tier]
            w.writerow(row_out)

    return {
        **counts,
        "_family_breakdown":      family_counts,
        "_family_tier_breakdown": family_tier_counts,
        "_task_breakdown":        task_counts,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_weighted", action="store_true",
                    help="Use importance-weighted overlap (sqrt(I·I) * overlap, "
                         "max-normalized per task) for all unit-level features. "
                         "Output goes to classify_weighted/ subdir.")
    ap.add_argument("--auto_threshold", action="store_true", default=False,
                    help="Set universal/family thresholds from data quantiles "
                         "(top-5% within-F floor, top-1% specialized tier).")
    ap.add_argument("--q_floor",  type=float, default=0.95,
                    help="Quantile for family/universal floor (default 0.95).")
    ap.add_argument("--q_strict", type=float, default=0.99,
                    help="Quantile for specialized tier (default 0.99).")
    args = ap.parse_args()

    importance_variants = ["consistency_share", "sigma_weighted"]
    family_variants     = {"v1": FAMILIES_V1, "v2": FAMILIES_V2, "v3": FAMILIES_V3}
    criteria            = ["min", "mean"]

    out_subdir = "classify_weighted" if args.use_weighted else "classify"

    for imp_variant in importance_variants:
        pkl = RUN_ROOT / imp_variant / "cross_task_result.pkl"
        if not pkl.exists():
            print(f"[skip] {pkl} not found")
            continue

        out_dir = RUN_ROOT / imp_variant / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*72}\n  importance variant: {imp_variant}")
        print(f"  pkl: {pkl}")
        print(f"  use_weighted: {args.use_weighted}  ->  out: {out_dir}")
        print(f"{'='*72}")
        result     = pickle.loads(pkl.read_bytes())
        tensor     = result["tensor"]            # nested dict with overlap/importance/etc.
        overlap    = tensor["overlap"]
        importance = tensor["importance"]
        unit_ids   = tensor["unit_ids"]
        unit_meta  = tensor["unit_meta"]
        task_names = tensor["task_names"]
        if args.use_weighted:
            print(f"  computing weighted_overlap = sqrt(norm_I · norm_I) · overlap "
                  f"(type-stratified normalization)...")
            overlap = compute_weighted_overlap(overlap, importance, unit_meta=unit_meta)
            print(f"  weighted_overlap range: min={overlap.min():.4f}  "
                  f"mean={overlap.mean():.4f}  max={overlap.max():.4f}")

        # Auto thresholds (data-driven)
        TH_local = dict(TH)             # working copy, may be updated below
        WITHIN_OVERRIDES_local = dict(WITHIN_OVERRIDES)
        WITHIN_STRICT_local    = dict(WITHIN_STRICT)
        if args.auto_threshold:
            name_to_idx = {n: i for i, n in enumerate(task_names)}
            fam_idx_v1 = {f: [name_to_idx[m] for m in members]
                          for f, members in FAMILIES_V1.items()}
            auto = compute_auto_thresholds(
                overlap, fam_idx_v1,
                quantile_floor=args.q_floor,
                quantile_strict=args.q_strict,
            )
            print(f"\n  -- Auto thresholds (q_floor={args.q_floor}, "
                  f"q_strict={args.q_strict}) --")
            print(f"    universal_min  : {auto['universal_min']:.4f}")
            print(f"    universal_mean : {auto['universal_mean']:.4f}")
            print(f"    within_min     : {auto['within_min']}")
            print(f"    within_mean    : {auto['within_mean']}")
            print(f"    strict_min     : {auto['strict_min']}")
            print(f"    strict_mean    : {auto['strict_mean']}")
            TH_local["universal_min"]  = auto["universal_min"]
            TH_local["universal_mean"] = auto["universal_mean"]
            # Per-family floors from the min-criterion quantiles; the
            # mean-criterion versions are selected per criterion below.
            WITHIN_OVERRIDES_local = dict(auto["within_min"])
            WITHIN_STRICT_local    = dict(auto["strict_min"])
        print(f"  shape: overlap={overlap.shape}  importance={importance.shape}")
        print(f"  tasks: {task_names}")

        # Pre-compute per-family features once for each family layout
        features_by_layout = {
            fv_name: compute_features(overlap, importance, unit_meta,
                                      unit_ids, task_names, fv_fam)
            for fv_name, fv_fam in family_variants.items()
        }

        summary_rows = []
        for fv_name, fv_fam in family_variants.items():
            for crit in criteria:
                tag = f"{fv_name}_{crit}"
                out_csv = out_dir / f"{tag}.csv"
                # Pick per-criterion overrides (auto thresholds vs hardcoded)
                if args.auto_threshold:
                    overrides = auto[f"within_{crit}"]
                    strict    = auto[f"strict_{crit}"]
                else:
                    overrides = WITHIN_OVERRIDES_local
                    strict    = WITHIN_STRICT_local
                cnts = write_classified(
                    features_by_layout[fv_name], out_csv, fv_fam, crit, TH_local,
                    within_overrides=overrides, within_strict=strict,
                )
                print(f"\n  variant: {tag}")
                print(f"    universal       : {cnts['universal']:3d}")
                print(f"    family-specific : {cnts['family']:3d}    "
                      f"breakdown={cnts['_family_breakdown']}")
                tier_bd = cnts['_family_tier_breakdown']
                if tier_bd:
                    for fname, tiers in sorted(tier_bd.items()):
                        if tiers["specialized"] + tiers["broad"] > 0:
                            print(f"      {fname:8s} : specialized={tiers['specialized']}  "
                                  f"broad={tiers['broad']}")
                print(f"    task-specific   : {cnts['task']:3d}    "
                      f"breakdown={cnts['_task_breakdown']}")
                print(f"    mixed           : {cnts['mixed']:3d}")
                print(f"    [saved] {out_csv}")
                summary_rows.append({
                    "variant":              tag,
                    "universal":            cnts["universal"],
                    "family":               cnts["family"],
                    "task_specific":        cnts["task"],
                    "mixed":                cnts["mixed"],
                    "family_breakdown":      str(cnts["_family_breakdown"]),
                    "family_tier_breakdown": str(cnts["_family_tier_breakdown"]),
                    "task_breakdown":        str(cnts["_task_breakdown"]),
                })

        # Save summary
        sum_csv = out_dir / "summary.csv"
        with open(sum_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"\n  [saved summary] {sum_csv}")

        # Save legend
        legend = out_dir / "legend.txt"
        with open(legend, "w") as f:
            f.write("Cross-task unit classification: legend\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Importance variant: {imp_variant}\n\n")
            f.write("Thresholds:\n")
            for k, v in TH.items():
                f.write(f"  {k:24s} = {v}\n")
            f.write("\nPer-family within overrides (raises within floor for isolated families):\n")
            for fname, v in WITHIN_OVERRIDES.items():
                f.write(f"  within[{fname:8s}] = {v}\n")
            f.write("\nFamily strict tier (within >= this  ->  family_tier='specialized', else 'broad'):\n")
            if WITHIN_STRICT:
                for fname, v in WITHIN_STRICT.items():
                    f.write(f"  strict[{fname:8s}] = {v}\n")
            else:
                f.write("  (none configured)\n")
            f.write("\nFamily layout v1 (orphan singletons):\n")
            for fname, members in FAMILIES_V1.items():
                f.write(f"  {fname:8s} = {members}\n")
            f.write("  (singletons: ioi_strong, col_obj; not family-eligible)\n\n")
            f.write("Family layout v2 (orphans grouped):\n")
            for fname, members in FAMILIES_V2.items():
                f.write(f"  {fname:8s} = {members}\n")
            f.write("\nClassification priority:\n")
            f.write("  1. universal       : score_<crit> >= θ_universal_<crit>\n")
            f.write("  2. family-F        : within_F_<crit> >= within_floor[F]  (default θ_within_default,\n")
            f.write("                                                            override for isolated families)\n")
            f.write("                       AND  within_F_<crit> - cross_F_max >= θ_margin\n")
            f.write("                       (pick family with largest margin if multiple)\n")
            f.write("  3. task-t*         : importance_dominance >= θ_task_dominance         (t* gets >=30% of\n")
            f.write("                                                                          unit's total imp)\n")
            f.write("                       AND task_outside_overlap_max <= θ_task_outside_overlap\n")
            f.write("                                                       (no other task strongly overlaps t*)\n")
            f.write("  4. mixed           : otherwise\n")
        print(f"  [saved legend ] {legend}")


if __name__ == "__main__":
    main()
