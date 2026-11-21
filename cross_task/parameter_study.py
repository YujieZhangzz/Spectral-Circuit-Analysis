"""Sensitivity analysis for the five unit-classification thresholds
(universal floor, within-family floor, margin δ, task dominance, task outside
overlap), using consistency_share importance and GT / SVA families.

Reads cross_task/<CT_RUN>/consistency_share/cross_task_result.pkl and writes
1-D sweeps, 2-D sweeps, ±10/20% perturbation tables and the default
classification to consistency_share/parameter_study/, for both the min and
mean criteria.

Usage: CT_RUN=run_gpt_12tasks python cross_task/parameter_study.py
"""
from __future__ import annotations
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cross_task"))
from classify_units import compute_features, classify   # noqa: E402

RUN_ROOT = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")
PKL      = RUN_ROOT / "consistency_share" / "cross_task_result.pkl"
OUT_DIR  = RUN_ROOT / "consistency_share" / "parameter_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "GT":  ["gt", "gt_price"],
    "SVA": ["sva", "sva_blimp", "sva_isare"],
}

# Default thresholds (matches classify_units.py)
TH_DEFAULT = {
    "universal_min":        0.08,
    "universal_mean":       0.20,
    "within_default":       0.10,
    "margin":               0.05,
    "task_dominance":       0.30,
    "task_outside_overlap": 0.15,
}
# In sweeps we disable per-family overrides so θ_F^floor is a single scalar.
WITHIN_OVERRIDES_OFF: Dict[str, float] = {}
WITHIN_STRICT_OFF:    Dict[str, float] = {}

CRITERION = "min"   # primary

# Sweep ranges. The universal range depends on the criterion because
# classify() reads th["universal_<criterion>"] (min: default 0.08,
# mean: default 0.20).

def get_ranges(criterion: str):
    return {
        # display name "universal" maps to TH key universal_{criterion}
        "universal": (np.linspace(0.00, 0.15, 31) if criterion == "min"
                       else np.linspace(0.00, 0.40, 41)),
        "within_default":       np.linspace(0.05, 0.50, 31),  # default 0.10
        "margin":               np.linspace(-0.05, 0.30, 36), # default 0.05
        "task_dominance":       np.linspace(0.10, 0.55, 31),  # default 0.30
        "task_outside_overlap": np.linspace(0.05, 0.50, 31),  # default 0.15
    }


def _th_key(param: str, criterion: str) -> str:
    """Map sweep display name -> TH dict key. Only `universal` is criterion-aware."""
    return f"universal_{criterion}" if param == "universal" else param

CATEGORY_KEYS = ["universal", "family_GT", "family_SVA", "task", "mixed"]
CATEGORY_COLORS = {
    "universal":  "#d62728",
    "family_GT":  "#1f77b4",
    "family_SVA": "#2ca02c",
    "task":       "#ff7f0e",
    "mixed":      "#888888",
}


def _classify_all(feats, th, criterion):
    """Run classify() on every unit. Return dict of category -> count."""
    counts = {k: 0 for k in CATEGORY_KEYS}
    for f in feats:
        cat, label, _tier = classify(
            f, FAMILIES, criterion, th,
            within_overrides=WITHIN_OVERRIDES_OFF,
            within_strict=WITHIN_STRICT_OFF,
        )
        if cat == "family":
            fname = label.split("_", 1)[1]
            key = f"family_{fname}"
            counts[key] = counts.get(key, 0) + 1
        else:
            counts[cat] = counts.get(cat, 0) + 1
    return counts


def _classify_labels(feats, th, criterion):
    """Return list of (unit_id, primary_label) at given thresholds."""
    out = []
    for f in feats:
        cat, label, _ = classify(
            f, FAMILIES, criterion, th,
            within_overrides=WITHIN_OVERRIDES_OFF,
            within_strict=WITHIN_STRICT_OFF,
        )
        out.append((f["unit_id"], label, cat))
    return out


def run_1d_sweep(feats, criterion):
    rows = []
    for param, grid in get_ranges(criterion).items():
        key = _th_key(param, criterion)
        for v in grid:
            th = dict(TH_DEFAULT)
            th[key] = float(v)
            cnts = _classify_all(feats, th, criterion)
            rows.append({
                "param": param, "value": float(v), "criterion": criterion, **cnts,
            })

    csv_path = OUT_DIR / "sweep_1d.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  [saved] {csv_path}")

    # 5-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    pretty = {
        "universal":            r"$\theta_{\mathrm{univ}}$",
        "within_default":       r"$\theta_F^{\mathrm{floor}}$  (within_default)",
        "margin":               r"$\delta$  (margin)",
        "task_dominance":       r"$\theta_{\mathrm{dom}}$  (task_dominance)",
        "task_outside_overlap": r"$\theta_{\mathrm{out}}$  (task_outside_overlap)",
    }
    ranges_for_crit = get_ranges(criterion)
    for i, param in enumerate(ranges_for_crit):
        ax = axes[i]
        sel = [r for r in rows if r["param"] == param]
        xs = [r["value"] for r in sel]
        for k in CATEGORY_KEYS:
            ys = [r[k] for r in sel]
            ax.plot(xs, ys, label=k, color=CATEGORY_COLORS[k], lw=2)
        default_v = TH_DEFAULT[_th_key(param, criterion)]
        ax.axvline(default_v, color="black", ls="--", lw=1, alpha=0.6,
                   label=f"default={default_v:.2f}")
        ax.set_xlabel(pretty[param])
        ax.set_ylabel("unit count")
        ax.set_title(pretty[param])
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
    axes[-1].axis("off")
    fig.suptitle(f"1-D parameter sweep  (criterion={criterion}, families=GT/SVA, "
                 f"importance=consistency_share)", fontsize=12)
    fig.tight_layout()
    png = OUT_DIR / f"sweep_1d_{criterion}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {png}")


def _run_2d(feats, criterion, x_param, y_param, x_grid, y_grid,
            x_label, y_label, tag):
    """Generic 2-D sweep. Records universal / family-GT / family-SVA / family-total."""
    n_y, n_x = len(y_grid), len(x_grid)
    U   = np.zeros((n_y, n_x), dtype=np.int32)
    FGT = np.zeros_like(U); FSVA = np.zeros_like(U); FTOT = np.zeros_like(U)
    for i, yv in enumerate(y_grid):
        for j, xv in enumerate(x_grid):
            th = dict(TH_DEFAULT)
            th[x_param] = float(xv)
            th[y_param] = float(yv)
            cnts = _classify_all(feats, th, criterion)
            U[i, j]    = cnts["universal"]
            FGT[i, j]  = cnts["family_GT"]
            FSVA[i, j] = cnts["family_SVA"]
            FTOT[i, j] = cnts["family_GT"] + cnts["family_SVA"]

    rows = []
    for i, yv in enumerate(y_grid):
        for j, xv in enumerate(x_grid):
            rows.append({
                x_param: float(xv), y_param: float(yv),
                "universal":  int(U[i, j]),
                "family_GT":  int(FGT[i, j]),
                "family_SVA": int(FSVA[i, j]),
                "family_total": int(FTOT[i, j]),
            })
    csv_path = OUT_DIR / f"sweep_2d_{tag}_{criterion}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  [saved] {csv_path}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    panels = [(U, "Universal"), (FGT, "Family-GT"),
              (FSVA, "Family-SVA"), (FTOT, "Family total")]
    for ax, (mat, title) in zip(axes, panels):
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis",
                       extent=[x_grid.min(), x_grid.max(),
                               y_grid.min(), y_grid.max()])
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.scatter([TH_DEFAULT[x_param]], [TH_DEFAULT[y_param]],
                   c="red", s=80, marker="x", label="default")
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"2-D sweep  {tag}  (criterion={criterion})", fontsize=12)
    fig.tight_layout()
    png = OUT_DIR / f"sweep_2d_{tag}_{criterion}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {png}")


def run_2d_sweep(feats, criterion):
    """Two 2-D sweeps:
       (a) θ_univ x δ      universal vs family cascade
       (b) θ_F^floor x δ   both gates of the family rule jointly
    """
    # (a) θ_univ x δ, sweeping the criterion-specific universal threshold
    univ_key   = f"universal_{criterion}"
    y_grid_uni = (np.linspace(0.00, 0.15, 25) if criterion == "min"
                   else np.linspace(0.00, 0.40, 25))
    _run_2d(
        feats, criterion,
        x_param="margin",        y_param=univ_key,
        x_grid=np.linspace(-0.05, 0.30, 25),
        y_grid=y_grid_uni,
        x_label=r"$\delta$ (margin)",
        y_label=r"$\theta_{\mathrm{univ}}$",
        tag="univ_x_margin",
    )
    # (b) θ_F^floor x δ (the family rule's two AND conditions)
    _run_2d(
        feats, criterion,
        x_param="margin",         y_param="within_default",
        x_grid=np.linspace(-0.05, 0.30, 25),
        y_grid=np.linspace( 0.05, 0.40, 25),
        x_label=r"$\delta$ (margin)",
        y_label=r"$\theta_F^{\mathrm{floor}}$ (within_default)",
        tag="floor_x_margin",
    )


def run_perturbation(feats, criterion):
    """At default, perturb each parameter by ±10% / ±20%; report flips."""
    base_labels = _classify_labels(feats, TH_DEFAULT, criterion)
    base = {uid: (lab, cat) for uid, lab, cat in base_labels}
    base_counts = _classify_all(feats, TH_DEFAULT, criterion)

    rows = []
    rows.append({"perturb": "DEFAULT", "param": "-", "value": "-",
                 "delta_universal": 0, "delta_family_GT": 0,
                 "delta_family_SVA": 0, "delta_task": 0, "delta_mixed": 0,
                 "n_flips": 0})

    for param in get_ranges(criterion):
        key = _th_key(param, criterion)
        center = TH_DEFAULT[key]
        for pct in [-0.20, -0.10, +0.10, +0.20]:
            th = dict(TH_DEFAULT)
            new_val = center + abs(center) * pct if center != 0 else center + pct * 0.05
            th[key] = new_val
            new_labels = _classify_labels(feats, th, criterion)
            new = {uid: (lab, cat) for uid, lab, cat in new_labels}
            new_counts = _classify_all(feats, th, criterion)
            n_flips = sum(1 for uid in base
                          if base[uid][0] != new[uid][0])
            rows.append({
                "perturb": f"{pct:+.0%}", "param": param,
                "value": round(new_val, 4),
                "delta_universal":  new_counts["universal"]   - base_counts["universal"],
                "delta_family_GT":  new_counts["family_GT"]   - base_counts["family_GT"],
                "delta_family_SVA": new_counts["family_SVA"]  - base_counts["family_SVA"],
                "delta_task":       new_counts["task"]        - base_counts["task"],
                "delta_mixed":      new_counts["mixed"]       - base_counts["mixed"],
                "n_flips":          n_flips,
            })
    csv_path = OUT_DIR / f"perturb_default_{criterion}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  [saved] {csv_path}")
    return base_counts, rows


def write_default_csv(feats, criterion):
    """Save baseline classification (the reference table for everything else)."""
    rows = []
    for f in feats:
        cat, label, _ = classify(
            f, FAMILIES, criterion, TH_DEFAULT,
            within_overrides=WITHIN_OVERRIDES_OFF,
            within_strict=WITHIN_STRICT_OFF,
        )
        rows.append({
            "unit_id":     f["unit_id"],
            "unit_type":   f["unit_type"],
            "layer":       f["layer"],
            "category":    cat,
            "label":       label,
            "score_min":   round(f["score_min"], 4),
            "score_mean":  round(f["score_mean"], 4),
            "within_GT_min":  round(f["within_GT_min"], 4),
            "within_SVA_min": round(f["within_SVA_min"], 4),
            "cross_GT_max":   round(f["cross_GT_max"], 4),
            "cross_SVA_max":  round(f["cross_SVA_max"], 4),
            "importance_dominance":     round(f["importance_dominance"], 4),
            "importance_argmax_task":   f["importance_argmax_task"],
            "task_outside_overlap_max": round(f["task_outside_overlap_max"], 4),
        })
    csv_path = OUT_DIR / f"default_classification_{criterion}.csv"
    with open(csv_path, "w", newline="") as f_:
        w = csv.DictWriter(f_, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  [saved] {csv_path}")


def main():
    print(f"Loading {PKL}")
    t = pickle.loads(PKL.read_bytes())["tensor"]
    overlap, importance = t["overlap"], t["importance"]
    unit_ids, unit_meta, task_names = t["unit_ids"], t["unit_meta"], t["task_names"]
    print(f"  overlap={overlap.shape}  importance={importance.shape}  "
          f"tasks={len(task_names)}  units={len(unit_ids)}")

    feats = compute_features(overlap, importance, unit_meta, unit_ids,
                             task_names, FAMILIES)
    print(f"  computed features for {len(feats)} units\n")

    for criterion in ["min", "mean"]:
        print(f"criterion={criterion}:")
        write_default_csv(feats, criterion)
        run_1d_sweep(feats, criterion)
        run_2d_sweep(feats, criterion)
        base, _ = run_perturbation(feats, criterion)
        print(f"  baseline: {base}\n")


if __name__ == "__main__":
    main()
