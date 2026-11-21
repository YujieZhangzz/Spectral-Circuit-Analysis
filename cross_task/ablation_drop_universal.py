"""
Ablation: drop the universal units and recompute task similarity.

If within-family similarities stay high while cross-family similarities
collapse after dropping the universal set, the task families reflect shared
task content rather than shared plumbing heads.

Environment variables:
    CT_RUN        run dir under cross_task/  (default: run_gpt_12tasks)
    RESULTS_DIR   parent dir of per-task <prefix><task><suffix>/ subspaces
                  (default: <repo>/results)
    TASK_PREFIX   per-task dir prefix    (default: gpt2_small_)
    TASK_SUFFIX   per-task dir suffix    (default: _N100_M5)
    DROP_TOP_N    drop top-N universals  (default: 0 = drop all classified universals)
    SIMILARITY    comma-separated similarity modes (default: importance_weighted,significant_mean)

Usage:
    python cross_task/ablation_drop_universal.py
    CT_RUN=run_pythia_12tasks TASK_PREFIX=pythia_ python cross_task/ablation_drop_universal.py
"""
from __future__ import annotations
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cross_task.weighted_unit_svd  import load_task_subspaces
from cross_task.cross_task_analysis import run_analysis


CT_RUN      = os.environ.get("CT_RUN", "run_gpt_12tasks")
RUN_DIR     = REPO_ROOT / "cross_task" / CT_RUN
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(REPO_ROOT / "results")))
TASK_PREFIX = os.environ.get("TASK_PREFIX", "gpt2_small_")
TASK_SUFFIX = os.environ.get("TASK_SUFFIX", "_N100_M5")
DROP_TOP_N  = int(os.environ.get("DROP_TOP_N", "0"))   # 0 = all classified universals
SIM_MODES   = [s.strip() for s in
               os.environ.get("SIMILARITY",
                              "importance_weighted,significant_mean").split(",")]


def discover_tasks(pkl_path: Path) -> list[str]:
    """Use the run's pkl to enumerate tasks."""
    return list(pickle.loads(pkl_path.read_bytes())["task_names"])


def discover_universals(run_dir: Path, top_n: int) -> list[int]:
    """Read universal unit ids from classify_weighted/v1_min.csv. If top_n>0,
    return the top-N by score_min from universal_units_full.csv instead."""
    if top_n > 0:
        path = run_dir / "consistency_share" / "universal_units_full.csv"
        rows = list(csv.DictReader(open(path)))
        rows.sort(key=lambda r: -float(r["score_min"]))
        return [int(r["unit_id"]) for r in rows[:top_n]]
    path = run_dir / "consistency_share" / "classify_weighted" / "v1_min.csv"
    rows = list(csv.DictReader(open(path)))
    return [int(r["unit_id"]) for r in rows if r["category"] == "universal"]


def task_subs_path(task: str) -> Path:
    return RESULTS_DIR / f"{TASK_PREFIX}{task}{TASK_SUFFIX}" / "weighted_unit_subspaces.npz"


def filter_units(subs: dict, drop: set) -> dict:
    return {uid: v for uid, v in subs.items() if int(uid) not in drop}


def print_matrix(sim, names, header):
    print(f"\n  --- {header} ---")
    print(f"  task_names: {names}")
    for i, n in enumerate(names):
        row = "  ".join(f"{sim[i,j]:.3f}" for j in range(len(names)))
        print(f"    {n:>14} | {row}")


def family_summary(sim, names, families):
    """Return dict: {family: (within_mean, cross_mean, ratio)}."""
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for fname, members in families.items():
        ids = [idx[m] for m in members if m in idx]
        if len(ids) < 2:
            continue
        within = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                within.append(sim[ids[a], ids[b]])
        cross = []
        in_set = set(ids)
        out_ids = [i for i in range(len(names)) if i not in in_set]
        for a in ids:
            for b in out_ids:
                cross.append(sim[a, b])
        within = np.array(within); cross = np.array(cross)
        out[fname] = (within.mean(), cross.mean(),
                      within.mean() / cross.mean() if cross.mean() else float("inf"))
    return out


FAMILIES = {
    "SVA":     ["sva", "sva_blimp", "sva_isare"],
    "PRONOUN": ["gender_bias", "gp", "person_pro"],
    "COUNTRY": ["country_cap", "person_country"],
    "GT":      ["gt", "gt_price"],
}


def main():
    pkl = RUN_DIR / "consistency_share" / "cross_task_result.pkl"
    if not pkl.exists():
        print(f"[fatal] {pkl} not found", file=sys.stderr); sys.exit(1)

    tasks = discover_tasks(pkl)
    universals = discover_universals(RUN_DIR, DROP_TOP_N)

    print(f"run_dir       = {RUN_DIR}")
    print(f"results_dir   = {RESULTS_DIR}")
    print(f"task_prefix   = {TASK_PREFIX!r}   suffix = {TASK_SUFFIX!r}")
    print(f"tasks ({len(tasks)})  = {tasks}")
    print(f"drop_top_n    = {DROP_TOP_N}  ({'top-N by score_min' if DROP_TOP_N else 'all classified universals'})")
    print(f"universals to drop ({len(universals)}) = {sorted(universals)}\n")

    print("Loading + filtering each task...")
    full_subs, drop_subs = {}, {}
    drop_set = set(universals)
    for task in tasks:
        path = task_subs_path(task)
        if not path.exists():
            print(f"  [warn] {task:<14s}  missing {path}, skipping")
            continue
        subs = load_task_subspaces(str(path))
        flt  = filter_units(subs, drop_set)
        n_dropped = len(subs) - len(flt)
        print(f"  {task:<14s}  units {len(subs)} -> {len(flt)}  (dropped {n_dropped})")
        full_subs[task]  = subs
        drop_subs[task]  = flt

    if not drop_subs:
        print("[fatal] no per-task subspaces loaded; check RESULTS_DIR/TASK_PREFIX",
              file=sys.stderr); sys.exit(1)

    for mode in SIM_MODES:
        print(f"\n{'=' * 72}")
        print(f"  similarity_mode = {mode}")
        print(f"{'=' * 72}")

        result_full = run_analysis(full_subs, baseline_n_samples=200,
                                   baseline_quantile=0.99,
                                   similarity_mode=mode, verbose=False)
        result_drop = run_analysis(drop_subs, baseline_n_samples=200,
                                   baseline_quantile=0.99,
                                   similarity_mode=mode, verbose=False)

        names_full = result_full["task_names"]
        names_drop = result_drop["task_names"]
        sim_full = result_full["task_similarity"]
        sim_drop = result_drop["task_similarity"]

        print_matrix(sim_full, names_full, "BASELINE (no drop)")
        print_matrix(sim_drop, names_drop, f"AFTER dropping {len(universals)} universals")

        # Family-level summary
        print(f"\n  --- in-family vs cross-family means ---")
        sf = family_summary(sim_full, names_full, FAMILIES)
        sd = family_summary(sim_drop, names_drop, FAMILIES)
        print(f"  {'family':<8s}   {'within (full->drop)':<22s}  {'cross (full->drop)':<22s}  {'ratio (full->drop)':<22s}")
        for fname in FAMILIES:
            if fname not in sf or fname not in sd: continue
            wf, cf, rf = sf[fname]; wd, cd, rd = sd[fname]
            print(f"  {fname:<8s}   {wf:.3f} -> {wd:.3f}        "
                  f"{cf:.3f} -> {cd:.3f}        "
                  f"{rf:.2f}x -> {rd:.2f}x")

        # Off-diagonal mean overall
        n = sim_full.shape[0]
        iu = np.triu_indices(n, k=1)
        off_full = sim_full[iu]; off_drop = sim_drop[iu]
        print(f"\n  off-diagonal mean: {off_full.mean():.4f} -> {off_drop.mean():.4f}  "
              f"(delta = {off_drop.mean() - off_full.mean():+.4f})")
        print(f"  off-diagonal max : {off_full.max():.4f} -> {off_drop.max():.4f}")

        # Save outputs
        out_dir = RUN_DIR / "ablation_drop_universal"
        out_dir.mkdir(exist_ok=True)
        np.savetxt(out_dir / f"sim_full_{mode}.csv", sim_full,
                   header=",".join(names_full), delimiter=",", fmt="%.4f", comments="")
        np.savetxt(out_dir / f"sim_drop_{mode}.csv", sim_drop,
                   header=",".join(names_drop), delimiter=",", fmt="%.4f", comments="")
        print(f"\n  [saved] {out_dir}/sim_full_{mode}.csv")
        print(f"  [saved] {out_dir}/sim_drop_{mode}.csv")


if __name__ == "__main__":
    main()
