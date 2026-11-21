"""
Cross-task subspace analysis for a small, hand-specified set of tasks.

Loads `weighted_unit_subspaces.npz` for each task in TASK_PATHS (edit to
match your results/<experiment_name>/ dirs), runs the cross-task analysis and
writes cross_task_result.pkl, task_similarity.csv and universal_units.csv to
the current directory.

Usage:
    python cross_task/analyze_cross_task.py --inspect   # check npz schemas
    python cross_task/analyze_cross_task.py
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from weighted_unit_svd import load_task_subspaces
from cross_task_analysis import run_analysis


# Paths to per-task subspace files (order does not matter).
TASK_PATHS = {
    "ioi":              "results/gpt2_small_N50_M6_mlp_grad_avg/weighted_unit_subspaces.npz",
    "gendered_pronoun": "results/gpt2_small_gp_N50_M6/weighted_unit_subspaces.npz",
    "country_capital":  "results/gpt2_small_cap_N50_M6/weighted_unit_subspaces.npz",
    "lama_p19":         "results/gpt2_small_p19_N50_M6/weighted_unit_subspaces.npz",
}

# Output paths
RESULT_PKL_PATH       = "cross_task_result.pkl"
SIMILARITY_CSV_PATH   = "task_similarity.csv"
UNIVERSAL_CSV_PATH    = "universal_units.csv"

# Analysis config
SIMILARITY_MODE       = "importance_weighted"   # uses per-unit importance scores
BASELINE_N_SAMPLES    = 200
BASELINE_QUANTILE     = 0.99
TOP_K_TO_REPORT       = 30


# Schema inspection

def inspect_npz(path: str):
    """
    Print the schema of a .npz file to check it matches what
    load_task_subspaces expects. Expected per-unit fields:

        __unit_ids__              : array of unit identifiers
        <unit_id>__U              : (d_k, rho) float16  left singular vectors
        <unit_id>__sigma          : (rho,)    float32  singular values
        <unit_id>__rho            : int32              effective rank
        <unit_id>__d_k            : int32              unit dimension
        <unit_id>__var            : float32            variance explained
        <unit_id>__type           : str                unit_type
        <unit_id>__layer          : int32              layer index
    """
    print(f"\n=== Inspecting {path} ===")
    z = np.load(path, allow_pickle=True)

    keys = list(z.keys())
    print(f"Total keys: {len(keys)}")

    if "__unit_ids__" in keys:
        unit_ids = list(z["__unit_ids__"])
        print(f"__unit_ids__ found: {len(unit_ids)} units")
        print(f"  first 5 unit_ids: {unit_ids[:5]}")

        # Show schema for first unit
        first = unit_ids[0]
        unit_keys = [k for k in keys if k.startswith(f"{first}__")]
        print(f"\nSchema for unit {first!r}:")
        for k in unit_keys:
            arr = z[k]
            shape = getattr(arr, "shape", "()")
            dtype = getattr(arr, "dtype", type(arr).__name__)
            preview = ""
            if hasattr(arr, "size") and arr.size <= 4:
                preview = f"  value={arr}"
            print(f"  {k:<40s}  shape={shape}  dtype={dtype}{preview}")
    else:
        # Unknown schema
        print("WARNING: '__unit_ids__' not found.")
        print("The npz file uses a different schema than load_task_subspaces "
              "expects. First 20 keys for inspection:")
        for k in keys[:20]:
            arr = z[k]
            shape = getattr(arr, "shape", "()")
            dtype = getattr(arr, "dtype", type(arr).__name__)
            print(f"  {k:<40s}  shape={shape}  dtype={dtype}")
        print()
        print("For a different schema, write an adapter that converts the npz")
        print("into the dict format expected by run_analysis() (see")
        print("cross_task_analysis.run_analysis docstring).")


def inspect_all():
    for task_name, path in TASK_PATHS.items():
        if not Path(path).exists():
            print(f"\n=== {task_name}: MISSING: {path} ===")
            continue
        inspect_npz(path)


# Loading with sanity checks

def load_all_tasks():
    """Load all per-task subspaces and validate that they share unit ids."""
    print("Loading task subspaces:")
    task_subspaces = {}
    for task_name, path in TASK_PATHS.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Task '{task_name}': file not found at {path}"
            )
        subs = load_task_subspaces(path)
        task_subspaces[task_name] = subs
        print(f"  {task_name:>20s}: {len(subs):4d} units  <-  {path}")

    # Validate that all tasks share the same unit set
    print("\nValidating unit consistency across tasks:")
    unit_sets = {t: set(s.keys()) for t, s in task_subspaces.items()}
    common = set.intersection(*unit_sets.values())
    union  = set.union(*unit_sets.values())

    print(f"  union of unit ids:        {len(union)}")
    print(f"  intersection (common):    {len(common)}")

    if len(common) < len(union):
        print(f"  WARNING: {len(union) - len(common)} unit(s) missing in "
              f"at least one task")
        for task_name, s in unit_sets.items():
            missing = union - s
            if missing:
                print(f"    {task_name}: missing {len(missing)}: "
                      f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        # Restrict to common units
        print(f"  Restricting analysis to {len(common)} common units")
        task_subspaces = {
            t: {u: s[u] for u in common}
            for t, s in task_subspaces.items()
        }

    # Sanity check on rho distribution per task
    print("\nRho statistics per task:")
    for t, subs in task_subspaces.items():
        rhos = [r['rho'] for r in subs.values()]
        d_ks = [r['d_k'] for r in subs.values()]
        zero_rho = sum(1 for r in rhos if r == 0)
        print(f"  {t:>20s}: rho min={min(rhos):2d} max={max(rhos):2d} "
              f"mean={np.mean(rhos):5.1f}  d_k min={min(d_ks):6d} "
              f"max={max(d_ks):>9d}  zero-rho units={zero_rho}")

    return task_subspaces


# Result export

def save_similarity_csv(result, path: str):
    sim = result['task_similarity']
    tn  = result['task_names']
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + tn)
        for i, t in enumerate(tn):
            w.writerow([t] + [f"{sim[i, j]:.6f}" for j in range(len(tn))])
    print(f"  saved similarity matrix -> {path}")


def save_universal_csv(result, path: str, top_k: int = 30):
    """Save the top-k universal units (sorted by mean overlap) to CSV."""
    ranked = sorted(result['ranked_units'], key=lambda r: -r['score_mean'])
    ranked = ranked[:top_k]
    n_tasks = len(result['task_names'])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ['rank', 'unit_id', 'label', 'unit_type', 'layer',
                  'score_mean', 'score_min', 'score_max',
                  'n_significant', 'n_pairs_total']
        # Add per-task rho columns
        for t in result['task_names']:
            header.append(f'rho_{t}')
        w.writerow(header)

        for rank, r in enumerate(ranked):
            meta = r['unit_meta']
            row = [
                rank, str(r['unit_id']),
                str(meta.get('label', '')),
                str(meta.get('unit_type', '')),
                meta.get('layer', -1),
                f"{r['score_mean']:.6f}", f"{r['score_min']:.6f}",
                f"{r['score_max']:.6f}",
                r['n_significant'], r['n_pairs_total'],
            ]
            row += list(r['rho_per_task'])
            w.writerow(row)
    print(f"  saved top {top_k} universal units -> {path}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                        help="Just inspect the schema of the npz files and exit")
    parser.add_argument("--mode", default=SIMILARITY_MODE,
                        choices=["mean", "significant_mean",
                                 "significant_count", "top_k_mean"],
                        help="Task similarity aggregation mode")
    args = parser.parse_args()

    if args.inspect:
        inspect_all()
        return

    # Load
    print("=" * 70)
    print("STEP 1: load per-task subspaces")
    print("=" * 70)
    task_subspaces = load_all_tasks()

    # Analyze
    print()
    print("=" * 70)
    print(f"STEP 2: cross-task analysis (mode={args.mode!r})")
    print("=" * 70)
    result = run_analysis(
        task_subspaces=task_subspaces,
        baseline_n_samples=BASELINE_N_SAMPLES,
        baseline_quantile=BASELINE_QUANTILE,
        similarity_mode=args.mode,
        verbose=True,
    )

    # Save
    print()
    print("=" * 70)
    print("STEP 3: save results")
    print("=" * 70)
    with open(RESULT_PKL_PATH, "wb") as f:
        pickle.dump(result, f)
    print(f"  saved full result pickle -> {RESULT_PKL_PATH}")
    save_similarity_csv(result, SIMILARITY_CSV_PATH)
    save_universal_csv(result, UNIVERSAL_CSV_PATH, top_k=TOP_K_TO_REPORT)

    # Summary
    print()
    print("=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    n_units = len(result['tensor']['unit_ids'])
    n_total_pairs = result['universality']['n_pairs_total']
    n_universal_strict = int(
        (result['universality']['n_significant'] == n_total_pairs).sum()
    )
    print(f"\nUniversality (out of {n_units} units, {n_total_pairs} task pairs):")
    print(f"  Strict universal (all {n_total_pairs} pairs significant): "
          f"{n_universal_strict}")

    print(f"\nTop 10 universal units (by mean overlap):")
    ranked_by_mean = sorted(result['ranked_units'],
                            key=lambda r: -r['score_mean'])[:10]
    for r in ranked_by_mean:
        meta = r['unit_meta']
        # Prefer the human-readable label (e.g. "L11H9"), fall back to the id
        label = str(meta.get('label', r['unit_id']))
        layer = meta.get('layer', -1)
        utype = str(meta.get('unit_type', '?'))
        print(f"  {label:>12s}  L{layer:<2d}  {utype:<10s}  "
              f"mean={r['score_mean']:.3f}  min={r['score_min']:.3f}  "
              f"sig={r['n_significant']}/{r['n_pairs_total']}")

    print(f"\nTask similarity matrix:")
    sim = result['task_similarity']
    tn  = result['task_names']
    print(f"  {'':>22s}", " ".join(f"{n[:12]:>12s}" for n in tn))
    for i, t in enumerate(tn):
        row = " ".join(f"{sim[i, j]:>12.4f}" for j in range(len(tn)))
        print(f"  {t:>22s}", row)

    print(f"\nHierarchy (read top-down for merge order):")
    Z = result['task_linkage']
    n_t = len(tn)
    for step, row in enumerate(Z):
        a, b, dist, n_obs = int(row[0]), int(row[1]), row[2], int(row[3])
        a_name = tn[a] if a < n_t else f"cluster_{a}"
        b_name = tn[b] if b < n_t else f"cluster_{b}"
        print(f"  merge {step}: {a_name} + {b_name}  at distance {dist:.4f}")

    print()
    print("Done. Inspect cross_task_result.pkl for full analysis output.")


if __name__ == "__main__":
    main()