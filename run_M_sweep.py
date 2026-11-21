"""Sweep the number of paraphrases M on IOI and score against the IOI circuit.

For each seed, one random 30-instance subset is reused across all M values.
Each (seed, M) run goes through run.py and is evaluated with eval_recall_at_k.
Writes per-run outputs, M_sweep_metrics.csv, M_sweep_summary.csv and
M_sweep_curves.png under results/M_sweep/.

Usage:
    python run_M_sweep.py [--seeds 42] [--ms 5 10] [--eval_only | --skip_eval] [--keep_heavy]
"""

from __future__ import annotations
import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Configuration
ROOT          = Path(__file__).parent
DATA_FILE     = ROOT / "data/ioi_sampled_gpt2_200_paraphrases.json"
BASE_CFG      = ROOT / "configs/gpt2_small_ioi_strong.yaml"
SWEEP_DIR     = ROOT / "results/M_sweep"
EVAL_SCRIPT   = ROOT / "eval_recall_at_k.py"
RUN_SCRIPT    = ROOT / "run.py"

M_VALUES      = [2, 3, 5, 8, 10, 12, 15]
SEEDS         = [42, 43, 44]
N_INSTANCES   = 30


HEAVY_FILES = [
    # Phase 0 (normally not saved with save_intermediate=False)
    "gram_matrix.pt", "sign_accum.pt", "grad_avg.pt", "grad_norms.pt",
    # Phase 4 U_rho_block*.pt / global_indices_block*.pt are removed by glob
    # Phase 5b
    "weighted_unit_subspaces.npz",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_eval",  action="store_true",
                    help="Run pipelines but skip metric evaluation.")
    ap.add_argument("--eval_only",  action="store_true",
                    help="Skip pipeline runs (already done); just eval + aggregate.")
    ap.add_argument("--keep_heavy", action="store_true",
                    help="Keep large tensors (U_rho_*, grad_avg, etc.) after each run.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help=f"Subset of seeds to run (default: all = {SEEDS}). "
                         "Useful for parallelizing seeds across GPUs via "
                         "CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--ms", type=int, nargs="+", default=None,
                    help=f"Subset of M values to run (default: all = {M_VALUES}).")
    return ap.parse_args()


def prepare_seed_subsets(all_data: list) -> dict:
    """For each seed, sample N_INSTANCES indices and dump a sub-JSON."""
    sweep_data_dir = SWEEP_DIR / "_data"
    sweep_data_dir.mkdir(parents=True, exist_ok=True)

    seed_paths = {}
    for s in SEEDS:
        rng = random.Random(s)
        indices = sorted(rng.sample(range(len(all_data)), N_INSTANCES))
        subset  = [all_data[i] for i in indices]
        out     = sweep_data_dir / f"seed{s}_N{N_INSTANCES}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        seed_paths[s] = out
        print(f"  seed={s}: indices[:5]={indices[:5]} -> {out.name}")
    return seed_paths


def build_run_config(base_cfg: dict, seed: int, M: int, subset_path: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["name"] = f"M_sweep/seed{seed}_M{M:02d}"
    cfg["data"]["format"]     = "ioi_sampled"
    cfg["data"]["path"]       = str(subset_path)
    cfg["data"]["N"]          = N_INSTANCES
    cfg["data"]["M"]          = M
    cfg.setdefault("heatmap", {})["enabled"] = False   # heatmap re-runs full backward; keep off
    cfg.setdefault("output", {})["plot"]               = True   # per-run plots in plots/ dir
    cfg["output"]["save_intermediate"]                 = False
    # The sweep only evaluates ranked_units.csv from Phase 5b.
    cfg.setdefault("direct_unit_svd", {})["enabled"]   = False
    return cfg


def cleanup_heavy(out_dir: Path) -> None:
    """Delete large tensors after a run completes; keep CSVs and small json/pt."""
    for name in HEAVY_FILES:
        p = out_dir / name
        if p.exists():
            p.unlink()
    for pat in ("U_rho_block*.pt", "global_indices_block*.pt"):
        for p in out_dir.glob(pat):
            p.unlink()


def run_single(cfg_path: Path, run_name: str) -> bool:
    """Invoke run.py on cfg_path. Return True on success."""
    cmd = [sys.executable, str(RUN_SCRIPT), "--config", str(cfg_path)]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"  [ERR] {run_name} failed (returncode={r.returncode})")
        return False
    return True


def main():
    args = parse_args()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dir = SWEEP_DIR / "_configs"
    cfg_dir.mkdir(exist_ok=True)

    seeds_to_run = args.seeds if args.seeds is not None else SEEDS
    ms_to_run    = args.ms    if args.ms    is not None else M_VALUES
    invalid_s = set(seeds_to_run) - set(SEEDS)
    invalid_m = set(ms_to_run) - set(M_VALUES)
    if invalid_s:
        sys.exit(f"[ERR] unknown seed(s): {invalid_s}; valid: {SEEDS}")
    if invalid_m:
        sys.exit(f"[ERR] unknown M value(s): {invalid_m}; valid: {M_VALUES}")
    print(f"[init] seeds={seeds_to_run}  Ms={ms_to_run}")

    # 0. base config + source data
    print(f"[0] loading source: {DATA_FILE}")
    with open(DATA_FILE) as f:
        all_data = json.load(f)
    print(f"  total instances: {len(all_data)}")

    with open(BASE_CFG) as f:
        base_cfg = yaml.safe_load(f)

    # 1. per-seed subsets (always write all so eval can find them)
    print(f"\n[1] prepare {len(SEEDS)} seed subsets (N={N_INSTANCES} each)")
    seed_paths = prepare_seed_subsets(all_data)

    # 2. run pipelines
    if not args.eval_only:
        n_total = len(seeds_to_run) * len(ms_to_run)
        print(f"\n[2] run pipelines: {len(seeds_to_run)} seeds x {len(ms_to_run)} M = {n_total} runs")
        for s in seeds_to_run:
            subset_path = seed_paths[s]
            for M in ms_to_run:
                run_name = f"seed{s}_M{M:02d}"
                out_dir  = ROOT / base_cfg["output"]["base_dir"] / "M_sweep" / run_name
                ranked   = out_dir / "ranked_units.csv"
                print(f"\n  --- {run_name} ---  (out: {out_dir.relative_to(ROOT)})")
                if ranked.exists():
                    print(f"  [skip] ranked_units.csv exists")
                    continue

                cfg      = build_run_config(base_cfg, s, M, subset_path)
                cfg_path = cfg_dir / f"{run_name}.yaml"
                with open(cfg_path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

                ok = run_single(cfg_path, run_name)
                if not ok:
                    continue
                if not args.keep_heavy:
                    cleanup_heavy(out_dir)
                    print(f"  [cleanup] removed heavy tensors in {out_dir.name}")

    if args.skip_eval:
        print("\n[done] --skip_eval, exiting before metrics.")
        return

    # 3. evaluate each run
    print(f"\n[3] evaluate each run")
    sys.path.insert(0, str(ROOT))
    from eval_recall_at_k import IOI_GT, load_ranking, evaluate, DEFAULT_KS

    # Evaluate the full (seed, M) grid unless --seeds / --ms restrict it.
    eval_seeds = seeds_to_run if (args.seeds is not None) else SEEDS
    eval_ms    = ms_to_run    if (args.ms    is not None) else M_VALUES
    rows = []
    for s in eval_seeds:
        for M in eval_ms:
            run_name = f"seed{s}_M{M:02d}"
            ranked   = ROOT / base_cfg["output"]["base_dir"] / "M_sweep" / run_name / "ranked_units.csv"
            if not ranked.exists():
                print(f"  [miss] {ranked.relative_to(ROOT)}")
                continue
            res  = evaluate(load_ranking(ranked), IOI_GT, DEFAULT_KS)
            row  = {"seed": s, "M": M, "mAP": res["map"]}
            for k, m in res["at_k"].items():
                row[f"precision@{k}"] = m["precision"]
                row[f"recall@{k}"]    = m["recall"]
            rows.append(row)
            print(f"  {run_name}: mAP={res['map']:.3f}  "
                  f"P@10={res['at_k'][10]['precision']:.2f}  "
                  f"R@30={res['at_k'][30]['recall']:.2f}")

    if not rows:
        print("[ERR] no successful runs to evaluate.")
        return

    df = pd.DataFrame(rows)
    metrics_csv = SWEEP_DIR / "M_sweep_metrics.csv"
    df.to_csv(metrics_csv, index=False, float_format="%.4f")
    print(f"\n  [saved] {metrics_csv.relative_to(ROOT)}  ({len(df)} rows)")

    # 4. aggregate (mean and std per M)
    metric_cols = [c for c in df.columns if c not in ("seed", "M")]
    agg = df.groupby("M")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    summary_csv = SWEEP_DIR / "M_sweep_summary.csv"
    agg.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"  [saved] {summary_csv.relative_to(ROOT)}")

    # 5. plot metric vs M
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = agg["M"].values
    ks = sorted({int(c.split("@")[1].split("_")[0])
                 for c in agg.columns if "@" in c})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) mAP
    ax = axes[0]
    ax.errorbar(xs, agg["mAP_mean"], yerr=agg["mAP_std"],
                marker="o", linewidth=2, capsize=4, color="#2c3e50")
    ax.set_xlabel("M (paraphrases per instance)")
    ax.set_ylabel("mAP")
    ax.set_title(f"mAP vs M (mean ± std over {len(SEEDS)} seeds)")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # (b) precision@K
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(ks):
        col = f"precision@{k}"
        ax.errorbar(xs, agg[f"{col}_mean"], yerr=agg[f"{col}_std"],
                    marker="^", capsize=3, label=f"P@{k}",
                    color=cmap(i / max(len(ks) - 1, 1)))
    ax.set_xlabel("M")
    ax.set_ylabel("precision")
    ax.set_title("precision@K vs M")
    ax.set_xticks(xs)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # (c) recall@K
    ax = axes[2]
    for i, k in enumerate(ks):
        col = f"recall@{k}"
        ax.errorbar(xs, agg[f"{col}_mean"], yerr=agg[f"{col}_std"],
                    marker="s", capsize=3, label=f"R@{k}",
                    color=cmap(i / max(len(ks) - 1, 1)))
    ax.set_xlabel("M")
    ax.set_ylabel("recall")
    ax.set_title("recall@K vs M")
    ax.set_xticks(xs)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_png = SWEEP_DIR / "M_sweep_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_png.relative_to(ROOT)}")

    # 6. stdout summary
    print(f"\n{'='*70}")
    print(f"  M-sweep summary  (mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*70}")
    hdr = ["M", "mAP", "P@10", "P@20", "P@26", "R@10", "R@26", "R@30"]
    print("  " + "  ".join(f"{h:>12}" for h in hdr))
    for _, row in agg.iterrows():
        cells = [
            f"{int(row['M']):>12}",
            f"{row['mAP_mean']:.3f}±{row['mAP_std']:.3f}",
            f"{row['precision@10_mean']:.2f}±{row['precision@10_std']:.2f}",
            f"{row['precision@20_mean']:.2f}±{row['precision@20_std']:.2f}",
            f"{row['precision@26_mean']:.2f}±{row['precision@26_std']:.2f}",
            f"{row['recall@10_mean']:.2f}±{row['recall@10_std']:.2f}",
            f"{row['recall@26_mean']:.2f}±{row['recall@26_std']:.2f}",
            f"{row['recall@30_mean']:.2f}±{row['recall@30_std']:.2f}",
        ]
        print("  " + "  ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()
