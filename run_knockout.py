"""Head-level and z-subspace knockout sweeps for one task, plus comparison plots.

Steps: (a) cumulative top-K head patching for each ranking method,
(b) cumulative top-K z-subspace patching, (c) plots. Requires run.py output
(ranked_units.csv); run_baselines.py and EAP-IG rankings (knockout.eap_ig_dir)
are optional and skipped if missing.

Usage:
    python run_knockout.py --config configs/gpt2_small_ioi_strong.yaml --device cuda:0
    python run_knockout.py --config configs/gpt2_small_ioi_strong.yaml --skip head zsub  # replot only
"""

from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils.io_utils import load_config, get_output_dir  # noqa: E402


# Helpers

def banner(title: str) -> None:
    print(f"\n{'-'*70}\n  {title}\n{'-'*70}")


def run_cmd(cmd, dry: bool = False) -> None:
    print("$", " ".join(shlex.quote(str(c)) for c in cmd))
    if dry:
        return
    proc = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def need(p: Path, hint: str = "") -> None:
    if not p.exists():
        print(f"\n[ERROR] missing: {p}")
        if hint:
            print(f"        hint: {hint}")
        sys.exit(2)


# Steps

def step_head_sweep(outdir: Path, csv: Path, methods: dict,
                    n: int, k_max: int, k_step: int,
                    device: str, model_name: str,
                    gt_mode: str, verb_list_csv,
                    sva_metric_variant: str,
                    start: int,
                    dry: bool) -> Path:
    banner("(a) head-level knockout sweep")
    need(csv, "prompts file given by knockout.data_csv")
    head_dir = outdir / "faithfulness_head"

    cmd = [
        sys.executable, "-m", "circuits_discovery.faithfulness.sweep_head",
        "--csv",     csv,
        "--n",       n,
        "--start",   start,
        "--model",   model_name,
        "--device",  device,
        "--k_max",   k_max,
        "--k_step",  k_step,
        "--out_dir", head_dir,
        "--gt_mode", gt_mode,
        "--sva_metric_variant", sva_metric_variant,
    ]
    if verb_list_csv:
        cmd += ["--verb_list_csv", str(verb_list_csv)]
    # Pass every method csv; sweep_head skips files that do not exist.
    flag_for = {
        "svd":       "--svd_csv",
        "magnitude": "--magnitude_csv",
        "direction": "--direction_csv",
        "eap":       "--eap_csv",
        "eap_ig":    "--eap_ig_csv",
        "eap_ig_kl": "--eap_ig_kl_csv",
    }
    for key, flag in flag_for.items():
        path = methods.get(key)
        if path is not None:
            cmd += [flag, str(path)]

    run_cmd(cmd, dry=dry)
    return head_dir


def ensure_z_subspaces(outdir: Path, csv: Path, n: int, start: int,
                       model_name: str,
                       device: str, dry: bool) -> Path:
    """Build outdir/z_subspaces.pt via subspace_z.py if it's not already there."""
    z_subs = outdir / "z_subspaces.pt"
    if z_subs.exists():
        return z_subs
    banner(f"(b-pre) build z_subspaces.pt  ({z_subs.name} missing)")
    run_cmd([
        sys.executable, "-m", "circuits_discovery.faithfulness.subspace_z",
        "--csv",    csv,
        "--n",      n,
        "--start",  start,
        "--model",  model_name,
        "--device", device,
        "--out",    z_subs,
    ], dry=dry)
    return z_subs


def step_zsub_sweep(outdir: Path, csv: Path, n: int, k_max: int,
                    k_step: int, device: str, model_name: str,
                    gt_mode: str, verb_list_csv,
                    sva_metric_variant: str, start: int, dry: bool) -> Path:
    banner("(b) subspace-z sweep")
    ranking_csv = outdir / "ranked_units.csv"
    need(ranking_csv, "produced by run.py")
    z_subs = ensure_z_subspaces(outdir, csv, n, start, model_name, device, dry)

    sub_dir = outdir / "faithfulness_subspace_z"
    cmd = [
        sys.executable, "-m", "circuits_discovery.faithfulness.sweep_subspace_z",
        "--csv",         csv,
        "--n",           n,
        "--start",       start,
        "--model",       model_name,
        "--device",      device,
        "--ranking_csv", ranking_csv,
        "--z_subspaces", z_subs,
        "--k_max",       k_max,
        "--k_step",      k_step,
        "--out_dir",     sub_dir,
        "--tag",         "svd",
        "--gt_mode",     gt_mode,
        "--sva_metric_variant", sva_metric_variant,
    ]
    if verb_list_csv:
        cmd += ["--verb_list_csv", str(verb_list_csv)]
    run_cmd(cmd, dry=dry)
    return sub_dir


def step_plots(head_dir: Path, sub_dir: Path, k_zoom: int,
               dry: bool) -> None:
    banner(f"(c) paper-quality plots (head & head-vs-zsub@K={k_zoom})")
    need(head_dir / "head_faithfulness.json", "output of step (a)")

    run_cmd([
        sys.executable, "-m", "circuits_discovery.faithfulness.plot_head_final",
        "--in_dir",  head_dir,
        "--sub_dir", sub_dir,
    ], dry=dry)

    run_cmd([
        sys.executable, "-m", "circuits_discovery.faithfulness.plot_head_vs_zsub",
        "--head_dir", head_dir,
        "--zsub_dir", sub_dir,
        "--k_max",    k_zoom,
    ], dry=dry)


# Main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True,
                   help="task yaml under configs/")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n",      type=int, default=100,
                   help="number of IOI prompts")
    p.add_argument("--k_max",  type=int, default=144,
                   help="max K for head sweep (full grid)")
    p.add_argument("--k_step", type=int, default=1)
    p.add_argument("--k_zoom", type=int, default=50,
                   help="K cutoff for head_vs_zsub_K{X}.png")
    p.add_argument("--csv", type=Path, default=None,
                   help="prompts csv (default: knockout.data_csv from the yaml)")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["head", "zsub", "plots"],
                   help="steps to skip")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg    = load_config(args.config)
    outdir = get_output_dir(cfg)
    ko_cfg = cfg.get("knockout", {}) or {}
    data_cfg   = cfg.get("data", {}) or {}
    model_name = cfg["model"]["name_or_path"]
    gt_mode    = ko_cfg.get("gt_mode", "prob_diff")
    verb_list_csv = ko_cfg.get("verb_list_csv")   # required only for task='sva' verb_set
    # SVA metric variant: explicit knockout.sva_metric_variant wins; otherwise
    # mirror data.loss_variant ONLY for SVA tasks (loss_variant for GT etc. is
    # not a valid SVA metric and would fail argparse downstream).
    _data_format = data_cfg.get("format", "")
    sva_metric_variant = ko_cfg.get("sva_metric_variant")
    if sva_metric_variant is None:
        sva_metric_variant = (data_cfg.get("loss_variant", "verb_set")
                              if _data_format == "sva" else "verb_set")
    # Row offset into knockout.data_csv. Defaults to data.start so the gradient
    # and knockout phases use the same slice; knockout.start overrides it.
    start = int(ko_cfg.get("start", data_cfg.get("start", 0)))

    # Prompts csv: --csv, else knockout.data_csv.
    csv_path = args.csv or (
        Path(ko_cfg["data_csv"]) if ko_cfg.get("data_csv") else None
    )
    if csv_path is None:
        sys.exit("[ERROR] no prompts csv: set knockout.data_csv in the yaml "
                 "or pass --csv")

    # Ranking csvs. SCA and raw-gradient baselines live under outdir.
    methods = {
        "svd":       outdir / "ranked_units.csv",
        "magnitude": outdir / "variance_baseline" / "ranked_units_magnitude.csv",
        "direction": outdir / "variance_baseline" / "ranked_units_direction.csv",
    }
    # EAP family rankings come from knockout.eap_ig_dir + eap_ig_tag.
    eap_dir = Path(ko_cfg["eap_ig_dir"]) if ko_cfg.get("eap_ig_dir") else None
    eap_tag = ko_cfg.get("eap_ig_tag")
    if eap_dir and eap_tag:
        methods["eap"]       = eap_dir / f"{eap_tag}_vanilla_head_ranking.csv"
        methods["eap_ig"]    = eap_dir / f"{eap_tag}_task_head_ranking.csv"
        methods["eap_ig_kl"] = eap_dir / f"{eap_tag}_kl_head_ranking.csv"

    print(f"\n{'='*60}")
    print(f"  Knockout pipeline")
    print(f"  config : {args.config}")
    print(f"  outdir : {outdir}")
    print(f"  csv    : {csv_path}")
    print(f"  gt_mode: {gt_mode}")
    if verb_list_csv:
        print(f"  verb_list_csv: {verb_list_csv}")
    print(f"  sva_metric_variant: {sva_metric_variant}")
    print(f"  data slice: rows [{start}, {start + args.n})  (start={start}, n={args.n})")
    print(f"  skip   : {args.skip or '(none)'}")
    print(f"  methods (will skip missing files at sweep time):")
    for k, v in methods.items():
        mark = "x" if Path(v).exists() else " "
        print(f"    [{mark}] {k:10s} {v}")
    print(f"{'='*60}")

    head_dir = outdir / "faithfulness_head"
    sub_dir  = outdir / "faithfulness_subspace_z"

    if "head" not in args.skip:
        head_dir = step_head_sweep(outdir, csv_path, methods,
                                   args.n, args.k_max, args.k_step,
                                   args.device, model_name,
                                   gt_mode, verb_list_csv,
                                   sva_metric_variant, start, args.dry_run)
    if "zsub" not in args.skip:
        sub_dir = step_zsub_sweep(outdir, csv_path, args.n, args.k_max,
                                  args.k_step, args.device, model_name,
                                  gt_mode, verb_list_csv,
                                  sva_metric_variant, start, args.dry_run)
    if "plots" not in args.skip:
        step_plots(head_dir, sub_dir, args.k_zoom, args.dry_run)

    banner("DONE")


if __name__ == "__main__":
    main()
