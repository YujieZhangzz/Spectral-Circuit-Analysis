"""Run the raw-gradient baselines (magnitude and direction) for one task.

Thin wrapper around spectral/variance_baseline.py that resolves the output
directory from the task config. Run run.py with the same config first.
Writes <outdir>/variance_baseline/{unit_metrics,ranked_units_magnitude,
ranked_units_direction}.csv.

Usage:
    python run_baselines.py --config configs/gpt2_small_ioi_strong.yaml --device cuda:0
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True,
                   help="task yaml under configs/ (e.g. gpt2_small_ioi_strong.yaml)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scope",  default="attn_VO_mlp",
                   help="param scope (passed to variance_baseline)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg        = load_config(args.config)
    outdir     = get_output_dir(cfg)
    model_name = cfg["model"]["name_or_path"]
    dtype      = cfg["model"].get("torch_dtype", "bfloat16")
    print(f"\n{'='*60}")
    print(f"  Baselines (magnitude + direction)")
    print(f"  config : {args.config}")
    print(f"  model  : {model_name}")
    print(f"  outdir : {outdir}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "spectral.variance_baseline",
        "--results_dir", str(outdir),
        "--model",       model_name,
        "--dtype",       dtype,
        "--device",      args.device,
        "--scope",       args.scope,
    ]
    print("$", " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
