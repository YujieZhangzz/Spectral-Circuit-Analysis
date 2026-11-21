"""Cross-task analysis: unit universality scores, task similarity matrix and
hierarchical clustering. Each task must have been run with run.py first
(needs <outdir>/weighted_unit_subspaces.npz).

Usage:
    python run_cross_task.py --configs configs/gpt2_small_ioi_strong.yaml \\
        configs/gpt2_small_cap.yaml configs/gpt2_small_gp.yaml [--names ioi cap gp]
"""

from __future__ import annotations
import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cross_task"))

from utils.io_utils    import load_config, get_output_dir            # noqa: E402
from utils.paper_style import paper_rc, paper_size                   # noqa: E402


# Helpers

def derive_task_name(cfg_path: Path) -> str:
    """Derive a display name from the config file name.
    'gpt2_small_ioi_strong.yaml' -> 'ioi_strong'
    'gpt2_small_cap.yaml'        -> 'cap'
    """
    stem = cfg_path.stem
    # Strip model prefixes such as 'gpt2_small_' / 'pythia_'.
    stem = re.sub(r"^(gpt2_small_|gpt2_|llama_|pythia_)", "", stem)
    # Strip '_N<int>_M<int>' suffixes.
    stem = re.sub(r"_N\d+(_M\d+)?$", "", stem)
    return stem


def banner(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# Main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True,
                   help="per-task config yamls (at least 2)")
    p.add_argument("--names", nargs="+", default=None,
                   help="display names (default: derived from config file names)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "cross_task"),
                   help="output directory")
    p.add_argument("--similarity-mode", default="importance_weighted",
                   choices=["importance_weighted", "significant_mean"])
    p.add_argument("--baseline-n",       type=int,   default=200)
    p.add_argument("--baseline-q",       type=float, default=0.99)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if len(args.configs) < 2:
        sys.exit("[ERROR] cross-task analysis needs at least 2 tasks")

    # Resolve config -> outdir -> npz path for each task.
    config_paths = [Path(c) for c in args.configs]
    if args.names:
        if len(args.names) != len(config_paths):
            sys.exit("[ERROR] --names must have the same length as --configs")
        names = args.names
    else:
        names = [derive_task_name(c) for c in config_paths]

    # Resolve primary + sigma sibling npz per task; auto-derive sigma if missing.
    from cross_task.weighted_unit_svd import (
        load_task_subspaces,
        save_task_subspaces,
        derive_sigma_subs,
    )

    npz_primary_paths: list = []
    npz_sigma_paths:   list = []
    primary_modes:     list = []
    for cfg_path, name in zip(config_paths, names):
        cfg          = load_config(str(cfg_path))
        outdir       = get_output_dir(cfg)
        npz_primary  = outdir / "weighted_unit_subspaces.npz"
        npz_sigma    = outdir / "weighted_unit_subspaces_sigma.npz"
        if not npz_primary.exists():
            sys.exit(f"[ERROR] missing {npz_primary}\n"
                     f"        run `python run.py --config {cfg_path}` first")
        if not npz_sigma.exists():
            print(f"[derive] {npz_sigma.name} missing; deriving sigma_weighted "
                  f"from primary npz ({name})")
            subs = load_task_subspaces(str(npz_primary))
            save_task_subspaces(derive_sigma_subs(subs), str(npz_sigma))
        mode = cfg.get("weighted_unit_svd", {}).get("importance_mode", "sigma_weighted")
        primary_modes.append(mode)
        npz_primary_paths.append(npz_primary)
        npz_sigma_paths.append(npz_sigma)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_label = primary_modes[0]
    if len(set(primary_modes)) > 1:
        print(f"[warn] mixed importance_mode across configs: {primary_modes}; "
              f"using '{primary_label}' as primary subdir name")

    # variants to run: primary first, then sigma sibling (skip if primary IS sigma)
    variants = [(primary_label, npz_primary_paths)]
    if primary_label != "sigma_weighted":
        variants.append(("sigma_weighted", npz_sigma_paths))

    banner("Cross-task analysis")
    print(f"  similarity_mode : {args.similarity_mode}")
    print(f"  baseline (n,q)  : ({args.baseline_n}, {args.baseline_q})")
    print(f"  out-dir         : {out_dir}")
    print(f"  variants        : {[v[0] for v in variants]}")
    for label, paths in variants:
        print(f"  inputs ({label}):")
        for n, p in zip(names, paths):
            print(f"    {n:20s} <- {p}")

    if args.dry_run:
        return

    from cross_task.cross_task_analysis import run_analysis

    for label, paths in variants:
        sub_out = out_dir / label
        sub_out.mkdir(parents=True, exist_ok=True)
        _run_one_variant(label, sub_out, names, paths, args,
                         load_task_subspaces, run_analysis)

    # Cross-mode comparison dendrogram (only if both variants ran)
    if len(variants) == 2:
        _plot_compare_dendrogram(
            out_dir / variants[0][0] / "cross_task_result.pkl",
            out_dir / variants[1][0] / "cross_task_result.pkl",
            out_dir / "dendrogram_compare",
            labels = (variants[0][0], variants[1][0]),
        )

    banner("DONE")


def _run_one_variant(label, sub_out, names, npz_paths, args,
                     load_task_subspaces, run_analysis):
    """Run cross-task analysis for one importance variant; save artifacts."""
    print(f"\n{'-'*60}\n  variant: {label}\n  out: {sub_out}\n{'-'*60}")
    task_subspaces = {
        name: load_task_subspaces(str(npz))
        for name, npz in zip(names, npz_paths)
    }
    result = run_analysis(
        task_subspaces,
        baseline_n_samples = args.baseline_n,
        baseline_quantile  = args.baseline_q,
        similarity_mode    = args.similarity_mode,
        verbose            = True,
    )

    pkl_path = sub_out / "cross_task_result.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n[saved] {pkl_path}")

    sim    = result["task_similarity"]
    tnames = result["task_names"]
    with open(sub_out / "task_similarity.csv", "w") as f:
        w = csv.writer(f)
        w.writerow([""] + list(tnames))
        for i, n in enumerate(tnames):
            w.writerow([n] + [f"{sim[i,j]:.4f}" for j in range(len(tnames))])
    print(f"[saved] {sub_out / 'task_similarity.csv'}")

    with open(sub_out / "universal_units.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["rank", "unit_id", "type", "layer",
                    "universality_significant_mean"])
        for rk, u in enumerate(result["ranked_units"], start=1):
            w.writerow([rk, u.get("unit_id"), u.get("type"),
                        u.get("layer"),
                        f"{u.get('significant_mean', float('nan')):.4f}"])
    print(f"[saved] {sub_out / 'universal_units.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        with plt.rc_context(paper_rc("single")):
            _, ax = plt.subplots(figsize=paper_size("single", aspect=1.45))
            dendrogram(result["task_linkage"], labels=tnames, ax=ax,
                       color_threshold=0)
            ax.set_title(label)
            ax.set_ylabel("Distance (1 − similarity)")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            base = sub_out / "dendrogram"
            plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
            plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[saved] {base.with_suffix('.pdf')}")
    except Exception as e:
        print(f"[warn] dendrogram skipped: {e}")


def _plot_compare_dendrogram(pkl_a, pkl_b, out_base, labels):
    """Side-by-side dendrograms for two variants."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        ra = pickle.loads(Path(pkl_a).read_bytes())
        rb = pickle.loads(Path(pkl_b).read_bytes())
        names = ra["task_names"]
        assert names == rb["task_names"], "task_names disagree across variants"

        with plt.rc_context(paper_rc("double")):
            _, axes = plt.subplots(
                1, 2,
                figsize=paper_size("double", aspect=0.55),
                sharey=True,
            )
            for ax, r, title in zip(axes, (ra, rb), labels):
                dendrogram(r["task_linkage"], labels=names, ax=ax,
                           color_threshold=0)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=20)
            axes[0].set_ylabel("Distance (1 − similarity)")
            plt.tight_layout()
            plt.savefig(Path(out_base).with_suffix(".pdf"), bbox_inches="tight")
            plt.savefig(Path(out_base).with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[saved] {Path(out_base).with_suffix('.pdf')}")
    except Exception as e:
        print(f"[warn] compare dendrogram skipped: {e}")


if __name__ == "__main__":
    main()
