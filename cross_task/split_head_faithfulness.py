"""Per-task vector PDFs for head-faithfulness, plus a separate shared legend.

For each task in `combine_head_faithfulness.PRESETS`, reads
results/<experiment_name>/faithfulness_head/head_faithfulness.json and writes
one vector PDF titled with the task's display name and no legend, so panels
can be arranged freely. A separate PDF holds the shared legend, in the same
order as the combined grid figure.

Usage:
    python cross_task/split_head_faithfulness.py [--preset gpt2]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils.paper_style import paper_rc, paper_size  # noqa: E402

from cross_task.combine_head_faithfulness import (  # noqa: E402
    PRESETS, METHOD_COLORS, LEGEND_ORDER, LEGEND_LW, RESULTS_DIR, OUT_DIR,
)


def _save_vec(fig, base: Path) -> None:
    """Vector PDF + 600-dpi PNG preview."""
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {pdf}")
    print(f"[saved] {png}")


def plot_one_task(json_path: Path, title: str, out_base: Path) -> None:
    data = json.loads(json_path.read_text())
    ks       = np.array(data["ks"])
    ld_clean = data["ld_clean"]
    ld_corr  = data["ld_corrupted"]
    span     = ld_clean - ld_corr
    rand_std = np.array(data["random_std"])
    curves   = {k: np.array(v) for k, v in data["curves"].items()}
    recovery = {k: (v - ld_corr) / span for k, v in curves.items()}

    # Map JSON method names to legend display names.
    rename = {"SVD (ours)": "SCA (ours)"}

    with plt.rc_context(paper_rc("single")):
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.45))
        for label_json, y in recovery.items():
            label = rename.get(label_json, label_json)
            if label not in METHOD_COLORS:
                continue
            c  = METHOD_COLORS[label]
            ls = "--" if label == "Random" else "-"
            lw = 0.9  if label == "Random" else 1.0
            ax.plot(ks, y, color=c, ls=ls, lw=lw)
            if label == "Random":
                ax.fill_between(ks, y - rand_std/span, y + rand_std/span,
                                color=c, alpha=0.15)
        ax.axhline(0.0, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.axhline(1.0, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("K (top-K attention units affected)")
        ax.set_ylabel("recovery score")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        fig.tight_layout()
        _save_vec(fig, out_base)


def plot_legend(out_base: Path) -> None:
    """Single-row legend matching the combined grid (7 entries)."""
    handles = []
    for label in LEGEND_ORDER:
        c  = METHOD_COLORS[label]
        ls = "--" if label == "Random" else "-"
        handles.append(Line2D([0], [0], color=c, ls=ls, lw=LEGEND_LW, label=label))

    with plt.rc_context(paper_rc("double")):
        # Height for one row of legend text
        fig = plt.figure(figsize=(7.0, 0.45))
        fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                   loc="center", ncol=len(LEGEND_ORDER),
                   frameon=False, columnspacing=1.4,
                   handlelength=2.6, handletextpad=0.5,
                   fontsize=8.5)
        _save_vec(fig, out_base)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="gpt2", choices=sorted(PRESETS.keys()))
    ap.add_argument("--out_subdir", default="per_task_split",
                    help=f"subdir under {OUT_DIR} (default: per_task_split)")
    args = ap.parse_args()

    tasks = PRESETS[args.preset]
    out_dir = OUT_DIR / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for folder, title in tasks:
        json_path = RESULTS_DIR / folder / "faithfulness_head" / "head_faithfulness.json"
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        out_base = out_dir / folder
        plot_one_task(json_path, title, out_base)

    plot_legend(out_dir / "legend")
    print(f"\n[done] {len(tasks)} panels + 1 legend in: {out_dir}")


if __name__ == "__main__":
    main()
