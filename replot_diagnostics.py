"""
Replot diagnostics (Gram heatmap, SC histogram, SC CDF) in paper format.

Each subplot is saved separately as a 600 DPI PDF (and a PNG preview).
Reads cached tensors from a run directory:
    <run_dir>/gram_matrix.pt
    <run_dir>/sc.pt
    <run_dir>/diagnostics_summary.json

Usage:
    python replot_diagnostics.py [run_dir]

Defaults to results/gpt2_small_ioi_strong_N100_M5_linear/.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from utils.paper_style import paper_rc, paper_size  # noqa: E402


# Palette: cobalt (noise), saffron (semantic), raspberry (threshold)
HIST_COLOR     = "#2E5BAA"   # vibrant cobalt blue (noise / CDF main)
HIST_EDGE      = "#1A3F7A"
SEM_FILL       = "#E89B3C"   # rich saffron amber  (semantic)
SEM_EDGE       = "#A66A1B"
THRESH_COLOR   = "#C73E63"   # vibrant raspberry   (threshold)
NOISE_COLOR    = "#94A3B8"   # cool slate          (reference lines)
GRID_COLOR     = "#E8ECF1"
PAPER_BG       = "#FFFFFF"

# Sequential white-to-cobalt colormap for the Gram heatmap.
GRAM_CMAP = LinearSegmentedColormap.from_list(
    "atelier_blue_seq",
    [
        (0.00, "#FFFFFF"),   # white
        (0.25, "#E2EAF6"),   # pale ice
        (0.50, "#A4BFE6"),   # soft sky
        (0.75, "#5D7FCC"),   # mid cobalt
        (1.00, "#1A3F7A"),   # deep cobalt
    ],
    N=512,
)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fc = fig.get_facecolor()
    fig.savefig(pdf, dpi=600, bbox_inches="tight", facecolor=fc)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=fc)
    print(f"  [saved] {pdf}")
    print(f"  [saved] {png}")


# Gram heatmap
def plot_gram(gram: np.ndarray, N: int, M: int, nsr: float, out_dir: Path) -> None:
    rc = paper_rc("single")
    rc.update({
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.linewidth":    0.7,
    })
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.0))
        fig.patch.set_facecolor(PAPER_BG)

        # Clip to off-diagonal 1st/99th percentiles so the block structure
        # (intra ~0.57, inter ~0.24) is not washed out by the diagonal (~1)
        # and rare strong negatives.
        off_diag = gram[~np.eye(gram.shape[0], dtype=bool)]
        vmin = float(np.percentile(off_diag, 1))
        vmax = float(np.percentile(off_diag, 99))

        im = ax.imshow(
            gram, aspect="equal", cmap=GRAM_CMAP, vmin=vmin, vmax=vmax,
            interpolation="nearest", rasterized=True,
        )
        ax.set_facecolor(PAPER_BG)

        NM = N * M
        step = max(1, N // 20)
        for k in range(step, N, step):
            ax.axhline(k * M - 0.5, color=PAPER_BG, lw=0.3, alpha=0.65)
            ax.axvline(k * M - 0.5, color=PAPER_BG, lw=0.3, alpha=0.65)

        ax.set_xlabel("Gradient vector index $j$")
        ax.set_ylabel("Gradient vector index $i$")
        ax.set_title(
            f"Gradient cosine similarity  ($NM={NM}$)\n"
            f"NSR $= {nsr:.3f}$",
        )

        # ~6 evenly-spaced ticks aligned to instance boundaries (no overlap)
        n_ticks = 6
        instances_per_tick = max(1, int(round(N / (n_ticks - 1))))
        ticks = np.arange(0, NM + 1, instances_per_tick * M)
        if ticks[-1] != NM:
            ticks = np.append(ticks, NM)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(direction="out", length=2.5)

        # height-matched colorbar (same height as the square heatmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar_ticks = np.linspace(vmin, vmax, 5)
        cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in cbar_ticks])
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(length=2)

        fig.tight_layout()
        _save(fig, out_dir, "diag_gram_heatmap")
        plt.close(fig)


# SC histogram
def plot_sc_hist(
    sc: np.ndarray, threshold: float, n_sem: int, D: int, out_dir: Path,
) -> None:
    rc = paper_rc("single")
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.45))
        fig.patch.set_facecolor(PAPER_BG)
        ax.set_facecolor(PAPER_BG)

        bins = np.linspace(0.0, 1.0, 61)
        sem_mask = sc >= threshold
        _, _, p_noise = ax.hist(
            sc[~sem_mask], bins=bins, color=HIST_COLOR, alpha=1.0,
            edgecolor=HIST_EDGE, linewidth=0.3, label="Noise",
        )
        _, _, p_sem = ax.hist(
            sc[sem_mask], bins=bins, color=SEM_FILL, alpha=1.0,
            edgecolor=SEM_EDGE, linewidth=0.3, label="Semantic",
        )
        for patch in list(p_noise) + list(p_sem):
            patch.set_rasterized(True)

        ax.axvline(threshold, color=THRESH_COLOR, linestyle=(0, (5, 2)),
                   linewidth=1.3, alpha=0.85,
                   label=fr"threshold $\tau={threshold:.2f}$")

        sem_frac = n_sem / D
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"Sign Consistency  $\mathrm{SC}^{(d)}$")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Dimension-wise SC distribution\n"
            f"semantic $= {n_sem:,}$ ({sem_frac:.1%} of $D={D:,}$)",
        )

        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
            else (f"{x/1e3:.0f}k" if x >= 1e3 else f"{int(x)}")
        ))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.4)

        fig.tight_layout()
        _save(fig, out_dir, "diag_sc_histogram")
        plt.close(fig)


# SC CDF
def plot_sc_cdf(
    sc: np.ndarray, threshold: float, n_sem: int, D: int, out_dir: Path,
) -> None:
    rc = paper_rc("single")
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.45))
        fig.patch.set_facecolor(PAPER_BG)
        ax.set_facecolor(PAPER_BG)

        sorted_sc = np.sort(sc)
        cdf_full = np.arange(1, len(sorted_sc) + 1) / len(sorted_sc) * 100.0
        n_pts = 4000
        if len(sorted_sc) > n_pts:
            idx = np.linspace(0, len(sorted_sc) - 1, n_pts).astype(np.int64)
            xs, ys = sorted_sc[idx], cdf_full[idx]
        else:
            xs, ys = sorted_sc, cdf_full

        # cool "noise" fill under the curve (left of threshold)
        left = xs <= threshold
        fill_noise = ax.fill_between(
            xs[left], 0.0, ys[left],
            color=HIST_COLOR, alpha=0.22, linewidth=0,
        )
        fill_noise.set_rasterized(True)

        # warm "semantic" tail (right of threshold)
        right = xs >= threshold
        fill_sem = ax.fill_between(
            xs[right], ys[right], 100.0,
            color=SEM_FILL, alpha=0.55, linewidth=0,
            label=f"semantic ({n_sem/D:.1%})",
        )
        fill_sem.set_rasterized(True)

        (line,) = ax.plot(xs, ys, color=HIST_COLOR, linewidth=1.9, label="CDF")
        line.set_rasterized(True)

        # threshold + noise-fraction guides
        noise_frac = 1.0 - n_sem / D
        ax.axvline(threshold, color=THRESH_COLOR, linestyle=(0, (5, 2)),
                   linewidth=1.3, alpha=0.85,
                   label=fr"threshold $\tau={threshold:.2f}$")
        ax.axhline(noise_frac * 100, color=NOISE_COLOR, linestyle=(0, (1, 2.5)),
                   linewidth=1.1,
                   label=f"noise frac = {noise_frac:.1%}")

        # accent dot at the threshold/CDF intersection
        ax.plot([threshold], [noise_frac * 100], marker="o", markersize=4.5,
                markerfacecolor=SEM_FILL, markeredgecolor=THRESH_COLOR,
                markeredgewidth=0.9, zorder=5)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 100.0)
        ax.set_xlabel(r"Sign Consistency  $\mathrm{SC}^{(d)}$")
        ax.set_ylabel("CDF (%)")
        ax.set_title("SC cumulative distribution")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.4)

        fig.tight_layout()
        _save(fig, out_dir, "diag_sc_cdf")
        plt.close(fig)


# Main
def main() -> None:
    default_run = REPO_ROOT / "results" / "gpt2_small_ioi_strong_N100_M5_linear"
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_run
    run_dir = run_dir.resolve()
    print(f"[run_dir] {run_dir}")

    summary = json.loads((run_dir / "diagnostics_summary.json").read_text())
    threshold = float(summary["sc_threshold"])
    n_sem = int(summary["n_semantic_dims"])
    n_noise = int(summary["n_noise_dims"])
    D = n_sem + n_noise
    nsr = float(summary["nsr"])

    gram = torch.load(run_dir / "gram_matrix.pt", map_location="cpu").numpy()
    sc = torch.load(run_dir / "sc.pt", map_location="cpu").numpy()

    # diagnostics_summary.json does not store N and M; M is fixed to the
    # default run's value (N100_M5) and N is inferred from the Gram size.
    NM = gram.shape[0]
    M = 5
    N = NM // M
    print(f"[shape] NM={NM}  ->  N={N}, M={M}, D={D:,}")

    out_dir = run_dir / "plots" / "diagnostics_paper"
    plot_gram(gram, N, M, nsr, out_dir)
    plot_sc_hist(sc, threshold, n_sem, D, out_dir)
    plot_sc_cdf(sc, threshold, n_sem, D, out_dir)
    print(f"\n[done] PDFs/PNGs -> {out_dir}")


if __name__ == "__main__":
    main()
