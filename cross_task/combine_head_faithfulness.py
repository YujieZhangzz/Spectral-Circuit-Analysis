"""Build a multi-task grid figure from per-task `head_faithfulness_linear.pdf` files.

Each results/<folder>/faithfulness_head/head_faithfulness_linear.pdf is
rasterized, its legend strip, white border and title row are cropped, and the
panels are arranged in a grid with task titles, (a)..(l) labels and one shared
legend. Output goes to results/cross_task_diagnostics/.

Usage:
  python cross_task/combine_head_faithfulness.py
  python cross_task/combine_head_faithfulness.py --preset pythia --rows 3 --cols 4
"""
from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

# Paths
RESULTS_DIR = Path("results")
OUT_DIR     = RESULTS_DIR / "cross_task_diagnostics"
SOURCE_PDF  = "head_faithfulness_linear.pdf"

# Task presets: lists of (folder_name, display_title).
PRESETS: dict[str, list[tuple[str, str]]] = {
    "gpt2": [
        ("gpt2_small_col_N100_M5",                "Colored Objects"),
        ("gpt2_small_country_cap_N100_M5",        "Country-Capital"),
        ("gpt2_small_gb_N100_M5",                 "Gender Bias"),
        ("gpt2_small_gp_N100_M5",                 "Gendered Pronoun"),
        ("gpt2_small_gt_boundary_N100_M5_linear", "Greater-Than"),
        ("gpt2_small_gt_price_N100_M5_linear",    "Greater-Than-Price"),
        ("gpt2_small_ioi_strong_N100_M5_linear",  "IOI"),
        ("gpt2_small_per_country_N100_M5",        "Person-Country"),
        ("gpt2_small_per_pro_N100_M5",            "Person-Profession"),
        ("gpt2_small_sva_blimp_N100_M5",          "SVA-BLiMP"),
        ("gpt2_small_sva_N100_M5_isare_back100",  "SVA-Binary"),
        ("gpt2_small_sva_N100_M5_ori",            "SVA"),
    ],
    "pythia": [
        ("pythia_col_obj_N100_M5",            "Colored Objects"),
        ("pythia_country_cap_N100_M5",        "Country-Capital"),
        ("pythia_gender_bias_N100_M5",        "Gender Bias"),
        ("pythia_gp_N100_M5",                 "Gendered Pronoun"),
        ("pythia_gt_N100_M5",                 "Greater-Than"),
        ("pythia_gt_boundary_N100_M5",        "Greater-Than"),
        ("pythia_gt_price_N100_M5",           "Greater-Than-Price"),
        ("pythia_gt_price_boundary_N100_M5",  "Greater-Than-Price"),
        ("pythia_ioi_strong_N100_M5",         "IOI"),
        ("pythia_person_country_N100_M5",     "Person-Country"),
        ("pythia_person_pro_N100_M5",         "Person-Profession"),
        ("pythia_sva_N100_M5",                "SVA"),
        ("pythia_sva_blimp_N100_M5",          "SVA-BLiMP"),
        ("pythia_sva_isare_N100_M5",          "SVA-Binary"),
    ],
}

# Styling
METHOD_COLORS = {
    "SCA (ours)":       "#1f77b4",
    "Random":           "gray",
    "Magnitude (raw)":  "#ff7f0e",
    "Direction (raw)":  "#2ca02c",
    "EAP":              "#d62728",
    "EAP-IG":           "#9467bd",
    "EAP-IG-KL":        "#8c564b",
}
LEGEND_ORDER = [
    "SCA (ours)", "Random", "Magnitude (raw)", "Direction (raw)",
    "EAP", "EAP-IG", "EAP-IG-KL",
]
LEGEND_LW = 3.2

# Cut the PDF below this y (at 200 dpi) to drop the per-PDF legend strip;
# the linear PDF has a ~60 px white gap at y=369..428 between plot and legend.
LEGEND_CUT_Y_AT_200DPI = 395


# Image helpers
def _strip_top_title(im: Image.Image) -> Image.Image:
    """Drop the topmost text band ("recovery vs K (linear)") if a
    title / white gap / plot pattern is detected. Scans only the central horizontal strip
    so the y-axis label on the left does not defeat the row-wise white check.
    """
    g = np.array(im.convert("L"))
    H, W = g.shape
    strip = g[:, int(W * 0.30):int(W * 0.85)]
    is_white = (strip > 250).all(axis=1)
    n = len(is_white)
    i = 0
    while i < n and is_white[i]: i += 1
    title_top = i
    if title_top >= n:
        return im
    while i < n and not is_white[i]: i += 1
    title_end = i
    while i < n and is_white[i]: i += 1
    plot_top = i
    if plot_top >= n:
        return im
    if (plot_top - title_top) > 0.25 * H:
        return im
    return im.crop((0, plot_top, im.width, im.height))


def rasterize_and_crop(pdf_path: Path, dpi: int = 600) -> np.ndarray:
    """Rasterize PDF, drop legend, tight-crop white, drop title row, re-crop."""
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
             str(pdf_path), str(prefix)],
            check=True,
        )
        im = Image.open(str(prefix) + ".png").convert("RGB")
        im.load()
    cut = int(LEGEND_CUT_Y_AT_200DPI * dpi / 200)
    if cut < im.height:
        im = im.crop((0, 0, im.width, cut))
    gray = np.array(im.convert("L"))
    nw = gray < 250
    if nw.any():
        ys, xs = np.where(nw)
        pad = max(1, dpi // 150)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(im.width,  int(xs.max()) + 1 + pad)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(im.height, int(ys.max()) + 1 + pad)
        im = im.crop((x0, y0, x1, y1))
    im = _strip_top_title(im)
    gray = np.array(im.convert("L"))
    nw = gray < 250
    if nw.any():
        ys, _ = np.where(nw)
        y0 = max(0, int(ys.min()) - 2)
        im = im.crop((0, y0, im.width, im.height))
    return np.asarray(im)


def make_legend_handles():
    handles = []
    for label in LEGEND_ORDER:
        c = METHOD_COLORS[label]
        ls = "--" if label == "Random" else "-"
        handles.append(Line2D([0], [0], color=c, ls=ls, lw=LEGEND_LW, label=label))
    return handles


# Grid sizing
def auto_grid(n: int) -> tuple[int, int]:
    """Pick a sensible (rows, cols) for n panels, preferring 3 cols where possible."""
    if n <= 3:           return 1, n
    if n <= 6:           return 2, 3
    if n <= 9:           return 3, 3
    if n <= 12:          return 4, 3
    if n <= 16:          return 4, 4
    rows = int(np.ceil(n / 4))
    return rows, 4


# Main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="gpt2", choices=sorted(PRESETS.keys()),
                    help="which task list to use (default: gpt2)")
    ap.add_argument("--rows", type=int, default=None,
                    help="number of grid rows (default: auto from #tasks)")
    ap.add_argument("--cols", type=int, default=None,
                    help="number of grid cols (default: auto from #tasks)")
    ap.add_argument("--fig_w", type=float, default=7.0,
                    help="figure width in inches (default: 7.0, double-column width)")
    ap.add_argument("--dpi_in", type=int, default=600,
                    help="rasterization DPI for source PDFs (default: 600)")
    ap.add_argument("--dpi_out", type=int, default=300,
                    help="DPI for the output PNG (default: 300)")
    ap.add_argument("--out_suffix", default=None,
                    help="extra suffix appended to output filename")
    args = ap.parse_args()

    tasks = PRESETS[args.preset]
    if not tasks:
        raise SystemExit(f"preset '{args.preset}' is empty; fill PRESETS in this script first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_base = OUT_DIR / f"head_faithfulness_grid_{args.preset}_{len(tasks)}tasks_linear{suffix}"

    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["STIX Two Text", "STIXGeneral",
                             "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
    })

    panels = []
    for folder, title in tasks:
        pdf = RESULTS_DIR / folder / "faithfulness_head" / SOURCE_PDF
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        img = rasterize_and_crop(pdf, dpi=args.dpi_in)
        panels.append((title, img))
        print(f"[loaded] {folder}  -> {img.shape}  '{title}'")

    n_rows, n_cols = (args.rows, args.cols) if (args.rows and args.cols) \
                                            else auto_grid(len(tasks))
    panel_aspect = panels[0][1].shape[1] / panels[0][1].shape[0]
    panel_w = args.fig_w / n_cols
    panel_h = panel_w / panel_aspect
    title_pad_in  = 0.16
    label_pad_in  = 0.20
    legend_pad_in = 0.55
    row_h = panel_h + title_pad_in + label_pad_in
    fig_h = row_h * n_rows + legend_pad_in

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.fig_w, fig_h),
                             squeeze=False)
    axes_flat = axes.flatten()

    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, (title, img) in enumerate(panels):
        ax = axes_flat[i]
        ax.imshow(img, aspect="auto", interpolation="none", resample=False)
        ax.set_axis_off()
        ax.set_title(title, fontsize=7.5, family="serif", pad=2)
        ax.text(0.5, -0.06, f"({letters[i]})",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=8.5, family="serif")

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        ncol=len(LEGEND_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
        fontsize=8.5,
        columnspacing=1.4,
        handlelength=2.6,
        handletextpad=0.5,
    )

    bottom_frac = (legend_pad_in + 0.05) / fig_h
    fig.subplots_adjust(left=0.01, right=0.99,
                        top=0.985, bottom=bottom_frac,
                        wspace=0.04, hspace=0.42)

    pdf_out = out_base.with_suffix(".pdf")
    png_out = out_base.with_suffix(".png")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=args.dpi_out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {pdf_out}")
    print(f"[saved] {png_out}")


if __name__ == "__main__":
    main()
