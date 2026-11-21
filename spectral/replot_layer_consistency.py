"""
Redraw the attn vs MLP layer-wise consistency plot from
gradient_heatmap_data.npz (no config or gradient recomputation needed).
Plotting code lives in spectral/gradient_heatmap.py.

Usage:
    python spectral/replot_layer_consistency.py \\
        --data-dir results/<experiment_name> [--preset double]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral.gradient_heatmap import (
    DATA_FILENAME, PAPER_PRESETS, DEFAULT_SPLIT_PRESET,
    load_heatmap_data, plot_layer_consistency_split,
)


def replot(data_dir: Path, out_dir: Path, preset: str) -> None:
    """data_dir holds the .npz; PDF/PNG go to out_dir (default data_dir/plots/)."""
    (_, _, _, layers, _, _, cosim_split) = load_heatmap_data(str(data_dir))
    if cosim_split is None:
        raise RuntimeError(
            f"{data_dir / DATA_FILENAME} has no split data; "
            "rerun gradient_heatmap with split_consistency=True.")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_layer_consistency_split(cosim_split, layers, str(out_dir),
                                 preset=preset)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help=f"Directory containing {DATA_FILENAME} (usually <outdir>/)")
    p.add_argument("--preset", default=DEFAULT_SPLIT_PRESET,
                   choices=list(PAPER_PRESETS.keys()),
                   help="Figure size preset: single | double")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for PDF/PNG (default <data-dir>/plots/)")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else data_dir / "plots"
    replot(data_dir, out_dir, preset=args.preset)


if __name__ == "__main__":
    main()
