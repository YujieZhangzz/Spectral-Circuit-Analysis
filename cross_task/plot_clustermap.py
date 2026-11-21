"""Cross-task similarity clustermap (heatmap + dendrogram + family band).

Reads task_similarity, task_linkage and task_names from
cross_task/<CT_RUN>/<variant>/cross_task_result.pkl and writes
<variant>/clustermap.{pdf,png}.

Usage:
    python cross_task/plot_clustermap.py                    # all variants
    python cross_task/plot_clustermap.py consistency_share
"""
from __future__ import annotations
import os, sys, pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    HPacker, TextArea, DrawingArea, AnchoredOffsetbox,
)
from scipy.cluster.hierarchy import dendrogram, leaves_list

from utils.paper_style import paper_rc

ROOT = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")

PRETTY = {
    "col_obj":        "Colored Objects",
    "country_cap":    "Country-Capital",
    "gender_bias":    "Gender Bias",
    "gp":             "Gender Pronoun",
    "gt":             "Greater-Than",
    "gt_price":       "Greater-Than$",
    "ioi_strong":     "IOI",
    "person_country": "Person-Country",
    "person_pro":     "Person-Profession",
    "sva":            "SVA",
    "sva_blimp":      "SVA-BLiMP",
    "sva_isare":      "SVA-Binary",
}

# Family names emphasise the shared computation in each cluster.
FAMILY = {
    "gt":             "Numeric Comparison",
    "gt_price":       "Numeric Comparison",
    "sva":            "Subject-Verb Agreement",
    "sva_blimp":      "Subject-Verb Agreement",
    "sva_isare":      "Subject-Verb Agreement",
    "country_cap":    "Factual Recall",
    "person_country": "Factual Recall",
    "gender_bias":    "Gendered Coreference",
    "gp":             "Gendered Coreference",
    "person_pro":     "Gendered Coreference",
    "col_obj":        "Singletons",
    "ioi_strong":     "Singletons",
}

FAM_COLOR = {
    "Numeric Comparison":     "#3D6FA3",  # light slate blue
    "Subject-Verb Agreement": "#6B9E4F",  # forest green
    "Factual Recall":         "#4FA68D",  # sage teal
    "Gendered Coreference":   "#A678B5",  # soft purple
    "Singletons":             "#B0B0B0",  # neutral gray (not a family)
}
FAM_ORDER = [
    "Numeric Comparison",
    "Subject-Verb Agreement",
    "Factual Recall",
    "Gendered Coreference",
    "Singletons",
]
NEUTRAL_LINK = "#BBBBBB"


def reorder_linkage(Z, names):
    """Swap children of one internal node so the leaf order reads better:
    put `sva` before `sva_blimp` in the SVA subtree."""
    Z = Z.copy()
    sva_blimp_idx = names.index("sva_blimp")
    for k in range(len(Z)):
        a, b = int(Z[k, 0]), int(Z[k, 1])
        if a == sva_blimp_idx:
            Z[k, 0], Z[k, 1] = b, a
            break
    return Z


def build_link_color_func(Z, names):
    """Color each internal node by family if all its leaves share one,
    else neutral gray. scipy passes internal node ids (n .. 2n-2)."""
    n = len(names)
    leaves_of = {i: {i} for i in range(n)}
    for k, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        leaves_of[n + k] = leaves_of[a] | leaves_of[b]

    def color_for(node_id: int) -> str:
        leaves = leaves_of[node_id]
        fams = {FAMILY[names[i]] for i in leaves}
        if len(fams) == 1:
            return FAM_COLOR[next(iter(fams))]
        return NEUTRAL_LINK

    return color_for


def plot_one(pkl_path: Path, out_base: Path,
             color_threshold: float = 0.82) -> None:
    r = pickle.loads(pkl_path.read_bytes())
    Z = reorder_linkage(r["task_linkage"], list(r["task_names"]))
    names = list(r["task_names"])
    sim = np.asarray(r["task_similarity"])

    order = leaves_list(Z)
    sim_o = sim[np.ix_(order, order)]
    names_o = [names[i] for i in order]
    labels_o = [PRETTY[n] for n in names_o]
    fams_o = [FAMILY[n] for n in names_o]
    n = len(names_o)

    # Mask diagonal (always 1.0) so off-diagonal contrast is preserved.
    mat = sim_o.copy().astype(float)
    np.fill_diagonal(mat, np.nan)
    off_max = float(np.nanmax(mat))

    rc = paper_rc("double")
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.spines.left"] = False
    rc["axes.spines.bottom"] = False
    # Enlarge all figure text so it matches the dendrogram leaf labels.
    rc["xtick.labelsize"] = 11
    rc["ytick.labelsize"] = 11
    rc["axes.labelsize"]  = 11
    rc["legend.fontsize"] = 11
    rc["font.size"]       = 11

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(11.4, 5.7))
        gs = fig.add_gridspec(
            2, 4,
            width_ratios=[0.04, 1.0, 0.04, 1.55],
            height_ratios=[1.0, 0.10],
            wspace=0.04, hspace=0.30,
        )
        ax_cbar = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_band = fig.add_subplot(gs[0, 2])
        ax_dend = fig.add_subplot(gs[0, 3])
        ax_leg  = fig.add_subplot(gs[1, :])

        # Heatmap with a single-hue (indigo) ramp
        cmap = mcolors.LinearSegmentedColormap.from_list("sim", [
            "#F8F8FB", "#DCE2EE", "#A8B6D2",
            "#6B7AA8", "#3B447A", "#1A1F46",
        ])
        cmap.set_bad("#F4F4F1")
        im = ax_heat.imshow(
            mat, cmap=cmap, vmin=0.0, vmax=off_max,
            aspect="equal", interpolation="nearest",
        )
        ax_heat.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax_heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax_heat.grid(which="minor", color="white", lw=0.6)
        ax_heat.tick_params(which="minor", length=0)
        ax_heat.set_xticks(np.arange(n))
        ax_heat.set_xticklabels(labels_o, rotation=45, ha="right")
        ax_heat.set_yticks([])
        for tick, fam in zip(ax_heat.get_xticklabels(), fams_o):
            tick.set_color(FAM_COLOR[fam])
            tick.set_fontweight("semibold")
        ax_heat.tick_params(axis="x", length=2, pad=2, labelsize=8.5)

        # Colorbar (left)
        cbar = fig.colorbar(im, cax=ax_cbar)
        ax_cbar.yaxis.set_ticks_position("left")
        ax_cbar.yaxis.set_label_position("left")
        ax_cbar.set_ylabel("Similarity", labelpad=14)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=2)

        # Family color band on the right edge of the heatmap
        ax_band.set_xlim(0, 1)
        ax_band.set_ylim(n - 0.5, -0.5)
        for i, fam in enumerate(fams_o):
            ax_band.add_patch(Rectangle(
                (0, i - 0.5), 1, 1,
                facecolor=FAM_COLOR[fam], edgecolor="none",
            ))
        ax_band.set_xticks([]); ax_band.set_yticks([])
        for s in ax_band.spines.values():
            s.set_visible(False)

        # Dendrogram with leaf labels (docks with the tree). The family
        # band on the heatmap side already encodes color, so we keep the
        # leaf labels as plain text and color them by family.
        labels_orig = [PRETTY[nm] for nm in names]
        link_color_func = build_link_color_func(Z, names)
        dendrogram(
            Z,
            ax=ax_dend,
            orientation="right",
            labels=labels_orig,
            leaf_font_size=11,
            link_color_func=link_color_func,
            above_threshold_color=NEUTRAL_LINK,
        )
        ax_dend.invert_yaxis()

        label_to_name = {PRETTY[nm]: nm for nm in names}
        for tick in ax_dend.get_yticklabels():
            nm = label_to_name[tick.get_text()]
            tick.set_color(FAM_COLOR[FAMILY[nm]])
            tick.set_fontweight("semibold")

        d_min = float(Z[:, 2].min())
        d_max = float(Z[:, 2].max())
        ax_dend.axvline(color_threshold, ls="--", lw=0.9,
                        color="#B22222", alpha=0.8)
        ax_dend.text(
            color_threshold, ax_dend.get_ylim()[1],
            f"  cut = {color_threshold:.2f}",
            color="#B22222", fontsize=11, va="top", ha="left",
        )
        x_pad = 0.02
        ax_dend.set_xlim(max(0.0, d_min - x_pad), d_max + x_pad)
        ax_dend.set_xlabel("Distance  (1 − similarity)")
        for s in ["top", "right", "left"]:
            ax_dend.spines[s].set_visible(False)
        ax_dend.spines["bottom"].set_visible(True)
        ax_dend.tick_params(axis="x", length=3)
        ax_dend.tick_params(axis="y", length=0, pad=2)
        ax_dend.grid(axis="x", ls=":", lw=0.5, color="#CCCCCC", alpha=0.7)
        ax_dend.set_axisbelow(True)

        # Legend: "Task family:" + 5 entries, packed horizontally and
        # centered as one block.
        ax_leg.axis("off")
        title_box = TextArea(
            "Task family:",
            textprops=dict(weight="bold", fontsize=11.5, color="#222"),
        )
        entry_boxes = []
        for f in FAM_ORDER:
            da = DrawingArea(13, 13, 0, 0)
            da.add_artist(Circle((6.5, 6.5), 4.5,
                                  fc=FAM_COLOR[f], ec="none"))
            txt = TextArea(f"  {f}",
                            textprops=dict(fontsize=11, color="#222"))
            entry_boxes.append(
                HPacker(children=[da, txt], align="center", pad=0, sep=2)
            )
        legend_block = HPacker(
            children=[title_box] + entry_boxes,
            align="center", pad=0, sep=14,
        )
        legend_anchor = AnchoredOffsetbox(
            loc="center", child=legend_block,
            pad=0, borderpad=0, frameon=False,
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=ax_leg.transAxes,
        )
        ax_leg.add_artist(legend_anchor)

        # Sync side panels to ax_heat's actual position (aspect='equal'
        # shrinks the heatmap inside its gridspec slot).
        fig.canvas.draw()
        hp = ax_heat.get_position()
        for ax in (ax_band, ax_dend, ax_cbar):
            p = ax.get_position()
            ax.set_position([p.x0, hp.y0, p.width, hp.height])


        # Pull the band tight against the heatmap. Reserve a clear gap
        # to the right of the band before the dendrogram leaf labels,
        # so the colored band and the colored labels don't visually fuse.
        bp = ax_band.get_position()
        ax_band.set_position([hp.x1 + 0.004, hp.y0, bp.width, hp.height])
        bp = ax_band.get_position()
        dp = ax_dend.get_position()
        gap_after_band = 0.010         # tight but distinct from band
        labels_room    = 0.105         # space for leaf-label text
        ax_dend.set_position([
            bp.x1 + gap_after_band + labels_room, hp.y0,
            dp.x1 - (bp.x1 + gap_after_band + labels_room), hp.height,
        ])
        ax_dend.tick_params(axis="y", pad=2)

        # Slight nudge down so the legend clears the rotated x-tick labels
        lp = ax_leg.get_position()
        ax_leg.set_position([lp.x0, max(0.0, lp.y0 - 0.015),
                              lp.width, lp.height])

        # Re-anchor the legend to the actual visible content's horizontal
        # center (using tight bboxes of heatmap+dendrogram+labels), so the
        # post-`bbox_inches='tight'` figure shows it truly centered.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        bboxes = [
            ax_heat.get_tightbbox(renderer).transformed(inv),
            ax_dend.get_tightbbox(renderer).transformed(inv),
            ax_cbar.get_tightbbox(renderer).transformed(inv),
        ]
        x_lo = min(b.x0 for b in bboxes)
        x_hi = max(b.x1 for b in bboxes)
        cx = (x_lo + x_hi) / 2
        lp = ax_leg.get_position()
        cy = lp.y0 + lp.height / 2
        legend_anchor.set_bbox_to_anchor((cx, cy), transform=fig.transFigure)

        plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_base.with_suffix('.pdf')}")
        print(f"[saved] {out_base.with_suffix('.png')}")


def main() -> None:
    variants = sys.argv[1:] or [d.name for d in ROOT.iterdir() if d.is_dir()]
    for v in variants:
        pkl = ROOT / v / "cross_task_result.pkl"
        if not pkl.exists():
            print(f"[skip] {pkl} not found")
            continue
        plot_one(pkl, ROOT / v / "clustermap")


if __name__ == "__main__":
    main()
