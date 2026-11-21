"""Table of head patching vs z-subspace patching recovery at several K, per
ranking method (console + markdown). Rows: head, z-sub, Δ = z-sub - head, and
capture = (1 - rec_zsub) / (1 - rec_head), the fraction of the head-patching
damage reproduced by z-sub (omitted when head patching drops < 5%).

Usage:
    python -m circuits_discovery.faithfulness.table_head_vs_zsub
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


KEY_KS = [5, 10, 20, 26, 30, 50, 100]

# (display, head-key-in-curves, zsub-tag)
PAIRS = [
    ("SVD (ours)", "SVD (ours)", "svd"),
    ("EAP",        "EAP",        "eap"),
    ("EAP-IG",     "EAP-IG",     "eap_ig"),
    ("EAP-IG-KL",  "EAP-IG-KL",  "eap_ig_kl"),
]


def auc(ks: np.ndarray, ys: np.ndarray) -> float:
    return float(np.trapz(ys, ks) / (ks[-1] - ks[0]))


def load(args):
    head = json.loads((args.head_dir / "head_faithfulness.json").read_text())
    head_ks = np.array(head["ks"])
    span = head["ld_clean"] - head["ld_corrupted"]
    head_recovery = {k: (np.array(v) - head["ld_corrupted"]) / span
                     for k, v in head["curves"].items()}

    zsub = {}
    for _, _, tag in PAIRS:
        path = args.zsub_dir / f"subspace_z_{tag}.json"
        if not path.exists(): continue
        d = json.loads(path.read_text())
        ks = np.array(d["ks"])
        sp = d["ld_clean"] - d["ld_corrupted"]
        rec = (np.array(d["logit_diff"]) - d["ld_corrupted"]) / sp
        zsub[tag] = (ks, rec)
    return head_ks, head_recovery, zsub


def at_k(ks, ys, k):
    i = int(np.where(ks == k)[0][0])
    return float(ys[i])


def fmt_console(head_ks, head_recovery, zsub) -> str:
    lines = []
    header = (f"  {'method':<15s} {'protocol':<8s} "
              + " ".join(f"{f'K={k}':>8s}" for k in KEY_KS)
              + f"  {'AUC':>7s}  {'min':>7s}  {'K*':>4s}")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for display, head_key, tag in PAIRS:
        # head row
        if head_key in head_recovery:
            y = head_recovery[head_key]
            row = [f"{at_k(head_ks, y, k):>+8.3f}" for k in KEY_KS]
            lines.append(f"  {display:<15s} {'head':<8s} " + " ".join(row)
                         + f"  {auc(head_ks, y):>+7.3f}"
                         + f"  {y.min():>+7.3f}  {head_ks[int(np.argmin(y))]:>4d}")
        # z-sub row
        if tag in zsub:
            ks_z, y_z = zsub[tag]
            row = [f"{at_k(ks_z, y_z, k):>+8.3f}" for k in KEY_KS]
            lines.append(f"  {'':<15s} {'z-sub':<8s} " + " ".join(row)
                         + f"  {auc(ks_z, y_z):>+7.3f}"
                         + f"  {y_z.min():>+7.3f}  {ks_z[int(np.argmin(y_z))]:>4d}")
        # Δ (z-sub − head) and capture rows
        if head_key in head_recovery and tag in zsub:
            y_h = head_recovery[head_key]
            ks_z, y_z = zsub[tag]
            deltas = [at_k(ks_z, y_z, k) - at_k(head_ks, y_h, k) for k in KEY_KS]
            lines.append(f"  {'':<15s} {'Δ':<8s} "
                         + " ".join(f"{d:>+8.3f}" for d in deltas))
            # capture = (1 - z_sub) / (1 - head); skipped when head damage < 5%.
            caps = []
            for k in KEY_KS:
                rh = at_k(head_ks, y_h, k)
                rz = at_k(ks_z, y_z, k)
                drop_h = 1.0 - rh
                if drop_h < 0.05:
                    caps.append(None)        # head barely damaged; ratio meaningless
                else:
                    caps.append((1.0 - rz) / drop_h)
            cells = []
            for c in caps:
                cells.append(f"{c:>7.0%}  " if c is not None else f"{'—':>7s}  ")
            lines.append(f"  {'':<15s} {'capture':<8s} " + " ".join(cells))
        lines.append("")
    return "\n".join(lines)


def fmt_markdown(head_ks, head_recovery, zsub) -> str:
    head_row  = (["Method", "Protocol"] + [f"K={k}" for k in KEY_KS]
                 + ["AUC", "min", "K*"])
    sep = ["---"] * len(head_row)
    rows = []

    for display, head_key, tag in PAIRS:
        if head_key in head_recovery:
            y = head_recovery[head_key]
            r = ([display, "head"]
                 + [f"{at_k(head_ks, y, k):+.3f}" for k in KEY_KS]
                 + [f"{auc(head_ks, y):+.3f}",
                    f"{y.min():+.3f}",
                    f"{head_ks[int(np.argmin(y))]}"])
            rows.append(r)
        if tag in zsub:
            ks_z, y_z = zsub[tag]
            r = (["", "z-sub"]
                 + [f"{at_k(ks_z, y_z, k):+.3f}" for k in KEY_KS]
                 + [f"{auc(ks_z, y_z):+.3f}",
                    f"{y_z.min():+.3f}",
                    f"{ks_z[int(np.argmin(y_z))]}"])
            rows.append(r)
        if head_key in head_recovery and tag in zsub:
            y_h = head_recovery[head_key]
            ks_z, y_z = zsub[tag]
            deltas = [at_k(ks_z, y_z, k) - at_k(head_ks, y_h, k) for k in KEY_KS]
            rows.append(["", "Δ"] + [f"{d:+.3f}" for d in deltas] + ["—", "—", "—"])
            caps = []
            for k in KEY_KS:
                rh = at_k(head_ks, y_h, k)
                rz = at_k(ks_z, y_z, k)
                drop_h = 1.0 - rh
                caps.append((1.0 - rz) / drop_h if drop_h >= 0.05 else None)
            rows.append(["", "capture"]
                        + [f"{c:.0%}" if c is not None else "—" for c in caps]
                        + ["—", "—", "—"])

    out = ["| " + " | ".join(head_row) + " |",
           "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_dir", type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/faithfulness_head"))
    ap.add_argument("--zsub_dir", type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/faithfulness_subspace_z"))
    ap.add_argument("--out_md", type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/faithfulness_head/"
                                 "head_vs_zsub_table.md"))
    args = ap.parse_args()

    head_ks, head_recovery, zsub = load(args)

    print()
    print(fmt_console(head_ks, head_recovery, zsub))

    md = fmt_markdown(head_ks, head_recovery, zsub)
    args.out_md.write_text(md)
    print(f"\n[saved markdown] {args.out_md}")


if __name__ == "__main__":
    main()
