"""Render perturb_default_{min,mean}.csv (from parameter_study.py) as LaTeX
booktabs tables in cross_task/<CT_RUN>/consistency_share/parameter_study/pdf/.

Usage: CT_RUN=run_gpt_12tasks python cross_task/parameter_study_table.py
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT  = REPO_ROOT / "cross_task" / os.environ.get("CT_RUN", "run_gpt_12tasks")
SRC_DIR   = RUN_ROOT / "consistency_share" / "parameter_study"
OUT_DIR   = SRC_DIR / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRETTY = {
    "universal":            r"$\theta_{\mathrm{univ}}$",
    "within_default":       r"$\theta_F^{\mathrm{floor}}$",
    "margin":               r"$\delta$",
    "task_dominance":       r"$\theta_{\mathrm{dom}}$",
    "task_outside_overlap": r"$\theta_{\mathrm{out}}$",
}
PARAM_ORDER = ["universal", "within_default", "margin",
               "task_dominance", "task_outside_overlap"]


def fmt(n: int) -> str:
    """Format an integer delta with explicit sign, '·' if 0."""
    if n == 0:
        return r"$\cdot$"
    return f"$+{n}$" if n > 0 else f"${n}$"


def render(criterion: str):
    csv_path = SRC_DIR / f"perturb_default_{criterion}.csv"
    rows = list(csv.DictReader(open(csv_path)))

    # Recover each default from its +10% row
    defaults = {}
    for r in rows:
        if r["perturb"] == "+10%" and r["param"] in PARAM_ORDER:
            v = float(r["value"])
            # +10% means default = v / 1.10
            defaults[r["param"]] = round(v / 1.10, 3)

    by_param = {p: {} for p in PARAM_ORDER}
    for r in rows:
        if r["param"] in PARAM_ORDER:
            by_param[r["param"]][r["perturb"]] = r

    perturbs = ["-20%", "-10%", "+10%", "+20%"]

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Local-perturbation sensitivity at $\pm10\%$ and $\pm20\%$"
        rf" of each threshold around its default (criterion = {criterion}). Each "
        rf"row reports the change in the count of every classification outcome "
        rf"and the total number of unit-level label flips relative to the default. "
        rf"A `$\cdot$' denotes zero change.}}",
        rf"  \label{{tab:perturb-{criterion}}}",
        r"  \small",
        r"  \begin{tabular}{llrrrrrrr}",
        r"    \toprule",
        r"    & & & \multicolumn{5}{c}{$\Delta$ count} & \\",
        r"    \cmidrule(lr){4-8}",
        r"    Threshold & Pert. & Value & Univ. & F-GT & F-SVA & Task & Mixed & \#flips \\",
        r"    \midrule",
    ]

    for p in PARAM_ORDER:
        for i, pct in enumerate(perturbs):
            r = by_param[p].get(pct)
            if r is None:
                continue
            sym = PRETTY[p] if i == 0 else ""
            val = f"${float(r['value']):.3g}$"
            lines.append(
                "    " +
                " & ".join([
                    sym, pct, val,
                    fmt(int(r["delta_universal"])),
                    fmt(int(r["delta_family_GT"])),
                    fmt(int(r["delta_family_SVA"])),
                    fmt(int(r["delta_task"])),
                    fmt(int(r["delta_mixed"])),
                    str(int(r["n_flips"])),
                ]) + r" \\"
            )
        if p != PARAM_ORDER[-1]:
            lines.append(r"    \addlinespace[2pt]")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    out = OUT_DIR / f"perturb_table_{criterion}.tex"
    out.write_text("\n".join(lines))
    print(f"[saved] {out}")


def main():
    for c in ["mean", "min"]:
        render(c)


if __name__ == "__main__":
    main()
