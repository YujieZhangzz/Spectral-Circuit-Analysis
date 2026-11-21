"""Subsample each task's dataset into a smaller csv for quick experiments.

Reads eap_data/{task}/{model}.csv and writes eap_data/{task}/{suffix}_{model}.csv.
Sampling is uniform, except gender-bias (balanced over label) and sva
(stratified by group, keeping the proportion of each construction).

Usage:
    python eap_data/sample_datasets.py --models gpt2 --tasks ioi sva
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import numpy as np

# Default sample size per task
TASK_N = {
    "ioi":                  1000,
    "greater-than":         1000,
    "greater-than-sequence": 825,
    "greater-than-price":   1000,
    "fact-retrieval-comma":  190,
    "hypernymy-comma":       198,
    "gender-bias":           986,  # gpt2 has 986 rows, pythia 782; n is capped at the file size
    "sva":                 12400,
}

DATA_DIR = Path(__file__).parent


def sample_task(task: str, model: str, n: int, seed: int) -> pd.DataFrame:
    csv_path = DATA_DIR / task / f"{model}.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    total = len(df)
    n = min(n, total)

    if task == "gender-bias":
        # n//2 rows per label
        per_class = n // 2
        sampled = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(per_class, len(g)), random_state=seed))
        )
        sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    elif task == "sva":
        # proportional sampling per group
        sampled = (
            df.groupby("group", group_keys=False)
            .apply(lambda g: g.sample(
                max(1, round(n * len(g) / total)),
                random_state=seed
            ))
        )
        # rounding can overshoot n; truncate
        sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True).head(n)

    else:
        sampled = df.sample(n, random_state=seed).reset_index(drop=True)

    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", default=["gpt2", "pythia"],
                        help="model csv names to process (without .csv)")
    parser.add_argument("--tasks",   nargs="+", default=list(TASK_N.keys()),
                        help="tasks to process")
    parser.add_argument("--n",       type=int,  default=None,
                        help="sample size for all tasks (overrides per-task defaults)")
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--suffix",  type=str,  default="sampled_all",
                        help="output file prefix, e.g. sampled -> sampled_gpt2.csv")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    for task in args.tasks:
        n = args.n if args.n is not None else TASK_N.get(task, 200)
        print(f"\n[{task}]  target n={n}")

        for model in args.models:
            sampled = sample_task(task, model, n, args.seed)
            if sampled is None:
                continue

            out_path = DATA_DIR / task / f"{args.suffix}_{model}.csv"
            sampled.to_csv(out_path, index=False)
            print(f"  {model}: {len(sampled)} rows -> {out_path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
