"""Load paired clean/corrupted prompts for activation patching.

The CSV schema is detected from the header:
  clean, corrupted, correct_idx, incorrect_idx             IOI-style, logit_diff
  clean, corrupted, clean_answer_idx, corrupted_answer_idx gender-bias, logit_diff
  clean, corrupted, country_idx, corrupted_country_idx     fact retrieval, logit_diff
  clean, corrupted, profession_idx, corrupted_profession_idx  person-profession, logit_diff
  clean, corrupted, correct_idx                            greater-than; label is the
                                                           year YY0 (not a token id)
  clean, corrupted, group, plural                          SVA; label is plural in {0, 1}
"""
from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch


@dataclass
class IOIBatch:
    clean:        List[str]
    corrupted:    List[str]
    labels:       torch.Tensor   # (B, 2) for "logit_diff"; (B,) for "greater_than"
    n:            int
    task:         str = "logit_diff"   # "logit_diff" | "greater_than" | "sva"

    def __len__(self) -> int:
        return self.n


def load_ioi(
    csv_path: Path,
    n: int | None = None,
    *,
    gt_mode: str = "prob_diff",
    tokenizer = None,
    start: int = 0,
) -> IOIBatch:
    """Read rows [start, start+n) (or all from `start` if n is None) from a paired CSV.

    `start` lets callers pick a disjoint subset (e.g. sva front-100 vs sva_isare
    back-100 of the same 200-row csv).

    gt_mode (greater-than CSVs only):
      - "prob_diff" (default): labels=(B,) of YY₀, task="greater_than",
        metric = Σp(YY>YY₀) − Σp(YY≤YY₀).
      - "boundary_logit_diff": labels=(B, 2) = [year_ids[YY₀+1], year_ids[YY₀]],
        task="logit_diff", metric = logit(YY₀+1) − logit(YY₀). Requires
        `tokenizer`. Use this when gradients were collected with the boundary CE loss.
    """
    cleans, corrupts, labels = [], [], []
    task = "logit_diff"
    correct_col = incorrect_col = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: empty CSV / no header row")
        fields = set(reader.fieldnames)

        if {"correct_idx", "incorrect_idx"} <= fields:
            # IOI-style: both columns present
            correct_col, incorrect_col = "correct_idx", "incorrect_idx"
            task = "logit_diff"
        elif {"clean_answer_idx", "corrupted_answer_idx"} <= fields:
            # gender-bias-style
            correct_col, incorrect_col = "clean_answer_idx", "corrupted_answer_idx"
            task = "logit_diff"
        elif {"country_idx", "corrupted_country_idx"} <= fields:
            # fact-retrieval-style
            correct_col, incorrect_col = "country_idx", "corrupted_country_idx"
            task = "logit_diff"
        elif {"profession_idx", "corrupted_profession_idx"} <= fields:
            # person-profession-style
            correct_col, incorrect_col = "profession_idx", "corrupted_profession_idx"
            task = "logit_diff"
        elif {"plural", "group"} <= fields:
            # SVA-style: plural ∈ {0,1}, NOT a token id
            correct_col = "plural"
            task = "sva"
        elif "correct_idx" in fields:
            # greater-than-style: only correct_idx (a YY year value, not a token id)
            correct_col = "correct_idx"
            task = "greater_than"
        else:
            raise KeyError(
                f"{csv_path}: expected one of: "
                f"(correct_idx, incorrect_idx) | "
                f"(clean_answer_idx, corrupted_answer_idx) | "
                f"(country_idx, corrupted_country_idx) | "
                f"(profession_idx, corrupted_profession_idx) | "
                f"(plural, group) | "
                f"(correct_idx)  — got {reader.fieldnames}"
            )

        for i, row in enumerate(reader):
            if i < start:
                continue
            cleans.append(row["clean"])
            corrupts.append(row["corrupted"])
            if task == "greater_than" or task == "sva":
                labels.append(int(row[correct_col]))
            else:
                labels.append([int(row[correct_col]), int(row[incorrect_col])])
            if n is not None and len(cleans) >= n:
                break

    # Optionally rewrite greater-than prob_diff as a boundary logit_diff
    if task == "greater_than" and gt_mode == "boundary_logit_diff":
        if tokenizer is None:
            raise ValueError(
                "load_ioi(gt_mode='boundary_logit_diff') requires `tokenizer`."
            )
        # build year_ids[0..99] (single tokens for GPT-2; verified upstream)
        year_ids: List[int] = []
        multi = []
        for i in range(100):
            enc = tokenizer.encode(f"{i:02d}", add_special_tokens=False)
            (multi if len(enc) != 1 else year_ids).append(enc[0] if len(enc) == 1 else (i, enc))
        if multi:
            raise ValueError(
                f"[gt boundary] {len(multi)} of 100 YY strings not single-token "
                f"for this tokenizer (e.g. {multi[:3]})."
            )
        new_labels = []
        for yy0 in labels:
            yy_correct = yy0 + 1 if yy0 < 99 else yy0 - 1
            new_labels.append([year_ids[yy_correct], year_ids[yy0]])
        labels   = new_labels
        task     = "logit_diff"
    elif task == "greater_than" and gt_mode != "prob_diff":
        raise ValueError(
            f"Unknown gt_mode={gt_mode!r} (expected 'prob_diff' | 'boundary_logit_diff')"
        )

    labels_t = torch.tensor(labels, dtype=torch.long)
    return IOIBatch(
        clean=cleans, corrupted=corrupts,
        labels=labels_t, n=len(cleans), task=task,
    )
