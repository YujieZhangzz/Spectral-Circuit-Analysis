"""Build a (clean, corrupted, correct_idx, incorrect_idx) CSV from IOI
paraphrases, readable by circuits_discovery.faithfulness.data.load_ioi.
The corrupted prompt replaces the last occurrence of the subject name with the
replacement name from the matching original IOI prompt, so clean and corrupted
differ only at the S token.

Usage:
    python -m circuits_discovery.faithfulness.build_paraphrase_csv \
        --paraphrase_json data/ioi_sampled_gpt2_200_paraphrases.json \
        --orig_csv        eap_data/ioi/sampled_all_gpt2_200.csv \
        --out_csv         results/eap_paraphrase_knockout/paraphrase.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path


def derive_replacement_name(clean: str, corrupted: str) -> str | None:
    """The original IOI corrupted prompt differs from clean at exactly one word
    (the S-name swap). Return that replacement name, or None if no single-word diff."""
    cw, kw = clean.split(), corrupted.split()
    if len(cw) != len(kw):
        return None
    diffs = [(i, c, k) for i, (c, k) in enumerate(zip(cw, kw)) if c != k]
    if len(diffs) != 1:
        return None
    return diffs[0][2].strip(",.;:")


def replace_last(text: str, target: str, replacement: str) -> str | None:
    """Replace the LAST whole-word occurrence of `target` with `replacement`.
    Returns None if target not found."""
    pattern = re.compile(rf"\b{re.escape(target)}\b")
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return text[:m.start()] + replacement + text[m.end():]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraphrase_json", type=Path, required=True)
    ap.add_argument("--orig_csv",        type=Path, required=True,
                    help="EAP IOI csv with matching `clean` rows so we can "
                         "look up the replacement S-name per item.")
    ap.add_argument("--out_csv",         type=Path, required=True)
    ap.add_argument("--max_per_item",    type=int, default=None,
                    help="cap paraphrases per item (default: all)")
    args = ap.parse_args()

    items = json.loads(args.paraphrase_json.read_text())
    orig_rows = list(csv.DictReader(args.orig_csv.open()))
    by_clean = {r["clean"]: r for r in orig_rows}

    # Pre-resolve replacement name + valid paraphrases per item so we can
    # interleave by paraphrase index across items.
    rows_by_item: list[list[tuple]] = []   # one list of rows per item
    n_skipped_unmatched = n_skipped_subject = 0
    for it in items:
        orig = by_clean.get(it["clean"])
        if orig is None:
            n_skipped_unmatched += 1
            continue
        replacement = derive_replacement_name(orig["clean"], orig["corrupted"])
        if replacement is None:
            n_skipped_unmatched += 1
            continue
        correct_idx, incorrect_idx = orig["correct_idx"], orig["incorrect_idx"]
        subject = it["subject"]
        paras = it["strong_semantic_structural_paraphrases"]
        if args.max_per_item is not None:
            paras = paras[: args.max_per_item]
        item_rows = []
        for p in paras:
            clean_q = p["query"]
            corrupted_q = replace_last(clean_q, subject, replacement)
            if corrupted_q is None:
                n_skipped_subject += 1
                continue
            meta = p.get("meta", {}) or {}
            item_rows.append((
                clean_q, corrupted_q, correct_idx, incorrect_idx,
                it["clean"], subject, replacement,
                meta.get("template_id", ""),
                "|".join(meta.get("structural_ops", [])),
            ))
        rows_by_item.append(item_rows)

    # Interleave by paraphrase index: row k = item k%N, paraphrase k//N, so
    # the first N rows hold one paraphrase per item (comparable to --n N on orig).
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    max_per = max((len(r) for r in rows_by_item), default=0)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clean", "corrupted", "correct_idx", "incorrect_idx",
                    "src_clean", "subject", "replacement",
                    "template_id", "structural_ops"])
        for p_idx in range(max_per):
            for item_rows in rows_by_item:
                if p_idx < len(item_rows):
                    w.writerow(item_rows[p_idx])
                    n_written += 1

    print(f"[built] {args.out_csv}")
    print(f"  written:                 {n_written}")
    print(f"  skipped (no orig match): {n_skipped_unmatched}")
    print(f"  skipped (no S in para):  {n_skipped_subject}")


if __name__ == "__main__":
    main()
