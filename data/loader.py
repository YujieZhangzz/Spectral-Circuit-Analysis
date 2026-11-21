"""Dataset loaders and task-specific losses.

Each load_records_* function parses one paraphrase JSON format into the
common record format used by gradient_collector. make_batch tokenizes a
(query, completion) pair with query labels masked to -100, so the loss is
computed on the completion only.
"""

from __future__ import annotations
import json
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


def _extract_subject_name(query: str, io_name: str) -> Optional[str]:
    """Return the subject name from an IOI query (the capitalized word that appears twice)."""
    words = re.findall(r'\b[A-Z][a-z]+\b', query)
    counts = Counter(words)
    for word, count in counts.most_common():
        if word != io_name and count >= 2:
            return word
    # fallback: any capitalized word that is not the IO name
    for word in words:
        if word != io_name:
            return word
    return None


# Paraphrase sources

SOURCE_MAP = {
    "semantic_only":    ["semantic_transformation"],
    "translation_only": ["translation"],
    "semantic_with_original": ["original", "semantic_transformation"],
    "translation_with_original": ["original", "translation"],
    "both":             ["semantic_transformation", "translation"],
    "all":              ["original", "semantic_transformation", "translation"],
}


def load_records(
    path: str,
    source: str,
    N: Optional[int] = None,
) -> List[Dict]:
    """
    Returns a list of records:
    {
        "instance_id":            int,
        "original_query":         str,
        "original_completion":    str,
        "paraphrase_queries":     List[str],   length M
        "paraphrase_completions": List[str],
    }
    """
    if source not in SOURCE_MAP:
        raise ValueError(f"Unknown source '{source}'. "
                         f"Choose from: {list(SOURCE_MAP)}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if N is not None:
        raw = raw[:N]

    records = []
    for idx, entry in enumerate(raw):
        clean = entry["clean"]
        orig  = clean["original"]

        queries, completions = [], []

        for src_key in SOURCE_MAP[source]:
            if src_key == "original":
                queries.append(orig["query"])
                completions.append(orig["completion"].strip())
            else:
                for item in clean.get(src_key, []):
                    queries.append(item["query"])
                    completions.append(item["completion"].strip())

        if not queries:
            continue

        io_name      = orig["completion"].strip().lstrip()
        subject_name = _extract_subject_name(orig["query"], io_name)
        records.append({
            "instance_id":            idx,
            "original_query":         orig["query"],
            "original_completion":    orig["completion"].strip(),
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                io_name,
            "subject_name":           subject_name,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance, source='{source}'")
    return records


def load_records_cap(
    path: str,
    source: str,
    N: Optional[int] = None,
) -> List[Dict]:
    """
    Load cap_paraphrases.json (country-capital format).

    JSON structure per entry:
      {
        "original":               {"query": "...", "completion": " Paris"},
        "translation":            [...] | null,
        "semantic_transformation":[...] | null,
        "meta":                   {"country": ..., "capital": ..., "template": ...}
      }

    Returns the same record format as load_records() so the rest of the
    pipeline (gradient_collector, run.py) works without modification.
    """
    if source not in SOURCE_MAP:
        raise ValueError(f"Unknown source '{source}'. "
                         f"Choose from: {list(SOURCE_MAP)}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if N is not None:
        raw = raw[:N]

    def _norm_completion(c: str) -> str:
        return c.strip()

    records = []
    for idx, entry in enumerate(raw):
        orig = entry["original"]
        queries, completions = [], []

        for src_key in SOURCE_MAP[source]:
            if src_key == "original":
                queries.append(orig["query"])
                completions.append(_norm_completion(orig["completion"]))
            else:
                items = entry.get(src_key) or []
                for item in items:
                    queries.append(item["query"])
                    completions.append(_norm_completion(item["completion"]))

        if not queries:
            continue

        records.append({
            "instance_id":            idx,
            "original_query":         orig["query"],
            "original_completion":    _norm_completion(orig["completion"]),
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "meta":                   entry.get("meta", {}),
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/cap] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance, source='{source}'")
    return records


# gp_paraphrases.json has the same structure as cap, so the same loader is reused
def load_records_gp(
    path: str,
    source: str,
    N: Optional[int] = None,
) -> List[Dict]:
    """Load gp_paraphrases.json (gendered-pronoun format, same structure as cap)."""
    records = load_records_cap(path, source, N)
    print(f"[Loader/gp] loaded {len(records)} instances, source='{source}'")
    return records


def load_records_ioi_strong(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,       # first M paraphrases per instance (None = all)
) -> List[Dict]:
    """
    Load ioi_dataset_strong_paraphrases.json.

    JSON structure:
      {
        "metadata": {...},
        "data": [
          {
            "clean": {"query": "...<trailing space>", "completion": "<IO>"},
            "completion": "<IO>",
            "subject": "<Subject>",
            "strong_semantic_structural_paraphrases": [
              {"query": "...<trailing space>", "completion": "<IO>", "meta": {...}},
              ...
            ]
          }
        ]
      }
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        clean       = entry["clean"]
        orig_query  = clean["query"]
        completion  = clean["completion"].strip()
        subject     = entry.get("subject", "").strip() or None

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]
        queries      = [p["query"] for p in paraphrases]
        completions  = [p["completion"].strip() for p in paraphrases]

        if not queries:
            continue

        if subject is None:
            subject = _extract_subject_name(orig_query, completion)

        records.append({
            "instance_id":            idx,
            "original_query":         orig_query,
            "original_completion":    completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                completion,
            "subject_name":           subject,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/ioi_strong] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


def load_records_ioi_sampled(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
) -> List[Dict]:
    """
    Load ioi_sampled_*_paraphrases.json (the M-sweep schema).

    JSON structure (top-level list):
      [
        {
          "clean":           "<full prompt str, NO trailing space>",
          "correct_token":   " <IO>",            # has leading space
          "corrupted_token": " <Subject>",       # has leading space
          "subject":         "<Subject>",
          "strong_semantic_structural_paraphrases": [
            {"query": "<prompt>", "correct_token": " <IO>",
             "corrupted_token": " <Subject>", "meta": {...}},
            ...
          ]
        },
        ...
      ]

    Returns the same record format as load_records() so the rest of the
    pipeline works without modification. The completion fields are kept
    verbatim (with leading space), since query+completion reconstructs
    correct text only when leading space is preserved.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", raw)

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        original_query      = entry["clean"]              # str (no trailing space)
        original_completion = entry["correct_token"]      # " IO" (with leading space)
        subject             = entry.get("subject", "").strip() or None

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]
        queries     = [p["query"]         for p in paraphrases]
        completions = [p["correct_token"] for p in paraphrases]

        if not queries:
            continue

        records.append({
            "instance_id":            idx,
            "original_query":         original_query,
            "original_completion":    original_completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                original_completion.strip() or original_completion,
            "subject_name":           subject,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/ioi_sampled] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


def load_records_gb_sampled(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
) -> List[Dict]:
    """
    Load gb_sampled_*_paraphrases.json (gender-bias M-sweep schema).

    JSON structure (top-level list):
      [
        {
          "clean":            "<prompt str, NO trailing space>",
          "corrupted":        "<corrupted prompt>",
          "clean_answer":     " she",        # has leading space
          "corrupted_answer": " he",
          "subject":          "<subject>",
          "strong_semantic_structural_paraphrases": [
            {"query": "<prompt>", "clean_answer": " she", "meta": {...}},
            ...
          ]
        },
        ...
      ]
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", raw)

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        original_query      = entry["clean"]
        original_completion = entry["clean_answer"]
        subject             = entry.get("subject", "").strip() or None

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]
        queries     = [p["query"]        for p in paraphrases]
        completions = [p["clean_answer"] for p in paraphrases]

        if not queries:
            continue

        records.append({
            "instance_id":            idx,
            "original_query":         original_query,
            "original_completion":    original_completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                original_completion.strip() or original_completion,
            "subject_name":           subject,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/gb_sampled] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


def load_records_gp_sampled(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
) -> List[Dict]:
    """
    Load gp_sampled_*_paraphrases.json (gendered-pronoun M-sweep schema).

    JSON structure (top-level list):
      [
        {
          "clean_query":          "<prompt str, NO trailing space>",
          "corrupted_query":      "<corrupted prompt>",
          "clean_completion":     " he",          # has leading space
          "corrupted_completion": " she",
          "subject":              "<name>",
          "strong_semantic_structural_paraphrases": [
            {"clean": "<prompt>", "corrupted": "<...>",
             "clean_completion": " he", "corrupted_completion": " she",
             "meta": {...}},
            ...
          ]
        },
        ...
      ]
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", raw)

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        original_query      = entry["clean_query"]
        original_completion = entry["clean_completion"]
        subject             = entry.get("subject", "").strip() or None

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]
        queries     = [p["clean"]            for p in paraphrases]
        completions = [p["clean_completion"] for p in paraphrases]

        if not queries:
            continue

        records.append({
            "instance_id":            idx,
            "original_query":         original_query,
            "original_completion":    original_completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                original_completion.strip() or original_completion,
            "subject_name":           subject,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/gp_sampled] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


def load_records_person_country(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
) -> List[Dict]:
    """
    Load pc_sampled_*_paraphrases.json (person -> nationality M-sweep schema).

    JSON structure (top-level list):
      [
        {
          "clean":            "<prompt>",        # leading space, NO trailing space
          "corrupted":        "<corrupted>",
          "clean_answer":     " Israel",         # leading space
          "corrupted_answer": " England",
          "subject":          "<Person>",
          "strong_semantic_structural_paraphrases": [
            {"clean": "<prompt>", "corrupted": "<...>",
             "clean_answer": " Israel", "corrupted_answer": " England",
             "meta": {...}},
            ...
          ]
        },
        ...
      ]
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", raw)

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        original_query      = entry["clean"]
        original_completion = entry["clean_answer"]
        subject             = entry.get("subject", "").strip() or None

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]
        queries     = [p["clean"]        for p in paraphrases]
        completions = [p["clean_answer"] for p in paraphrases]

        if not queries:
            continue

        records.append({
            "instance_id":            idx,
            "original_query":         original_query,
            "original_completion":    original_completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                original_completion.strip() or original_completion,
            "subject_name":           subject,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/person_country] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


def load_records_ioi_v3(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
) -> List[Dict]:
    """
    Load ioi_v3_strong_semantic_structural_strict_prefix.json.

    JSON structure:
      {
        "metadata": {...},
        "data": [
          {
            "clean": "...<query> <IO>",
            "completion": "<IO>",
            "strong_semantic_structural_paraphrases": [
              {"query": "...<trailing space>", "completion": "<IO>", "meta": {...}},
              ...
            ]
          }
        ]
      }

    Returns the same record format as load_records() so the rest of the
    pipeline works without modification.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        completion = entry["completion"].strip()
        clean_str  = entry["clean"]

        # Reconstruct original query: remove trailing IO name from clean string
        if clean_str.endswith(completion):
            original_query = clean_str[: -len(completion)]
        else:
            original_query = clean_str + " "

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]
        queries      = [p["query"] for p in paraphrases]
        completions  = [p["completion"].strip() for p in paraphrases]

        if not queries:
            continue

        subject_name = _extract_subject_name(original_query, completion)
        records.append({
            "instance_id":            idx,
            "original_query":         original_query,
            "original_completion":    completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                completion,
            "subject_name":           subject_name,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/ioi_v3] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


def load_records_gt(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
) -> List[Dict]:
    """
    Load greater-than paraphrase JSON (gt_sampled_*_paraphrases.json).

    JSON structure (top-level list):
      [
        {
          "clean":          "<prompt ending in ' the year 17'>",
          "corrupted":      "<corrupted prompt>",
          "correct_idx":    <int 0..99>,        # YY₀ threshold (NOT a target)
          "subject":        "1702",
          "noun":           "plan",
          "end_century":    "17",
          "strong_semantic_structural_paraphrases": [
            {"clean": "...", "corrupted": "...", "correct_idx": <int>, "meta": {...}},
            ...
          ]
        },
        ...
      ]

    Greater-than has no single "correct token"; the answer space is
    {YY : YY > YY₀}. We therefore use a custom prob-diff loss instead of CE
    (see make_gt_prob_diff_loss_fn). The completion stored in records is just
    a 2-digit YY₀ placeholder (with leading space) so that make_batch can locate
    the prediction position; the loss_fn ignores its content and reads the
    full year-token logits at that position.

    Each paraphrase carries its own correct_idx in `loss_extras_per_paraphrase`,
    consumed by gradient_collector.collect -> obtain_gradients(loss_fn, loss_extras).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", raw)

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        original_query = entry["clean"]
        yy0_orig       = int(entry["correct_idx"])
        original_completion = f" {yy0_orig:02d}"   # placeholder; loss_fn ignores
        subject        = entry.get("subject", "").strip() or None

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]

        queries     = [p["clean"] for p in paraphrases]
        completions = [f" {int(p['correct_idx']):02d}" for p in paraphrases]
        loss_extras = [{"correct_idx": int(p["correct_idx"])} for p in paraphrases]

        if not queries:
            continue

        records.append({
            "instance_id":            idx,
            "original_query":         original_query,
            "original_completion":    original_completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "loss_extras_per_paraphrase": loss_extras,
            "subject_name":           subject,
            "noun":                   entry.get("noun"),
            "end_century":            entry.get("end_century"),
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/gt] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


def build_gt_year_ids(tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """
    Tokenize "00".."99" standalone and return a (100,) LongTensor of token ids.

    Used by the GT prob-diff loss to read logits at the YY-prediction position.
    All 100 strings are single tokens in GPT-2 BPE (verified). For other
    tokenizers we assert the single-token property; if it fails the caller
    needs a custom valid-token list (cf. Hanna et al. 2023's NYT subset).
    """
    ids: List[int] = []
    multi: List[Tuple[int, List[int]]] = []
    for i in range(100):
        enc = tokenizer.encode(f"{i:02d}", add_special_tokens=False)
        if len(enc) != 1:
            multi.append((i, enc))
        ids.append(enc[0] if len(enc) == 1 else -1)
    if multi:
        raise ValueError(
            f"[gt] {len(multi)} of 100 YY strings are not single tokens for "
            f"this tokenizer (e.g. {multi[:3]}). Use a tokenizer-specific "
            f"valid-YY subset."
        )
    return torch.tensor(ids, dtype=torch.long)


def make_gt_prob_diff_loss_fn(year_ids: torch.Tensor) -> Callable:
    """
    Returns a loss_fn(outputs, batch, extras) -> scalar tensor for greater-than.

    Loss = -(logsumexp p(YY > YY₀) - logsumexp p(YY ≤ YY₀))    on the clean
    prompt's prediction position (last input token). Minimizing this loss
    pushes mass onto valid YYs and away from invalid ones, the same direction
    as the prob_diff metric used in the knockouts.

    `extras["correct_idx"]` is YY₀ for the current paraphrase (0..99).
    Prediction position is taken from batch["labels"]: the first index where
    label != -100 is the first completion token, so logits[pred_pos - 1]
    predicts it.
    """
    year_ids_local = year_ids   # captured by closure; moved to device on first call

    def loss_fn(outputs, batch: dict, extras: Optional[dict]) -> torch.Tensor:
        if extras is None or "correct_idx" not in extras:
            raise ValueError("gt_prob_diff_loss requires extras['correct_idx']")
        yy0 = int(extras["correct_idx"])

        labels = batch["labels"][0]                        # (seq,)
        nonignored = (labels != -100).nonzero(as_tuple=True)[0]
        if nonignored.numel() == 0:
            raise ValueError("gt_prob_diff_loss: no completion tokens in labels")
        cs = int(nonignored[0].item())                     # first completion idx
        pred_pos = cs - 1
        last_logits = outputs.logits[0, pred_pos, :]       # (vocab,)

        log_probs = F.log_softmax(last_logits, dim=-1)
        ids = year_ids_local.to(log_probs.device)
        invalid_ids = ids[: yy0 + 1]                       # YY ≤ YY₀
        valid_ids   = ids[yy0 + 1 :]                       # YY > YY₀

        log_p_v = torch.logsumexp(log_probs[valid_ids],   dim=0)
        log_p_i = torch.logsumexp(log_probs[invalid_ids], dim=0)
        return -(log_p_v - log_p_i)

    return loss_fn


def make_gt_boundary_logit_diff_loss_fn(year_ids: torch.Tensor) -> Callable:
    """
    Returns loss_fn(outputs, batch, extras) -> scalar tensor for greater-than,
    using a 2-token logit_diff on the boundary pair (YY₀+1, YY₀).

    Loss = -(logit[YY₀+1] - logit[YY₀])  on the prediction position.

    Compared to make_gt_prob_diff_loss_fn:
      - that variant aggregates over all 100 single-token YYs via logsumexp
        (matches eap-ig-faithfulness prob_diff_greater_than metric);
      - this variant uses a single boundary pair, giving a sharper
        per-instance signal: different YY₀ values give different gradient
        directions, whereas the prob_diff loss tends to collapse to one
        dominant direction (close to the magnitude baseline).

    Edge case YY₀=99 (no valid YY > 99): fall back to (YY₀-1, YY₀); rare in the
    dataset.

    Use with:
      - eap-ig-faithfulness task='greater-than-boundary' (logit_diff metric)
      - knockout gt_mode='boundary_logit_diff'
        (circuits_discovery/faithfulness/data.py)
    """
    year_ids_local = year_ids

    def loss_fn(outputs, batch: dict, extras: Optional[dict]) -> torch.Tensor:
        if extras is None or "correct_idx" not in extras:
            raise ValueError("gt_boundary_logit_diff requires extras['correct_idx']")
        yy0 = int(extras["correct_idx"])
        target_yy = yy0 + 1 if yy0 < 99 else yy0 - 1
        target_id = int(year_ids_local[target_yy].item())
        other_id  = int(year_ids_local[yy0].item())

        labels = batch["labels"][0]
        nonignored = (labels != -100).nonzero(as_tuple=True)[0]
        if nonignored.numel() == 0:
            raise ValueError("gt_boundary_logit_diff: no completion tokens in labels")
        pred_pos = int(nonignored[0].item()) - 1
        last_logits = outputs.logits[0, pred_pos, :]

        target_l = last_logits[target_id]
        other_l  = last_logits[other_id]
        return -(target_l - other_l)

    return loss_fn


def make_gt_boundary_ce_loss_fn(year_ids: torch.Tensor) -> Callable:
    """
    Returns a loss_fn(outputs, batch, extras) -> scalar tensor for greater-than,
    using single-token CE on the boundary YY = YY₀ + 1 (smallest valid).

    Under the prob_diff loss all instances share the same valid/invalid token
    sets (only the threshold differs), so gradient directions collapse and
    SVD importance reduces to the magnitude baseline. Targeting one
    per-instance token keeps the gradient directions distinct across YY₀.

    Edge case YY₀=99 (no valid YY > 99): fall back to YY₀-1, which is the
    only well-defined "near-threshold" alternative. Such samples should be
    rare in the dataset.
    """
    year_ids_local = year_ids

    def loss_fn(outputs, batch: dict, extras: Optional[dict]) -> torch.Tensor:
        if extras is None or "correct_idx" not in extras:
            raise ValueError("gt_boundary_ce requires extras['correct_idx']")
        yy0 = int(extras["correct_idx"])
        target_yy = yy0 + 1 if yy0 < 99 else yy0 - 1
        target_id = int(year_ids_local[target_yy].item())

        labels = batch["labels"][0]
        nonignored = (labels != -100).nonzero(as_tuple=True)[0]
        if nonignored.numel() == 0:
            raise ValueError("gt_boundary_ce: no completion tokens in labels")
        pred_pos = int(nonignored[0].item()) - 1
        last_logits = outputs.logits[0, pred_pos, :]

        log_probs = F.log_softmax(last_logits, dim=-1)
        return -log_probs[target_id]

    return loss_fn


# SVA (subject-verb agreement, Marvin & Linzen)

def load_records_sva(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
    start: int = 0,                # skip the first `start` instances before taking N
) -> List[Dict]:
    """
    Load SVA paraphrase JSON.

    Expected JSON (top-level list):
      [
        {
          "clean":          "<prompt; matrix verb missing at end>",
          "corrupted":      "<prompt with subject number flipped>",
          "plural":         0 | 1,             # 0 = singular subj, 1 = plural subj
          "subject":        "song",
          "clean_answer":   " is",             # 1-token gold continuation (placeholder)
          "corrupted_answer": " are",
          "strong_semantic_structural_paraphrases": [
            {"clean": "...", "corrupted": "...",
             "clean_answer": " is", "corrupted_answer": " are",
             "meta": {...}},
            ...
          ]
        },
        ...
      ]

    Loss is sva_prob_diff (full singular vs plural verb-set), not CE on
    `clean_answer`. The completion stored in records is just `clean_answer`
    so make_batch can locate the prediction position; the loss_fn ignores
    the target value and reads logsumexp over the verb-set IDs at that pos.

    Each paraphrase carries `{"plural": int}` in `loss_extras_per_paraphrase`;
    plural label is shared across paraphrases of one instance.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", raw)

    # Slice [start, start+N) so callers can pick disjoint subsets
    # (e.g. sva uses [0, 100), sva_isare uses [100, 200) of the same JSON).
    if start:
        data = data[start:]
    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        plural          = int(entry["plural"])
        original_query  = entry["clean"]
        # Gold continuation is " is" / " are" / etc.; we store it but loss_fn
        # ignores its identity. Falls back to a generic " is"/" are" if missing.
        original_completion = entry.get(
            "clean_answer", " is" if plural == 0 else " are"
        )

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]

        queries     = [p["clean"] for p in paraphrases]
        completions = [
            p.get("clean_answer", " is" if plural == 0 else " are")
            for p in paraphrases
        ]
        loss_extras = [{"plural": plural} for _ in paraphrases]

        if not queries:
            continue

        records.append({
            "instance_id":            start + idx,        # absolute idx into source JSON
            "original_query":         original_query,
            "original_completion":    original_completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "loss_extras_per_paraphrase": loss_extras,
            "subject_name":           entry.get("subject"),
            "plural":                 plural,
            "group":                  entry.get("group"),
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/sva] N={len(records)} instances "
          f"(start={start}, slice=[{start}, {start + len(records)})), "
          f"M={M_values[0]} paraphrases/instance")
    return records


def build_sva_verb_ids(
    tokenizer: PreTrainedTokenizer,
    verb_list_csv: str,
    strict: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (singular_ids, plural_ids), the single-token verb token ids,
    mirroring eap.metrics.get_singular_and_plural exactly.

    Behavior:
      - Iterate every token in the vocab. Keep tokens whose detokenized form
        starts with the BPE word-initial marker (Ġ for gpt2/pythia BPE).
      - Singular: include if the stripped word is in singular_set.
      - Plural:   include only if its corresponding singular form is *also*
        a single-token (under strict=True). Otherwise plural set ends up much
        bigger than singular set, biasing prob_diff.

    Returns LongTensors on CPU; caller moves to device.
    """
    import pandas as pd  # local import; pandas is heavy and not needed elsewhere

    df = pd.read_csv(verb_list_csv)
    sing_list = df["sing"].tolist()
    plur_list = df["plur"].tolist()
    sing_set  = set(sing_list)
    plur_set  = set(plur_list)
    assert not (sing_set & plur_set), "verb list has overlap between sing/plur"

    sing_ids: List[int] = []
    plur_ids: List[int] = []
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    for i in range(vocab_size):
        tok = tokenizer._convert_id_to_token(i)
        if tok is None or len(tok) == 0 or tok[0] != "Ġ":
            continue
        word = tok[1:]
        if word in sing_set:
            sing_ids.append(i)
        elif word in plur_set:
            sing_form = sing_list[plur_list.index(word)]
            sing_enc  = tokenizer(f" {sing_form}", add_special_tokens=False)["input_ids"]
            if not strict or (
                len(sing_enc) == 1 and sing_enc[0] != tokenizer.unk_token_id
            ):
                plur_ids.append(i)

    return (torch.tensor(sing_ids, dtype=torch.long),
            torch.tensor(plur_ids, dtype=torch.long))


def build_sva_is_are_ids(
    tokenizer: PreTrainedTokenizer,
) -> Tuple[int, int]:
    """
    Return (is_id, are_id) for the two-token SVA logit-diff variant.

    Both " is" and " are" must tokenize to a single token under the
    target tokenizer (true for gpt2 / pythia BPE). Caller can use these
    ids with `make_sva_is_are_logit_diff_loss_fn`.
    """
    is_enc  = tokenizer(" is",  add_special_tokens=False)["input_ids"]
    are_enc = tokenizer(" are", add_special_tokens=False)["input_ids"]
    if len(is_enc) != 1 or len(are_enc) != 1:
        raise ValueError(
            f"' is' / ' are' must tokenize to single tokens; "
            f"got is={is_enc}, are={are_enc}"
        )
    return int(is_enc[0]), int(are_enc[0])


def make_sva_is_are_logit_diff_loss_fn(
    is_id:  int,
    are_id: int,
) -> Callable:
    """
    Returns loss_fn(outputs, batch, extras) -> scalar tensor for SVA,
    using only the two tokens ' is' and ' are'.

    Loss = -(logit[target] - logit[other])  on the prediction position
    (last input token before completion).

    target = is_id  if extras['plural'] == 0  (singular subject)
           = are_id if extras['plural'] == 1  (plural subject)

    Equivalent to a 2-class log-prob diff: log_softmax shifts cancel,
    so logit[is]−logit[are] = log_p(is)−log_p(are). We use raw logits
    to skip the full-vocab softmax pass.

    Difference vs `make_sva_prob_diff_loss_fn`:
      - that variant aggregates all single-token sing/plur verbs via
        logsumexp (matches eap-ig-faithfulness verb_list_csv metric);
      - this variant collapses the target/other sets to a single pair
        of tokens, giving a sharper but narrower signal; useful when
        the SVD spectrum on the verb-set variant collapses to one
        dominant direction (close to the magnitude baseline).

    `extras["plural"]` is 0/1 for the current instance (shared across
    paraphrases of one instance).
    """
    def loss_fn(outputs, batch: dict, extras: Optional[dict]) -> torch.Tensor:
        if extras is None or "plural" not in extras:
            raise ValueError("sva_is_are_logit_diff_loss requires extras['plural']")
        p = int(extras["plural"])

        labels = batch["labels"][0]
        nonignored = (labels != -100).nonzero(as_tuple=True)[0]
        if nonignored.numel() == 0:
            raise ValueError("sva_is_are_logit_diff_loss: no completion tokens in labels")
        pred_pos = int(nonignored[0].item()) - 1
        last_logits = outputs.logits[0, pred_pos, :]

        is_l  = last_logits[is_id]
        are_l = last_logits[are_id]
        diff  = (is_l - are_l) if p == 0 else (are_l - is_l)
        return -diff

    return loss_fn


def make_sva_prob_diff_loss_fn(
    singular_ids: torch.Tensor,
    plural_ids:   torch.Tensor,
) -> Callable:
    """
    Returns loss_fn(outputs, batch, extras) -> scalar tensor for SVA.

    Loss = -(logsumexp p(target_set) - logsumexp p(other_set))     on the
    clean prompt's prediction position (last input token).

    target_set = singular_ids  if extras['plural'] == 0
               = plural_ids    if extras['plural'] == 1

    Same direction as the SVA prob_diff metric (label==0: singular minus plural).

    `extras["plural"]` is 0/1 for the current instance (shared across paraphrases).
    """
    sing_local = singular_ids
    plur_local = plural_ids

    def loss_fn(outputs, batch: dict, extras: Optional[dict]) -> torch.Tensor:
        if extras is None or "plural" not in extras:
            raise ValueError("sva_prob_diff_loss requires extras['plural']")
        p = int(extras["plural"])

        labels = batch["labels"][0]
        nonignored = (labels != -100).nonzero(as_tuple=True)[0]
        if nonignored.numel() == 0:
            raise ValueError("sva_prob_diff_loss: no completion tokens in labels")
        pred_pos = int(nonignored[0].item()) - 1
        last_logits = outputs.logits[0, pred_pos, :]
        log_probs = F.log_softmax(last_logits, dim=-1)

        sing_lp = torch.logsumexp(log_probs[sing_local.to(log_probs.device)], dim=0)
        plur_lp = torch.logsumexp(log_probs[plur_local.to(log_probs.device)], dim=0)
        diff = (sing_lp - plur_lp) if p == 0 else (plur_lp - sing_lp)
        return -diff

    return loss_fn


def load_records_country_cap(
    path: str,
    source: str = "paraphrases",   # ignored; kept for API consistency
    N: Optional[int] = None,
    M: Optional[int] = None,
) -> List[Dict]:
    """
    Load country_cap_*_paraphrases.json (country-capital strong-paraphrase schema).

    JSON structure (top-level list):
      [
        {
          "clean":     "<prompt str ending in 'the capital of', no trailing space>",
          "corrupted": "<corrupted prompt>",
          "country":   "Turkmenistan",
          "capital":   "Ashgabat",
          "subject":   "Ashgabat",
          "country_idx": <int>,
          "extended":  {"clean": "...", "corrupted": "..."},
          "strong_semantic_structural_paraphrases": [
            {"clean": "<prompt>", "corrupted": "<prompt>", "meta": {...}},
            ...
          ]
        },
        ...
      ]

    The completion is `entry["country"]` (shared across all paraphrases of an
    entry). A leading space is prepended so make_batch's query+completion join
    produces correct " Turkmenistan" tokenisation (prompts have no trailing
    space).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else raw.get("data", raw)

    if N is not None:
        data = data[:N]

    records = []
    for idx, entry in enumerate(data):
        original_query      = entry["clean"]
        country             = entry["country"].strip()
        original_completion = " " + country
        subject             = entry.get("subject", "").strip() or None

        paraphrases = entry.get("strong_semantic_structural_paraphrases", [])
        if M is not None:
            paraphrases = paraphrases[:M]
        queries     = [p["clean"] for p in paraphrases]
        completions = [original_completion] * len(paraphrases)

        if not queries:
            continue

        records.append({
            "instance_id":            idx,
            "original_query":         original_query,
            "original_completion":    original_completion,
            "paraphrase_queries":     queries,
            "paraphrase_completions": completions,
            "io_name":                country,
            "subject_name":           subject,
        })

    M_values = [len(r["paraphrase_queries"]) for r in records]
    assert len(set(M_values)) == 1, \
        f"Inconsistent M across instances: {set(M_values)}"

    print(f"[Loader/country_cap] N={len(records)} instances, "
          f"M={M_values[0]} paraphrases/instance")
    return records


# Tokenisation

def make_batch(
    query: str,
    completion: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 128,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a (query, completion) pair into a model-ready batch.

    Labels are -100 on the query (ignored by CE) and token ids on the
    completion. The completion start is the first token whose offset end
    exceeds len(query), found via offset_mapping. This works whether a
    trailing space in the query is its own token (its end == len(query)) or
    is merged into the first completion token.
    """
    full_text = query + completion
    enc = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
    full_ids = enc["input_ids"]
    offsets  = enc["offset_mapping"]   # List[(start_char, end_char)]

    char_boundary = len(query)
    comp_start = next(
        (i for i, (_, e) in enumerate(offsets) if e > char_boundary),
        len(full_ids),   # fallback; should not happen
    )

    # Truncate from the left of the query, keeping the completion.
    if len(full_ids) > max_seq_len:
        c_len      = len(full_ids) - comp_start
        keep_q     = max(1, max_seq_len - c_len)
        trim       = comp_start - keep_q
        full_ids   = full_ids[trim:]
        comp_start = keep_q

    input_ids      = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels         = input_ids.clone()
    labels[0, :comp_start] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }
