"""Head-level activation patching (Wang et al. 2022 / EAP-IG protocol).

Cache each head's z (blocks.{l}.attn.hook_z, before W_O) on the corrupted
prompt, run the clean prompt with the selected heads' z replaced by the
corrupted values, and read the task metric at the last input position.
patch_heads([]) is the clean baseline; patching all heads is close to, but not
identical to, the corrupted run (embeddings and MLPs still see clean input).
"""
from __future__ import annotations
from typing import Iterable, List, Tuple

import torch
from transformer_lens import HookedTransformer


def tokenize_pair(model: HookedTransformer, clean: List[str], corrupted: List[str]):
    """Joint tokenize clean+corrupted to get equal padded length n_pos."""
    assert len(clean) == len(corrupted)
    model.tokenizer.padding_side = "right"
    tokens = model.to_tokens(list(clean) + list(corrupted), prepend_bos=True)
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    mask = (tokens != pad_id).long()
    mask[:, 0] = 1   # BOS == eos for gpt2; force pos 0 unmasked
    B = len(clean)
    return tokens[:B], tokens[B:], mask[:B].sum(1), tokens.size(1)


def cache_corrupted_z(model: HookedTransformer,
                      corrupted_tokens: torch.Tensor) -> dict:
    """Run corrupted; return dict {layer: (B, pos, H, d_head) tensor}."""
    z_filter = lambda name: name.endswith(".attn.hook_z")
    with torch.no_grad():
        _, cache = model.run_with_cache(
            corrupted_tokens,
            names_filter=z_filter,
            return_type=None,
        )
    out = {}
    for l in range(model.cfg.n_layers):
        out[l] = cache[f"blocks.{l}.attn.hook_z"].detach()
    return out


def run_clean_with_patches(
    model:           HookedTransformer,
    clean_tokens:    torch.Tensor,
    corrupted_z:     dict,
    heads_to_patch:  Iterable[Tuple[int, int]],
) -> torch.Tensor:
    """Run clean with the specified (layer, head) z-slices replaced by
    corresponding corrupted z slices. Returns full logits (B, pos, vocab)."""
    by_layer: dict = {}
    for (l, h) in heads_to_patch:
        by_layer.setdefault(l, []).append(h)

    fwd_hooks = []
    for l, heads in by_layer.items():
        heads_t = torch.tensor(sorted(set(heads)), dtype=torch.long,
                               device=corrupted_z[l].device)
        cz = corrupted_z[l]   # (B, pos, H, d_head)
        def make_hook(cz_tensor, head_idx):
            def hook(z, hook):
                # z: (B, pos, H, d_head); replace specified head dim
                z[:, :, head_idx, :] = cz_tensor[:, :, head_idx, :]
                return z
            return hook
        fwd_hooks.append((f"blocks.{l}.attn.hook_z", make_hook(cz, heads_t)))

    with torch.no_grad():
        logits = model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks,
                                      return_type="logits")
    return logits


def logit_diff_at_last_pos(
    logits:        torch.Tensor,    # (B, pos, V)
    input_lengths: torch.Tensor,    # (B,)
    labels:        torch.Tensor,    # (B, 2)  [correct, incorrect]
) -> torch.Tensor:
    """Mean of (logit_correct - logit_incorrect) read at position
    input_length-1 (last input token, model predicts next)."""
    B = logits.size(0)
    last_pos_logits = logits[torch.arange(B, device=logits.device),
                             input_lengths.to(logits.device) - 1]    # (B, V)
    picked = torch.gather(last_pos_logits, -1, labels.to(logits.device))  # (B, 2)
    return (picked[:, 0] - picked[:, 1]).mean()


def get_year_token_ids(tokenizer) -> torch.Tensor:
    """Token id for each 2-digit year string '00'..'99' (same as the EAP-IG
    greater-than metric). Takes input_ids[0]; the YY strings are single tokens
    in the GPT-2 / Pythia BPE."""
    return torch.tensor(
        [tokenizer(f"{yy:02d}").input_ids[0] for yy in range(100)],
        dtype=torch.long,
    )


def prob_diff_greater_than_at_last_pos(
    logits:        torch.Tensor,    # (B, pos, V)
    input_lengths: torch.Tensor,    # (B,)
    labels:        torch.Tensor,    # (B,)  YY₀ ∈ [0, 99]
    year_token_ids: torch.Tensor,   # (100,) token id for each YY
) -> torch.Tensor:
    """Greater-than prob_diff metric (Hanna et al. 2023):
    at the last input position,
        Σ p(YY) for YY > YY₀  −  Σ p(YY) for YY ≤ YY₀
    where probabilities are softmax over the full vocab and the sum is
    over the 100 year-token ids.  Returns the batch mean."""
    B = logits.size(0)
    dev = logits.device
    last_pos_logits = logits[torch.arange(B, device=dev),
                             input_lengths.to(dev) - 1]                # (B, V)
    probs = torch.softmax(last_pos_logits, dim=-1)
    year_probs = probs[:, year_token_ids.to(dev)]                       # (B, 100)
    yy0 = labels.to(dev)                                                # (B,)
    results = []
    for i in range(B):
        cut = int(yy0[i].item()) + 1
        results.append(year_probs[i, cut:].sum() - year_probs[i, :cut].sum())
    return torch.stack(results).mean()


def prob_diff_sva_at_last_pos(
    logits:        torch.Tensor,    # (B, pos, V)
    input_lengths: torch.Tensor,    # (B,)
    labels:        torch.Tensor,    # (B,)  plural ∈ {0, 1}
    singular_ids:  torch.Tensor,    # (S,)  vocab ids for single-token singular verbs
    plural_ids:    torch.Tensor,    # (P,)  vocab ids for single-token plural verbs
) -> torch.Tensor:
    """SVA prob_diff metric (Marvin & Linzen 2018):
    at the last input position, compute
        where(plural==0, Σp(sing) − Σp(plur), Σp(plur) − Σp(sing))
    where probabilities are softmax over the full vocab and the sums are
    over tokenizer-derived singular/plural verb-token sets. Returns batch mean."""
    B = logits.size(0)
    dev = logits.device
    last_pos_logits = logits[torch.arange(B, device=dev),
                             input_lengths.to(dev) - 1]                # (B, V)
    probs = torch.softmax(last_pos_logits, dim=-1)
    sing = probs[:, singular_ids.to(dev)].sum(-1)                       # (B,)
    plur = probs[:, plural_ids.to(dev)].sum(-1)                         # (B,)
    lab = labels.to(dev)
    return torch.where(lab == 0, sing - plur, plur - sing).mean()


def build_sva_verb_token_ids(
    tokenizer, verb_list_csv, strict: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy of data.loader.build_sva_verb_ids, kept here to avoid importing
    data.loader. Keep the two in sync."""
    import pandas as pd

    df = pd.read_csv(verb_list_csv)
    sing_list = df["sing"].tolist()
    plur_list = df["plur"].tolist()
    sing_set  = set(sing_list)
    plur_set  = set(plur_list)

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


def make_metric_fn(task: str, model: HookedTransformer,
                   input_lengths: torch.Tensor, labels: torch.Tensor,
                   *, verb_list_csv=None, sva_metric_variant: str = "verb_set"):
    """Build a closure metric_fn(logits) -> scalar tensor, dispatched by task.

    The returned callable carries:
      .name  : short metric name ('logit_diff' / 'prob_diff')
      .label : human-readable y-axis label for plots

    For task == "sva", two variants:
      - 'verb_set' (default): aggregate over single-token sing/plur verbs from
        `verb_list_csv` (path to combined_verb_list.csv); same as the
        EAP-IG prob_diff_sva metric.
      - 'is_are': restrict target/other to (' is', ' are'); metric becomes a
        2-token logit_diff at the last input position. Matches the
        sva is_are loss in data/loader.py.
    """
    if task == "greater_than":
        year_ids = get_year_token_ids(model.tokenizer).to(input_lengths.device)

        def metric(logits):
            return prob_diff_greater_than_at_last_pos(
                logits, input_lengths, labels, year_ids,
            )
        metric.name = "prob_diff"
        metric.label = "prob_diff = Σp(YY>YY₀) − Σp(YY≤YY₀)"
        return metric

    if task == "sva":
        if sva_metric_variant == "is_are":
            # 2-token logit_diff at last input position, target chosen by plural label.
            is_enc  = model.tokenizer(" is",  add_special_tokens=False)["input_ids"]
            are_enc = model.tokenizer(" are", add_special_tokens=False)["input_ids"]
            if len(is_enc) != 1 or len(are_enc) != 1:
                raise ValueError(
                    f"sva is_are metric: ' is'/' are' must tokenize to single tokens; "
                    f"got is={is_enc}, are={are_enc}"
                )
            is_id  = int(is_enc[0])
            are_id = int(are_enc[0])
            dev    = input_lengths.device

            def metric(logits):
                B = logits.size(0)
                last_pos_logits = logits[
                    torch.arange(B, device=logits.device),
                    input_lengths.to(logits.device) - 1,
                ]                                                               # (B, V)
                is_l  = last_pos_logits[:, is_id]
                are_l = last_pos_logits[:, are_id]
                lab   = labels.to(logits.device)
                return torch.where(lab == 0, is_l - are_l, are_l - is_l).mean()

            metric.name = "logit_diff"
            metric.label = "logit_diff = logit(' is'|' are' target) − logit(other)"
            return metric

        if sva_metric_variant != "verb_set":
            raise ValueError(
                f"sva_metric_variant={sva_metric_variant!r} not recognized; "
                "expected 'verb_set' or 'is_are'"
            )

        if verb_list_csv is None:
            raise ValueError(
                "make_metric_fn(task='sva', sva_metric_variant='verb_set') "
                "requires verb_list_csv (path to combined_verb_list.csv)"
            )
        sing_ids, plur_ids = build_sva_verb_token_ids(model.tokenizer, verb_list_csv)
        sing_ids = sing_ids.to(input_lengths.device)
        plur_ids = plur_ids.to(input_lengths.device)

        def metric(logits):
            return prob_diff_sva_at_last_pos(
                logits, input_lengths, labels, sing_ids, plur_ids,
            )
        metric.name = "prob_diff"
        metric.label = (f"prob_diff = Σp(target_verbs) − Σp(other_verbs)  "
                        f"(|S|={sing_ids.numel()}, |P|={plur_ids.numel()})")
        return metric

    # Default: two-token logit diff (IOI / gender-bias / fact-retrieval-comma)
    def metric(logits):
        return logit_diff_at_last_pos(logits, input_lengths, labels)
    metric.name = "logit_diff"
    metric.label = "logit_diff = logit(correct) − logit(incorrect)"
    return metric


def all_heads(n_layers: int, n_heads: int) -> List[Tuple[int, int]]:
    return [(l, h) for l in range(n_layers) for h in range(n_heads)]
