"""z-subspace activation patching sweep. For each of the top-K heads of a
ranking, replace only the component of z in the head's U_z subspace:
    z_new = (I - P_h) z_clean + P_h z_corrupted,   P_h = U_z U_z^T
U_z comes from circuits_discovery.faithfulness.subspace_z.

Usage:
    python -m circuits_discovery.faithfulness.sweep_subspace_z --device cuda:0
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from circuits_discovery.faithfulness.data     import load_ioi
from circuits_discovery.faithfulness.patching import (
    tokenize_pair, cache_corrupted_z, make_metric_fn,
)


def parse_lh(label: str) -> Tuple[int, int]:
    l, h = label[1:].split("H")
    return int(l), int(h)


def load_attn_ranking(csv_path: Path) -> List[Tuple[int, int]]:
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("unit_type") != "attn_head":
                continue
            rows.append((int(row["rank"]), parse_lh(row["label"])))
    rows.sort()
    return [lh for _, lh in rows]


def run_clean_with_subspace_patches(
    model:           HookedTransformer,
    clean_tokens:    torch.Tensor,
    corrupted_z:     dict,                  # {l: (B, pos, H, d_head)}
    z_subspaces:     dict,                  # {(l,h): tensor (d_head, rho)}
    heads_to_patch:  List[Tuple[int, int]],
) -> torch.Tensor:
    """At each (l, h) in heads_to_patch, replace clean z's projection on
    U_z[(l,h)] with corrupted z's projection on the same subspace; leave
    the orthogonal complement of clean z unchanged. Returns full logits."""
    by_layer: Dict[int, List[int]] = {}
    for (l, h) in heads_to_patch:
        by_layer.setdefault(l, []).append(h)

    fwd_hooks = []
    for l, heads in by_layer.items():
        cz = corrupted_z[l]   # (B, pos, H, d_head)
        # Per-head projectors P_l ∈ R^(H, d_head, d_head); only heads in
        # head_set are touched by the hook.
        H, d_head = cz.shape[2], cz.shape[3]
        P_l = torch.zeros(H, d_head, d_head, device=cz.device, dtype=cz.dtype)
        for h in heads:
            U = z_subspaces[(l, h)].to(cz.device, dtype=cz.dtype)   # (d_head, rho)
            P_l[h] = U @ U.T
        head_set = set(heads)

        def make_hook(P_l, cz, head_set):
            def hook(z, hook):
                # z: (B, pos, H, d_head)
                for h in head_set:
                    Ph = P_l[h]                       # (d_head, d_head)
                    # split z = (I-P) z + P z   ;  replace P z with P z_corrupted
                    zh = z[:, :, h, :]                # (B, pos, d_head)
                    proj_clean = zh @ Ph              # (B, pos, d_head)
                    proj_corr  = cz[:, :, h, :] @ Ph
                    z[:, :, h, :] = zh - proj_clean + proj_corr
                return z
            return hook
        fwd_hooks.append((f"blocks.{l}.attn.hook_z", make_hook(P_l, cz, head_set)))

    with torch.no_grad():
        logits = model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks,
                                      return_type="logits")
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="prompts dataset csv (clean+corrupted pairs)")
    ap.add_argument("--n",      type=int, default=100)
    ap.add_argument("--start",  type=int, default=0,
                    help="Skip first --start rows of csv before taking --n.")
    ap.add_argument("--model",  default="gpt2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ranking_csv", type=Path, required=True,
                    help="head ranking csv for one method (matches --tag)")
    ap.add_argument("--z_subspaces", type=Path, required=True,
                    help="z_subspaces.pt from subspace_z.py (per-head SVD subspace U_z)")
    ap.add_argument("--k_max",  type=int, default=144)
    ap.add_argument("--k_step", type=int, default=1)
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="output dir; writes subspace_z_<tag>.json")
    ap.add_argument("--tag", required=True,
                    help="output filename suffix: subspace_z_<tag>.json "
                         "(e.g. svd / eap / eap_ig / eap_ig_kl)")
    ap.add_argument("--gt_mode", default="prob_diff",
                    choices=["prob_diff", "boundary_logit_diff"],
                    help="Greater-than metric: prob_diff (default) or "
                         "boundary_logit_diff (= logit(YY₀+1) - logit(YY₀))")
    ap.add_argument("--verb_list_csv", type=Path, default=Path(
        "eap_data/sva/combined_verb_list.csv"),
        help="Required for task='sva' verb_set variant: sing/plur verb pair csv")
    ap.add_argument("--sva_metric_variant", default="verb_set",
                    choices=["verb_set", "is_are"],
                    help="SVA metric: 'verb_set' (default) or 'is_are' "
                         "(logit_diff on ' is'/' are')")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading {args.model} on {args.device}")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head

    print(f"[2/5] Loading prompt pairs and z-space subspaces "
          f"(start={args.start}, slice=[{args.start}, {args.start + args.n}))")
    batch = load_ioi(args.csv, n=args.n, start=args.start,
                     gt_mode=args.gt_mode, tokenizer=model.tokenizer)
    print(f"      task = {batch.task!r}  (gt_mode={args.gt_mode!r})")
    clean_tokens, corrupted_tokens, input_lengths, n_pos = tokenize_pair(
        model, batch.clean, batch.corrupted,
    )
    clean_tokens     = clean_tokens.to(args.device)
    corrupted_tokens = corrupted_tokens.to(args.device)
    input_lengths    = input_lengths.to(args.device)
    labels           = batch.labels.to(args.device)
    metric_fn        = make_metric_fn(batch.task, model, input_lengths, labels,
                                       verb_list_csv=args.verb_list_csv,
                                       sva_metric_variant=args.sva_metric_variant)
    if batch.task == "sva":
        print(f"      sva_metric_variant = {args.sva_metric_variant!r}  "
              f"({metric_fn.name})")

    z_data = torch.load(args.z_subspaces, map_location="cpu", weights_only=False)
    z_subs = {lh: v["U_z"] for lh, v in z_data["subspaces"].items()}
    rhos   = {lh: v["rho"] for lh, v in z_data["subspaces"].items()}
    avg_rho = sum(rhos.values()) / len(rhos)
    print(f"      {len(z_subs)} (layer, head) subspaces, avg ρ_z = {avg_rho:.1f}")

    print(f"[3/5] Caching corrupted z (B={batch.n}, n_pos={n_pos})")
    corrupted_z = cache_corrupted_z(model, corrupted_tokens)

    # baselines
    with torch.no_grad():
        ld_clean = metric_fn(model(clean_tokens)).item()
        ld_corr  = metric_fn(model(corrupted_tokens)).item()
    print(f"      clean = {ld_clean:+.4f}   corrupted = {ld_corr:+.4f}  ({metric_fn.name})")

    print(f"[4/5] Loading SVD attn-only ranking")
    ranking = load_attn_ranking(args.ranking_csv)
    print(f"      {len(ranking)} attn heads")

    ks = list(range(0, args.k_max + 1, args.k_step))
    if 0 not in ks: ks = [0] + ks
    if args.k_max not in ks: ks.append(args.k_max)

    print(f"[5/5] Sweeping K=0..{ks[-1]} (step={args.k_step}, {len(ks)} pts)")
    curve: List[float] = []
    cum_dims: List[int] = []
    for k in tqdm(ks):
        head_set = ranking[:k]
        logits = run_clean_with_subspace_patches(
            model, clean_tokens, corrupted_z, z_subs, head_set,
        )
        m = metric_fn(logits).item()
        del logits
        cum_rho = sum(rhos[lh] for lh in head_set)
        curve.append(m); cum_dims.append(cum_rho)

    # save
    out_json = args.out_dir / f"subspace_z_{args.tag}.json"
    out_json.write_text(json.dumps({
        "ks":           ks,
        "ld_clean":     ld_clean,
        "ld_corrupted": ld_corr,
        "logit_diff":   curve,                  # metric value (key name used by the plot scripts)
        "cumulative_dims_removed": cum_dims,
        "n":            batch.n,
        "task":         batch.task,
        "metric":       metric_fn.name,
        "method":       "SVD z-subspace activation patching (per-head)",
    }, indent=2))
    print(f"      [saved] {out_json}")

    print()
    print(f"  K=0    -> {metric_fn.name} = {curve[0]:+.4f}  (should match clean)")
    for target in (5, 10, 26, 50, 100):
        if target in ks:
            i = ks.index(target)
            print(f"  K={target:<4d} -> {metric_fn.name} = {curve[i]:+.4f}  "
                  f"({cum_dims[i]} z-dims projected)")
    print(f"  K={ks[-1]:<4d} -> {metric_fn.name} = {curve[-1]:+.4f}  "
          f"({cum_dims[-1]} z-dims projected)")


if __name__ == "__main__":
    main()
