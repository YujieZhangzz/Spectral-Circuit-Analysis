"""
Weight-knockout check of each block's subspace U_rho: project the block's
weights onto the orthogonal complement of a basis and measure the drop in
clean completion log-prob, mean_i log P(completion_token_0 | query_i).

Bases compared: U_rho (d_k × ρ), a random orthonormal basis of the same rank
(averaged over n_random draws), and the top-ρ SC one-hot directions.

Usage:
  python spectral/verify_knockout.py --config configs/gpt2_small_ioi_strong.yaml [--blocks 3 4] [--n_random 10]
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

# make the repo root importable regardless of the working directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.io_utils    import load_config, get_output_dir, PhaseCache
from utils.model_utils import load_model_and_tokenizer, get_param_filter
from data.loader       import load_records
from spectral.block_finder import detect_arch, build_unit_layout


# Evaluation (clean inputs only)

def compute_completion_logprob(
    model,
    tokenizer,
    records:  List[dict],
    device:   str,
    max_len:  int = 128,
) -> float:
    """
    Mean completion log-prob over records:

    score = mean_i log P(completion_token_0 | paraphrase_queries[0]_i)

    Uses clean inputs only; the drop after knockout is the importance proxy.
    """
    model.eval()
    logprobs = []

    for rec in records:
        query      = rec["paraphrase_queries"][0]
        completion = rec["paraphrase_completions"][0]

        tokens_q = tokenizer.encode(query,      add_special_tokens=False)
        tokens_c = tokenizer.encode(completion, add_special_tokens=False)

        if not tokens_c:
            continue

        full_tokens = torch.tensor(
            [tokens_q + tokens_c], device=device
        )
        if full_tokens.shape[1] > max_len:
            full_tokens = full_tokens[:, -max_len:]

        with torch.no_grad():
            logits = model(full_tokens).logits[0]   # (seq, vocab)

        # log prob of the first completion token
        pos    = len(tokens_q) - 1
        target = tokens_c[0]
        lp     = F.log_softmax(logits[pos], dim=-1)[target].item()
        logprobs.append(lp)

    return float(np.mean(logprobs)) if logprobs else 0.0


# Map flat gradient indices to model parameters

def build_param_seg_map(
    model,
    param_filter: Callable[[str], bool],
) -> List[Tuple[str, int, int]]:
    """
    Return [(param_name, flat_start, flat_end), ...] over filtered parameters,
    in the same order as obtain_gradients.
    """
    segs   = []
    offset = 0
    for name, param in model.named_parameters():
        if not param_filter(name):
            continue
        n = param.numel()
        segs.append((name, offset, offset + n))
        offset += n
    return segs


# Knockout helpers

def apply_knockout(
    model,
    param_segs:    List[Tuple[str, int, int]],
    block_indices: torch.Tensor,   # (d_k,) flat indices
    U:             torch.Tensor,   # (d_k, r) orthonormal basis
) -> None:
    """
    In place: project the parameters at block_indices onto the orthogonal
    complement of span(U).

    w_knocked = w - U @ (U.T @ w)
    """
    # For each parameter, find which block_indices fall inside it
    seg_dict = {name: (start, end)
                for name, start, end in param_segs}

    param_groups: Dict[str, List[int]] = {}   # param_name -> local indices in d_k
    global_to_local = {}                      # flat_idx -> position in block_indices
    for local_pos, flat_idx in enumerate(block_indices.tolist()):
        global_to_local[flat_idx] = local_pos

    for name, flat_start, flat_end in param_segs:
        local_positions = []
        param_positions = []
        for local_pos, flat_idx in enumerate(block_indices.tolist()):
            if flat_start <= flat_idx < flat_end:
                local_positions.append(local_pos)
                param_positions.append(flat_idx - flat_start)

        if local_positions:
            param_groups[name] = (
                torch.tensor(local_positions, dtype=torch.long),
                torch.tensor(param_positions, dtype=torch.long),
            )

    # gather block weights, project, write back
    param_dict = dict(model.named_parameters())
    device = next(model.parameters()).device

    for name, (local_pos, param_pos) in param_groups.items():
        param = param_dict[name]
        w_flat = param.data.view(-1)

        # block weights on the model device
        w_block  = w_flat[param_pos].float()
        U_local  = U[local_pos, :].float().to(device)
        w_block  = w_block.to(device)

        # projection: w - U_local @ U_local.T @ w_block
        proj  = U_local @ (U_local.T @ w_block)
        w_new = w_block - proj

        # write back in the original dtype
        w_flat[param_pos] = w_new.to(param.dtype)


def restore_params(model, original_params: Dict[str, torch.Tensor]) -> None:
    """Restore the parameters saved before knockout."""
    for name, param in model.named_parameters():
        if name in original_params:
            param.data.copy_(original_params[name])


def random_orthonormal(d_k: int, r: int) -> torch.Tensor:
    """Random (d_k, r) orthonormal basis."""
    A = torch.randn(d_k, r)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :r]   # (d_k, r)


def sc_topk_basis(sc: torch.Tensor, block_indices: torch.Tensor, r: int) -> torch.Tensor:
    """
    Basis of the r one-hot directions with the highest SC in the block.
    sc: (D,) global SC vector
    block_indices: (d_k,) flat indices of the block
    Returns: (d_k, r) orthonormal basis (one-hot columns)
    """
    sc_block = sc[block_indices]          # (d_k,)
    topk     = sc_block.argsort(descending=True)[:r]
    U        = torch.zeros(len(block_indices), r)
    for col, row in enumerate(topk):
        U[row, col] = 1.0
    return U


# Per-block verification

def run_knockout_verification(
    model,
    tokenizer,
    records:       List[dict],
    param_segs:    List[Tuple[str, int, int]],
    block_id:      int,
    block_indices: torch.Tensor,     # (d_k,)
    U_rho:         torch.Tensor,     # (d_k, ρ)
    sc:            torch.Tensor,     # (D,) full SC vector
    device:        str,
    n_random:      int = 5,
    max_len:       int = 128,
    eval_fn=None,                    # None -> compute_completion_logprob
) -> Dict[str, float]:
    """
    Run the three knockout conditions on one block and return the scores.
    """
    if eval_fn is None:
        eval_fn = compute_completion_logprob

    d_k = len(block_indices)
    rho = U_rho.shape[1]

    print(f"\n[Knockout] Block {block_id}  d_k={d_k:,}  ρ={rho}")

    # save original parameters
    original = {name: param.data.clone()
                for name, param in model.named_parameters()
                if any(name == s[0] for s in param_segs)}

    # Baseline: no knockout
    score_clean = eval_fn(model, tokenizer, records, device, max_len)
    print(f"  clean         logit_diff = {score_clean:.4f}")

    # Condition 1: U_rho knockout
    apply_knockout(model, param_segs, block_indices, U_rho)
    score_urho = eval_fn(model, tokenizer, records, device, max_len)
    restore_params(model, original)
    drop_urho = score_clean - score_urho
    print(f"  U_rho         logit_diff = {score_urho:.4f}  "
          f"drop = {drop_urho:+.4f}")

    # Condition 2: random basis (mean over n_random draws)
    random_scores = []
    for t in range(n_random):
        U_rand = random_orthonormal(d_k, rho).to(U_rho.device)
        apply_knockout(model, param_segs, block_indices, U_rand)
        s = eval_fn(model, tokenizer, records, device, max_len)
        restore_params(model, original)
        random_scores.append(s)

    score_random = float(np.mean(random_scores))
    drop_random  = score_clean - score_random
    print(f"  random(n={n_random})    logit_diff = {score_random:.4f}  "
          f"drop = {drop_random:+.4f}  ±{np.std(random_scores):.4f}")

    # Condition 3: SC top-ρ one-hot basis
    U_sc = sc_topk_basis(sc, block_indices, rho)
    apply_knockout(model, param_segs, block_indices, U_sc)
    score_sc = eval_fn(model, tokenizer, records, device, max_len)
    restore_params(model, original)
    drop_sc = score_clean - score_sc
    print(f"  sc_topk       logit_diff = {score_sc:.4f}  "
          f"drop = {drop_sc:+.4f}")

    # Importance ratio
    if abs(drop_random) > 1e-6:
        ratio = drop_urho / drop_random
        print(f"  importance ratio (U_rho / random) = {ratio:.2f}x")
    else:
        ratio = float("nan")
        print(f"  importance ratio = nan (random drop≈0)")

    return {
        "block_id":    block_id,
        "d_k":         d_k,
        "rho":         rho,
        "score_clean": score_clean,
        "score_urho":  score_urho,
        "score_random": score_random,
        "score_sc":    score_sc,
        "drop_urho":   drop_urho,
        "drop_random": drop_random,
        "drop_sc":     drop_sc,
        "random_std":  float(np.std(random_scores)),
        "importance_ratio": ratio,
    }


# Plotting

def plot_knockout_results(
    results:  List[Dict],
    save_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(results)
    if n == 0:
        return

    labels    = [f"B{r['block_id']}\n(ρ={r['rho']})" for r in results]
    drop_urho = [r["drop_urho"]   for r in results]
    drop_rand = [r["drop_random"] for r in results]
    drop_sc   = [r["drop_sc"]     for r in results]
    rand_std  = [r["random_std"]  for r in results]

    x     = np.arange(n)
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n * 1.5), 5))

    # Left: drop per condition
    ax1.bar(x - width, drop_urho, width, label="U_rho",
            color="#4f86c6", alpha=0.85)
    ax1.bar(x,         drop_rand, width, label="random",
            color="#aaaaaa", alpha=0.85, yerr=rand_std, capsize=3)
    ax1.bar(x + width, drop_sc,   width, label="SC-top-ρ",
            color="#e05c5c", alpha=0.85)

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Logit diff drop (clean - knocked)")
    ax1.set_title("Knockout effect by condition\n(higher = more important)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Right: importance ratio
    ratios = [r["importance_ratio"] for r in results
              if not np.isnan(r["importance_ratio"])]
    valid_labels = [f"B{r['block_id']}" for r in results
                    if not np.isnan(r["importance_ratio"])]

    if ratios:
        colors = ["#4f86c6" if r > 1 else "#e05c5c" for r in ratios]
        ax2.bar(range(len(ratios)), ratios, color=colors, alpha=0.85)
        ax2.axhline(1.0, color="gray", linestyle="--",
                    label="random baseline (1.0×)")
        ax2.set_xticks(range(len(ratios)))
        ax2.set_xticklabels(valid_labels, fontsize=8)
        ax2.set_ylabel("Importance ratio (U_rho drop / random drop)")
        ax2.set_title("U_rho importance vs random\n(>1 = better than random)")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "knockout_verification.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] {out}")


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True)
    p.add_argument("--blocks",   nargs="*", type=int, default=None,
                   help="Block ids to verify (default: all)")
    p.add_argument("--n_random", type=int, default=5,
                   help="Number of random-basis draws")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    outdir = get_output_dir(cfg)
    cache  = PhaseCache(outdir)

    # knockout settings (CLI overrides config)
    ko_cfg            = cfg.get("knockout", {})
    n_random          = args.n_random if args.n_random != 5 else ko_cfg.get("n_random", 5)
    importance_thresh = ko_cfg.get("importance_threshold", 1.5)
    cfg_blocks        = ko_cfg.get("blocks", None)   # null = all

    print(f"\n{'='*60}")
    print(f"  Weight Knockout Verification")
    print(f"  config: {args.config}")
    print(f"  n_random={n_random}  eval=completion_logprob  "
          f"importance_threshold={importance_thresh}")
    print(f"{'='*60}\n")

    # Model
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name_or_path"],
        cfg["model"]["torch_dtype"],
        cfg["model"]["device"],
    )
    device  = cfg["model"]["device"]
    arch    = detect_arch(model)
    n_layers = getattr(model.config, "num_hidden_layers",
                       getattr(model.config, "n_layer", None))

    scope        = cfg["params"]["scope"]
    include_mlp  = cfg["params"].get("include_mlp", True)
    param_filter = get_param_filter(scope, n_layers)

    # Data
    records = load_records(
        cfg["data"]["path"],
        cfg["data"]["source"],
        cfg["data"]["N"],
    )
    max_len = cfg["data"]["max_seq_len"]

    # Parameter segment map
    param_segs = build_param_seg_map(model, param_filter)
    print(f"[Setup] {len(param_segs)} filtered param segments")

    # Blocks
    blocks_data = [b for b in cache.load_json("blocks")
                   if "unit_ids" in b]
    print(f"[Setup] {len(blocks_data)} blocks found")

    # SC
    sc = cache.load_tensor("sc")   # (D,) float32

    # blocks to verify: CLI --blocks > config knockout.blocks > all
    block_ids = (args.blocks or cfg_blocks) if (args.blocks or cfg_blocks) \
                else list(range(len(blocks_data)))
    print(f"[Setup] Verifying blocks: {block_ids}")

    # Rebuild unit_layout
    unit_layout = build_unit_layout(
        model, param_filter, arch, include_mlp=include_mlp
    )

    # Evaluation function (clean inputs only)
    eval_fn = compute_completion_logprob
    print(f"[Setup] eval_fn={eval_fn.__name__}")

    # Knockout per block
    all_results = []

    for block_id in block_ids:
        if block_id >= len(blocks_data):
            print(f"[WARN] Block {block_id} not found, skipping")
            continue

        b = blocks_data[block_id]

        # U_rho must be cached
        if not cache.exists(f"U_rho_block{block_id}"):
            print(f"[WARN] U_rho_block{block_id} not cached, "
                  f"run full pipeline first")
            continue

        U_rho = cache.load_tensor(f"U_rho_block{block_id}")  # (d_k, ρ)

        # rebuild block_indices
        unit_ids     = b["unit_ids"]
        block_indices = torch.cat([
            unit_layout[uid].get_indices() for uid in unit_ids
        ])

        result = run_knockout_verification(
            model        = model,
            tokenizer    = tokenizer,
            records      = records,
            param_segs   = param_segs,
            block_id     = block_id,
            block_indices= block_indices,
            U_rho        = U_rho,
            sc           = sc,
            device       = device,
            n_random     = n_random,
            max_len      = max_len,
            eval_fn      = eval_fn,
        )
        all_results.append(result)

    # Save results
    save_dir = str(outdir / "plots")
    cache.save_json(all_results, "knockout_results")
    plot_knockout_results(all_results, save_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Knockout Verification Summary")
    print(f"{'='*60}")
    print(f"  {'Block':>6}  {'ρ':>4}  {'drop_Urho':>10}  "
          f"{'drop_rand':>10}  {'ratio':>7}")
    print(f"  {'-'*50}")
    for r in all_results:
        print(f"  B{r['block_id']:>5}  {r['rho']:>4}  "
              f"{r['drop_urho']:>+10.4f}  "
              f"{r['drop_random']:>+10.4f}  "
              f"{r['importance_ratio']:>7.2f}x")

    # blocks above knockout.importance_threshold
    important = [r for r in all_results
                 if not np.isnan(r["importance_ratio"])
                 and r["importance_ratio"] > importance_thresh]
    print(f"\n  Blocks with ratio > {importance_thresh}× random: "
          f"{[r['block_id'] for r in important]}")


if __name__ == "__main__":
    main()