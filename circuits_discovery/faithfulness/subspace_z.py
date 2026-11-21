"""Per-head z-space SVD subspaces for subspace activation patching.

For each (layer, head), take delta_z = z_clean - z_corrupted at the last input
position (N x d_head), compute its SVD, and keep the top-ρ right singular
vectors (ρ = effective rank at η explained variance) as U_z (d_head x ρ).
Saves {eta, n_layers, n_heads, d_head, subspaces[(l, h)] = {U_z, sigma, rho,
delta_z_norm_sq}} to --out.

Usage:
    python -m circuits_discovery.faithfulness.subspace_z --device cuda:0
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from circuits_discovery.faithfulness.data     import load_ioi
from circuits_discovery.faithfulness.patching import (
    tokenize_pair, cache_corrupted_z,
)


def cache_z(model: HookedTransformer, tokens: torch.Tensor) -> dict:
    """Run model, cache z at every layer; return {l: tensor (B, pos, H, d_head)}."""
    z_filter = lambda name: name.endswith(".attn.hook_z")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=z_filter, return_type=None)
    return {l: cache[f"blocks.{l}.attn.hook_z"].detach()
            for l in range(model.cfg.n_layers)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("eap_data/ioi/gpt2.csv"))
    ap.add_argument("--n",      type=int, default=100)
    ap.add_argument("--start",  type=int, default=0,
                    help="Skip first --start rows of csv before taking --n.")
    ap.add_argument("--model",  default="gpt2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eta",    type=float, default=0.90,
                    help="Explained-variance threshold for effective rank")
    ap.add_argument("--out",    type=Path,
                    default=Path("results/"
                                 "gpt2_small_ioi_strong_N100_M5/z_subspaces.pt"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading {args.model} on {args.device}")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    print(f"      n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}")

    print(f"[2/5] Loading {args.n} IOI pairs from {args.csv} "
          f"(start={args.start}, slice=[{args.start}, {args.start + args.n}))")
    batch = load_ioi(args.csv, n=args.n, start=args.start)
    clean_tokens, corrupted_tokens, input_lengths, n_pos = tokenize_pair(
        model, batch.clean, batch.corrupted,
    )
    clean_tokens     = clean_tokens.to(args.device)
    corrupted_tokens = corrupted_tokens.to(args.device)
    input_lengths    = input_lengths.to(args.device)
    print(f"      n={batch.n}, n_pos={n_pos}")

    print(f"[3/5] Caching z (clean + corrupted) at all {n_layers} layers")
    z_clean = cache_z(model, clean_tokens)
    z_corr  = cache_z(model, corrupted_tokens)
    total_mb = sum(z.numel() * 4 for z in z_clean.values()) * 2 / 1e6
    print(f"      cached: {total_mb:.1f} MB")

    print(f"[4/5] Computing per-head SVD on delta_z at last input position "
          f"(η = {args.eta})")
    B = batch.n
    last_pos = (input_lengths - 1).long()                        # (B,)
    arange = torch.arange(B, device=args.device)
    subspaces = {}
    rhos = []
    for l in range(n_layers):
        # cache shape: (B, n_pos, H, d_head)
        zc = z_clean[l][arange, last_pos]                        # (B, H, d_head)
        zr = z_corr[l][arange, last_pos]                         # (B, H, d_head)
        delta = (zc - zr).float()                                # (B, H, d_head)
        for h in range(n_heads):
            d = delta[:, h, :]                                   # (B, d_head)
            # SVD: d = U Σ V^T;  V ∈ R^(d_head × min(B,d_head)) is right basis
            U_full, S, Vh = torch.linalg.svd(d, full_matrices=False)
            sigma = S
            sigma_sq = sigma * sigma
            cum = torch.cumsum(sigma_sq, 0) / sigma_sq.sum().clamp_min(1e-30)
            rho = int((cum < args.eta).sum().item()) + 1         # smallest k with cum >= η
            rho = max(1, min(rho, sigma.numel()))
            # right basis: V columns are in d_head space
            V = Vh.T                                              # (d_head, min(B,d_head))
            U_z = V[:, :rho].contiguous().cpu()                  # (d_head, rho)
            subspaces[(l, h)] = {
                "U_z":   U_z,
                "sigma": sigma.cpu(),
                "rho":   rho,
                "delta_z_norm_sq": float((d * d).sum().item()),
            }
            rhos.append(rho)

    print(f"      ρ distribution across {n_layers*n_heads} heads:")
    print(f"         mean = {sum(rhos)/len(rhos):.1f}, "
          f"min = {min(rhos)}, max = {max(rhos)}, "
          f"sum = {sum(rhos)}")
    # per-layer summary
    print(f"      ρ by layer:")
    for l in range(n_layers):
        layer_rhos = [subspaces[(l, h)]["rho"] for h in range(n_heads)]
        print(f"         L{l:>2}: {layer_rhos}  (sum={sum(layer_rhos)})")

    print(f"[5/5] Saving to {args.out}")
    torch.save({
        "eta":       args.eta,
        "n_layers":  n_layers,
        "n_heads":   n_heads,
        "d_head":    d_head,
        "subspaces": subspaces,
    }, args.out)
    print(f"      [saved]")


if __name__ == "__main__":
    main()
