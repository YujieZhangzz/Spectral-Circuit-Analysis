"""
Unit layout construction and block partitioning.

Units are the granularity at which gradients are sliced:
  attn_head : V + O weights of one Q-head (V shared within a GQA group)
  mlp_gate  : MLP gate_proj (SwiGLU) or the up-projection (GPT-2 c_fc, Pythia h_to_4h)
  mlp_up    : MLP up_proj (SwiGLU only)
  mlp_down  : MLP down_proj / c_proj / 4h_to_h

Blocks are groups of units, found either by clustering a unit-similarity
matrix (find_blocks) or directly by layer (find_blocks_by_layer).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ArchInfo

@dataclass
class ArchInfo:
    name:             str      # "gpt2" | "llama" | "qwen2"
    n_heads:          int
    n_kv_heads:       int
    d_model:          int
    d_head:           int
    has_combined_qkv: bool     # True for GPT-2, False for LLaMA/Qwen

    @property
    def uses_gqa(self) -> bool:
        return self.n_kv_heads < self.n_heads

    @property
    def gqa_ratio(self) -> int:
        return self.n_heads // self.n_kv_heads


def detect_arch(model) -> ArchInfo:
    cfg = model.config
    cls = type(model).__name__.lower()
    model_type = getattr(cfg, "model_type", "").lower()
    if "gpt2" in cls or hasattr(cfg, "n_embd"):
        d = cfg.n_embd; nh = cfg.n_head
        return ArchInfo("gpt2", nh, nh, d, d // nh, True)
    elif "gptneox" in cls or model_type == "gpt_neox":
        # Pythia: combined QKV (interleaved per head), no GQA
        d   = cfg.hidden_size
        nh  = cfg.num_attention_heads
        return ArchInfo("pythia", nh, nh, d, d // nh, has_combined_qkv=True)
    else:
        d   = cfg.hidden_size
        nh  = cfg.num_attention_heads
        nkv = getattr(cfg, "num_key_value_heads", nh)
        return ArchInfo("llama", nh, nkv, d, d // nh, False)


# UnitInfo

@dataclass
class UnitInfo:
    unit_id:    int
    layer:      int
    unit_type:  str            # "attn_head" | "mlp_gate" | "mlp_up" | "mlp_down"
    head:       Optional[int]  # Q-head index (attn_head only)
    label:      str

    # Contiguous range: MLP weights, v_proj row slices, GPT-2 c_proj row slices
    dim_start: Optional[int] = None
    dim_end:   Optional[int] = None

    # Non-contiguous indices: GPT-2 c_attn V columns, LLaMA o_proj columns.
    # Filled in by build_unit_layout when needed.
    _v_indices: Optional[torch.Tensor] = field(default=None, repr=False)  # V / v_proj part
    _o_indices: Optional[torch.Tensor] = field(default=None, repr=False)  # O / o_proj part

    def get_indices(self) -> torch.Tensor:
        """
        Return the LongTensor indices of this unit in the flat gradient vector.
        """
        parts = []
        if self._v_indices is not None:
            parts.append(self._v_indices)
        if self._o_indices is not None:
            parts.append(self._o_indices)
        if parts:
            return torch.cat(parts) if len(parts) > 1 else parts[0]
        # MLP / contiguous range
        assert self.dim_start is not None
        return torch.arange(self.dim_start, self.dim_end, dtype=torch.long)

    @property
    def n_dims(self) -> int:
        if self._v_indices is not None or self._o_indices is not None:
            return (len(self._v_indices) if self._v_indices is not None else 0) + \
                   (len(self._o_indices) if self._o_indices is not None else 0)
        return self.dim_end - self.dim_start


# build_unit_layout

def build_unit_layout(
    model,
    param_filter: Callable[[str], bool],
    arch:         Optional[ArchInfo] = None,
    include_mlp:  bool = True,
) -> List[UnitInfo]:
    """
    Build the list of UnitInfo from model.named_parameters(). Offsets follow
    the same flattening order as obtain_gradients.

    Args:
        model:        pretrained model
        param_filter: the same filter passed to obtain_gradients
        arch:         ArchInfo (auto-detected if None)
        include_mlp:  whether to create MLP units

    Returns:
        List[UnitInfo], sorted by (layer, head, unit_type)
    """
    if arch is None:
        arch = detect_arch(model)

    d, nh, nkv, dh = arch.d_model, arch.n_heads, arch.n_kv_heads, arch.d_head

    # Step 1: record (name, offset, numel, shape) for every filtered parameter.
    # Offsets accumulate over filtered parameters only, matching the cat order
    # in obtain_gradients.
    param_segs: List[Tuple[str, int, int, Tuple]] = []
    offset = 0
    for name, param in model.named_parameters():
        n = param.numel()
        if param_filter(name):
            param_segs.append((name, offset, n, tuple(param.shape)))
            offset += n

    print(f"[UnitLayout] arch={arch.name}, {len(param_segs)} filtered params, "
          f"D_filtered={sum(s[2] for s in param_segs):,}")

    # Step 2: map each parameter to UnitInfo(s), per architecture
    units: List[UnitInfo] = []
    uid = 0
    if arch.name == "gpt2":
        layer_pat = r"transformer\.h\.(\d+)\."
    elif arch.name == "pythia":
        layer_pat = r"gpt_neox\.layers\.(\d+)\."
    else:
        layer_pat = r"model\.layers\.(\d+)\."

    for name, seg_start, numel, shape in param_segs:
        m = re.search(layer_pat, name)
        if m is None:
            continue
        layer = int(m.group(1))

        # GPT-2
        if arch.name == "gpt2":
            if ".attn.c_attn.weight" in name:
                # shape (d, 3d); V columns per Q-head (non-contiguous)
                n_cols = 3 * d
                for h in range(nh):
                    col_s = 2 * d + h * dh   # first column of the V block
                    v_idx = torch.tensor([
                        seg_start + row * n_cols + c
                        for row in range(d)
                        for c in range(col_s, col_s + dh)
                    ], dtype=torch.long)
                    units.append(UnitInfo(
                        unit_id=uid, layer=layer,
                        unit_type="attn_head", head=h,
                        label=f"L{layer}H{h}",
                        _v_indices=v_idx,
                    ))
                    uid += 1

            elif ".attn.c_proj.weight" in name:
                # shape (d, d); O rows per Q-head (contiguous), attached to the
                # existing unit for the same (layer, head)
                for h in range(nh):
                    o_start = seg_start + h * dh * d
                    o_end   = o_start + dh * d
                    existing = _find_unit(units, layer, h)
                    if existing is not None:
                        existing._o_indices = torch.arange(
                            o_start, o_end, dtype=torch.long
                        )
                    else:
                        # only c_proj passed the filter (no c_attn): standalone unit
                        units.append(UnitInfo(
                            unit_id=uid, layer=layer,
                            unit_type="attn_head", head=h,
                            label=f"L{layer}H{h}",
                            dim_start=o_start, dim_end=o_end,
                        ))
                        uid += 1

            elif ".mlp.c_fc.weight" in name and include_mlp:
                # GPT-2 MLP up-projection, categorised as mlp_gate
                units.append(UnitInfo(
                    unit_id=uid, layer=layer,
                    unit_type="mlp_gate", head=None,
                    label=f"L{layer}_MLP_fc",
                    dim_start=seg_start, dim_end=seg_start + numel,
                ))
                uid += 1

            elif ".mlp.c_proj.weight" in name and include_mlp:
                # GPT-2 MLP down-projection
                units.append(UnitInfo(
                    unit_id=uid, layer=layer,
                    unit_type="mlp_down", head=None,
                    label=f"L{layer}_MLP_proj",
                    dim_start=seg_start, dim_end=seg_start + numel,
                ))
                uid += 1

        # Pythia / GPT-NeoX
        elif arch.name == "pythia":
            if ".attention.query_key_value.weight" in name:
                # shape (3*d, d); rows are output features, laid out as
                # (num_heads, 3, head_size) interleaved per head:
                #   head h occupies rows [h*3*dh : (h+1)*3*dh]
                #     Q: rows [h*3*dh        : h*3*dh + dh]
                #     K: rows [h*3*dh + dh   : h*3*dh + 2*dh]
                #     V: rows [h*3*dh + 2*dh : (h+1)*3*dh]
                # Each row is `d` elements wide (in_features), row-major:
                #   flat index of (row r, col c) = seg_start + r*d + c
                for h in range(nh):
                    v_row_s = h * 3 * dh + 2 * dh
                    v_row_e = (h + 1) * 3 * dh
                    v_idx = torch.arange(
                        seg_start + v_row_s * d,
                        seg_start + v_row_e * d,
                        dtype=torch.long,
                    )
                    units.append(UnitInfo(
                        unit_id=uid, layer=layer,
                        unit_type="attn_head", head=h,
                        label=f"L{layer}H{h}",
                        _v_indices=v_idx,
                    ))
                    uid += 1

            elif ".attention.dense.weight" in name:
                # shape (d, d); output projection.
                # Input is (..., num_heads, head_size) flattened to (..., d), so
                # head h occupies input columns [h*dh : (h+1)*dh].
                # In (out=d, in=d) row-major each row contributes dh non-contiguous
                # columns per head (same slicing as LLaMA o_proj).
                n_in = d
                for h in range(nh):
                    o_idx = torch.tensor([
                        seg_start + row * n_in + c
                        for row in range(d)
                        for c in range(h * dh, (h + 1) * dh)
                    ], dtype=torch.long)
                    existing = _find_unit(units, layer, h)
                    if existing is not None:
                        existing._o_indices = o_idx
                    else:
                        units.append(UnitInfo(
                            unit_id=uid, layer=layer,
                            unit_type="attn_head", head=h,
                            label=f"L{layer}H{h}",
                            _o_indices=o_idx,
                        ))
                        uid += 1

            elif ".mlp.dense_h_to_4h.weight" in name and include_mlp:
                # Pythia MLP up-projection (no SwiGLU), categorised as mlp_gate
                units.append(UnitInfo(
                    unit_id=uid, layer=layer,
                    unit_type="mlp_gate", head=None,
                    label=f"L{layer}_MLP_h_to_4h",
                    dim_start=seg_start, dim_end=seg_start + numel,
                ))
                uid += 1

            elif ".mlp.dense_4h_to_h.weight" in name and include_mlp:
                units.append(UnitInfo(
                    unit_id=uid, layer=layer,
                    unit_type="mlp_down", head=None,
                    label=f"L{layer}_MLP_4h_to_h",
                    dim_start=seg_start, dim_end=seg_start + numel,
                ))
                uid += 1

        # LLaMA / Qwen
        else:
            if ".self_attn.v_proj.weight" in name:
                # shape (nkv*dh, d); V rows per KV-head, shared by several Q-heads.
                # Each Q-head's _v_indices point at the rows of its KV-head.
                for h in range(nh):
                    hk    = h // arch.gqa_ratio   # KV-head index
                    v_s   = seg_start + hk * dh * d
                    v_e   = v_s + dh * d
                    units.append(UnitInfo(
                        unit_id=uid, layer=layer,
                        unit_type="attn_head", head=h,
                        label=f"L{layer}H{h}",
                        _v_indices=torch.arange(v_s, v_e, dtype=torch.long),
                    ))
                    uid += 1

            elif ".self_attn.o_proj.weight" in name:
                # shape (d, nh*dh); O columns per Q-head (non-contiguous),
                # attached to the existing unit
                n_out = nh * dh
                for h in range(nh):
                    o_idx = torch.tensor([
                        seg_start + row * n_out + c
                        for row in range(d)
                        for c in range(h * dh, (h + 1) * dh)
                    ], dtype=torch.long)
                    existing = _find_unit(units, layer, h)
                    if existing is not None:
                        existing._o_indices = o_idx
                    else:
                        units.append(UnitInfo(
                            unit_id=uid, layer=layer,
                            unit_type="attn_head", head=h,
                            label=f"L{layer}H{h}",
                            _o_indices=o_idx,
                        ))
                        uid += 1

            elif ".mlp.gate_proj.weight" in name and include_mlp:
                units.append(UnitInfo(
                    unit_id=uid, layer=layer,
                    unit_type="mlp_gate", head=None,
                    label=f"L{layer}_MLP_gate",
                    dim_start=seg_start, dim_end=seg_start + numel,
                ))
                uid += 1

            elif ".mlp.up_proj.weight" in name and include_mlp:
                units.append(UnitInfo(
                    unit_id=uid, layer=layer,
                    unit_type="mlp_up", head=None,
                    label=f"L{layer}_MLP_up",
                    dim_start=seg_start, dim_end=seg_start + numel,
                ))
                uid += 1

            elif ".mlp.down_proj.weight" in name and include_mlp:
                units.append(UnitInfo(
                    unit_id=uid, layer=layer,
                    unit_type="mlp_down", head=None,
                    label=f"L{layer}_MLP_down",
                    dim_start=seg_start, dim_end=seg_start + numel,
                ))
                uid += 1

    # Step 3: reassign unit_id sequentially
    for i, u in enumerate(units):
        u.unit_id = i

    # Sort by (layer, head or 999, unit_type)
    type_order = {"attn_head": 0, "mlp_gate": 1, "mlp_up": 2, "mlp_down": 3}
    units.sort(key=lambda u: (u.layer, u.head if u.head is not None else 999,
                               type_order.get(u.unit_type, 9)))

    # Summary
    attn_units = [u for u in units if u.unit_type == "attn_head"]
    mlp_units  = [u for u in units if u.unit_type != "attn_head"]
    print(f"[UnitLayout] Total={len(units)} units  "
          f"(attn={len(attn_units)}, mlp={len(mlp_units)})")
    if units:
        sample = units[0]
        print(f"  Sample: {sample.label}  n_dims={sample.n_dims:,}")

    return units


def _find_unit(units: List[UnitInfo], layer: int, head: int) -> Optional[UnitInfo]:
    """Find the attn_head unit for (layer, head) among the existing units."""
    for u in reversed(units):   # most recently added units are at the end
        if u.layer == layer and u.head == head and u.unit_type == "attn_head":
            return u
    return None


# BlockResult

@dataclass
class BlockResult:
    unit_sim:        torch.Tensor       # (n_units, n_units) cosine similarity
    blocks:          List[List[int]]    # K blocks, each a list of unit_ids
    block_indices:   List[torch.Tensor] # flat-gradient indices of each block
    unit_layout:     List[UnitInfo]
    n_units:         int
    K:               int
    threshold_used:  float              # threshold actually used (auto or manual)
    threshold_source: str               # "auto_valley" | "auto_otsu" | "auto_percentile" | "manual"

    def summary(self) -> str:
        lines = [f"\n[BlockFinder] K={self.K} blocks  "
                 f"(threshold={self.threshold_used:.3f}, src={self.threshold_source}):"]
        for k, uids in enumerate(self.blocks):
            layers  = sorted({self.unit_layout[u].layer for u in uids})
            utypes  = set(self.unit_layout[u].unit_type for u in uids)
            d_k     = len(self.block_indices[k])
            lines.append(
                f"  Block {k}: {len(uids)} units, "
                f"layers={layers}, types={utypes}, d_k={d_k:,}"
            )
        return "\n".join(lines)


# find_blocks

def find_blocks(
    unit_layout:    List[UnitInfo],
    sign_accum:     torch.Tensor,            # (N, D) int8; unused here
    M:              int,                     # number of paraphrases; unused here
    grad_avg:       Optional[torch.Tensor] = None,  # (N, D) float16, required
    sc:             Optional[torch.Tensor] = None,  # unused
    semantic_mask:  Optional[torch.Tensor] = None,  # (D,) bool, semantic-dimension mask
    sim_threshold:  Optional[float] = None,
    min_block_size: int = 4,
    min_dims:       int = 100,              # min #semantic dims per unit; else use all dims
    save_dir:       Optional[str] = None,
) -> BlockResult:
    """
    Cluster units into blocks using kernel alignment in the N x N Gram space.

      G_u = grad_avg[:, unit_indices]          # (N, d_k), optionally semantic dims only
      K_u = G_u @ G_u.T                        # (N, N), independent of d_k
      sim(u,v) = <K_u, K_v>_F / (‖K_u‖_F · ‖K_v‖_F)

    Because every K_u is N x N, attn heads and MLP units can be compared
    directly. Keeping the per-instance structure (rather than averaging over
    instances) means units that respond to the same instances score high.

    Steps: (A) Gram matrix per unit, (B) pairwise kernel alignment,
    (C) threshold from the off-diagonal similarity distribution,
    (D) average-linkage clustering.
    """
    if grad_avg is None:
        raise ValueError(
            "[BlockFinder] grad_avg is required for r_u construction. "
            "Set cfg.collector.store_grad_avg=true (default) and re-run Phase 0."
        )

    n_units = len(unit_layout)
    print(f"[BlockFinder] n_units={n_units}, rep=kernel_alignment (N×N Gram, cross-type enabled)")

    # Step A: Gram matrix K_u = G_u G_u^T for each unit
    N = grad_avg.shape[0]
    kernels: List[torch.Tensor] = []   # each (N, N)

    type_groups: Dict[str, List[int]] = {}   # only used for debug printing
    for uid, u in enumerate(unit_layout):
        type_groups.setdefault(u.unit_type, []).append(uid)

    for uid in range(n_units):
        idx = unit_layout[uid].get_indices()          # (d_k,)
        if semantic_mask is not None:
            sem_local = semantic_mask[idx]            # (d_k,) bool
            if int(sem_local.sum()) >= min_dims:
                G_u = grad_avg[:, idx[sem_local]].float()   # (N, n_sem)
            else:
                G_u = grad_avg[:, idx].float()              # fall back to all dims
        else:
            G_u = grad_avg[:, idx].float()
        K_u = G_u @ G_u.T                                   # (N, N)
        kernels.append(K_u)

    # Step B: kernel alignment sim(u,v) = <K_u,K_v>_F / (‖K_u‖·‖K_v‖)
    K_stack  = torch.stack(kernels, dim=0)           # (n_units, N, N)
    K_flat   = K_stack.view(n_units, -1)             # (n_units, N*N)
    K_norms  = K_flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    K_norm   = K_flat / K_norms                      # (n_units, N*N) L2-normalized
    sim      = K_norm @ K_norm.T                     # (n_units, n_units)
    sim.fill_diagonal_(1.0)

    print(f"[BlockFinder] Full sim matrix: "
          f"mean={sim.mean().item():.3f}, std={sim.std().item():.3f}")

    # Debug: within-type similarity distribution
    print("\n[Debug] Per-type similarity summary (kernel alignment):")
    for utype, uids in type_groups.items():
        if len(uids) < 2:
            continue
        mask = torch.zeros(n_units, n_units, dtype=torch.bool)
        for i in uids:
            for j in uids:
                if i != j:
                    mask[i, j] = True
        vals = sim[mask]
        print(f"  {utype:12s}  mean={vals.mean():.3f}  "
              f"std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")
    print()

    # Step C: threshold from all off-diagonal pairs (including cross-type)
    off_diag_mask = ~torch.eye(n_units, dtype=torch.bool)
    all_vals      = sim[off_diag_mask]

    if sim_threshold is None:
        threshold, candidates, source = suggest_threshold(
            sim, unit_layout, save_dir=save_dir,
            intra_vals=all_vals,
        )
        print(f"[BlockFinder] Auto threshold={threshold:.3f} (source={source})")
        print(f"  Candidates: " +
              ", ".join(f"{k}={v:.3f}" for k, v in candidates.items()))
    else:
        threshold = sim_threshold
        source    = "manual"
        print(f"[BlockFinder] Manual threshold={threshold:.3f}")

    # Step D: clustering
    blocks = _scipy_cluster(sim, threshold, min_block_size, unit_layout)

    block_indices = []
    for block_uids in blocks:
        parts = [unit_layout[uid].get_indices() for uid in block_uids]
        block_indices.append(torch.cat(parts))

    result = BlockResult(
        unit_sim         = sim,
        blocks           = blocks,
        block_indices    = block_indices,
        unit_layout      = unit_layout,
        n_units          = n_units,
        K                = len(blocks),
        threshold_used   = threshold,
        threshold_source = source,
    )
    print(result.summary())
    return result


def _scipy_cluster(
    sim:            torch.Tensor,
    threshold:      float,
    min_block_size: int,
    unit_layout:    List[UnitInfo],
) -> List[List[int]]:
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from collections import defaultdict

    n       = sim.shape[0]
    dist    = np.clip(1.0 - sim.numpy().astype(np.float64), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)

    Z      = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    raw: Dict[int, List[int]] = defaultdict(list)
    for uid, cid in enumerate(labels):
        raw[int(cid)].append(uid)

    final, leftover = [], []
    for members in raw.values():
        if len(members) >= min_block_size:
            final.append(sorted(members))
        else:
            leftover.extend(members)

    if leftover:
        if final:
            for uid in leftover:
                # assign to the block with the highest mean similarity (any type)
                best = max(range(len(final)),
                           key=lambda k: sim[uid, final[k]].mean().item())
                final[best].append(uid)
        else:
            final.append(sorted(leftover))

    final.sort(key=lambda b: b[0])
    return final


# suggest_threshold: derive a clustering threshold from the similarity distribution

def suggest_threshold(
    sim:         torch.Tensor,
    unit_layout: List[UnitInfo],
    save_dir:    Optional[str] = None,
    intra_vals:  Optional[torch.Tensor] = None,  # pair values to use instead of the upper triangle
) -> Tuple[float, Dict[str, float], str]:
    """
    Derive a clustering threshold from the pairwise similarity distribution.
    Candidates: KDE valley (preferred), Otsu (fallback), 25th percentile (reported).
    intra_vals: if given, use these values instead of all upper-triangle pairs.
    """
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    n  = sim.shape[0]
    ul = unit_layout

    # Use intra_vals if given, else all upper-triangle pairs
    if intra_vals is not None:
        pairwise = intra_vals.numpy().astype(np.float64)
    else:
        mask     = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pairwise = sim[mask].numpy().astype(np.float64)

    if len(pairwise) < 3:
        # too few units: return a fixed default
        return 0.4, {"fallback": 0.4}, "fallback"

    # Within-type pairs, grouped by unit_type (for plotting)
    type_set = sorted({u.unit_type for u in ul})
    pair_groups: Dict[str, List[float]] = {t: [] for t in type_set}
    for i in range(n):
        for j in range(i + 1, n):
            ti = ul[i].unit_type
            tj = ul[j].unit_type
            if ti == tj:
                pair_groups[ti].append(sim[i, j].item())

    # Method 1: KDE valley
    valley_threshold = _kde_valley(pairwise)

    # Method 2: Otsu
    otsu_threshold = _otsu(pairwise)

    # Method 3: 25th percentile
    p25_threshold = float(np.percentile(pairwise, 25))

    candidates = {
        "valley":      valley_threshold if valley_threshold is not None else float("nan"),
        "otsu":        otsu_threshold,
        "percentile25": p25_threshold,
    }

    # Pick the final threshold
    if valley_threshold is not None:
        suggested, source = valley_threshold, "auto_valley"
    else:
        suggested, source = otsu_threshold,  "auto_otsu"
        print(f"  [Threshold] No clear valley found, falling back to Otsu ({otsu_threshold:.3f})")

    # Plot
    if save_dir is not None:
        _plot_threshold_analysis(
            pairwise, pair_groups, candidates, suggested, source,
            save_dir=save_dir,
        )

    return suggested, candidates, source


def _kde_valley(values: "np.ndarray") -> Optional[float]:
    """
    Find a valley (local minimum) of the KDE within [p10, p90].

    If there are several, return the one closest to the median.
    Returns None if there is no meaningful valley (unimodal distribution).
    """
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    kde = gaussian_kde(values, bw_method="silverman")
    x   = np.linspace(values.min() - 0.05, values.max() + 0.05, 1000)
    y   = kde(x)

    # local minima = peaks of -y
    valleys_idx, props = find_peaks(-y, prominence=0.01 * y.max())

    if len(valleys_idx) == 0:
        return None   # unimodal

    # restrict to [p10, p90]
    p10, p90 = np.percentile(values, 10), np.percentile(values, 90)
    valid = [idx for idx in valleys_idx if p10 <= x[idx] <= p90]

    if not valid:
        return None

    # several valleys: take the one closest to the median
    median = np.median(values)
    best   = min(valid, key=lambda idx: abs(x[idx] - median))
    return float(x[best])


def _otsu(values: "np.ndarray") -> float:
    """
    Otsu threshold on the similarity values (maximises between-class variance).
    """
    import numpy as np

    # discretise into 256 bins
    bins   = np.linspace(values.min(), values.max(), 257)
    hist, edges = np.histogram(values, bins=bins)
    hist   = hist.astype(float) / hist.sum()
    mids   = (edges[:-1] + edges[1:]) / 2

    best_var, best_t = -1.0, mids[0]
    total_mean = (hist * mids).sum()

    w0, mu0 = 0.0, 0.0
    for i in range(len(hist)):
        w0  += hist[i]
        mu0 += hist[i] * mids[i]
        w1   = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        mu1      = (total_mean - mu0) / w1
        var_b    = w0 * w1 * (mu0 / w0 - mu1) ** 2
        if var_b > best_var:
            best_var = var_b
            best_t   = mids[i]

    return float(best_t)


def _plot_threshold_analysis(
    pairwise:    "np.ndarray",
    pair_groups: Dict[str, List[float]],
    candidates:  Dict[str, float],
    suggested:   float,
    source:      str,
    save_dir:    str,
) -> None:
    """
    Two panels:
      left:  histogram + KDE of all pairs, with the candidate thresholds
      right: KDE per unit type (within-type pairs)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import gaussian_kde
    except ImportError:
        print("[Threshold plot] matplotlib / scipy not available, skip.")
        return

    from utils.paper_style import paper_size
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=paper_size("double", aspect=2.4))
    fig.suptitle(
        f"Pairwise Similarity Distribution  -  Suggested threshold={suggested:.3f} ({source})",
        fontsize=12
    )

    x_grid = np.linspace(-1.0, 1.0, 500)

    # Left: overall distribution
    ax1.hist(pairwise, bins=60, density=True,
             color="#b0c4de", alpha=0.5, label="histogram")

    kde = gaussian_kde(pairwise, bw_method="silverman")
    y   = kde(x_grid)
    ax1.plot(x_grid, y, color="#2c5f8a", lw=2, label="KDE")

    # candidate thresholds
    line_styles = {
        "valley":       ("#e05c5c", "--", "KDE valley"),
        "otsu":         ("#f5a623", "-.", "Otsu"),
        "percentile25": ("#7ed321", ":",  "25th pct"),
    }
    for key, (color, ls, label) in line_styles.items():
        v = candidates.get(key, float("nan"))
        if np.isnan(v):
            continue
        is_chosen = (abs(v - suggested) < 1e-6)
        lw = 2.5 if is_chosen else 1.5
        ax1.axvline(v, color=color, linestyle=ls, linewidth=lw,
                    label=f"{label}={v:.3f}" + (" (chosen)" if is_chosen else ""))

    ax1.set_xlabel("Cosine similarity")
    ax1.set_ylabel("Density")
    ax1.set_title("All pairwise similarities")
    ax1.legend(fontsize=8)
    ax1.set_xlim(-1.05, 1.05)
    ax1.grid(alpha=0.3)

    # summary stats
    ax1.text(0.02, 0.97,
             f"n_pairs={len(pairwise):,}\n"
             f"mean={pairwise.mean():.3f}\n"
             f"std={pairwise.std():.3f}",
             transform=ax1.transAxes, fontsize=8,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Right: per unit type
    group_colors = {
        "attn_head": "#4f86c6",
        "mlp_gate":  "#e05c5c",
        "mlp_up":    "#f5a623",
        "mlp_down":  "#7ed321",
    }
    any_plotted = False
    for gname, vals in pair_groups.items():
        if len(vals) < 5:
            continue
        vals_arr = np.array(vals)
        if vals_arr.std() < 1e-6:
            print(f"  [skip] {gname}: std≈0, cannot fit KDE")
            continue
        try:
            kde_g = gaussian_kde(vals_arr, bw_method="silverman")
        except np.linalg.LinAlgError:
            print(f"  [skip] {gname}: KDE singular matrix")
            continue
        y_g   = kde_g(x_grid)
        ax2.plot(x_grid, y_g, color=group_colors[gname], lw=2,
                 label=f"{gname} (n={len(vals):,})")
        ax2.fill_between(x_grid, y_g, alpha=0.12, color=group_colors[gname])
        any_plotted = True

    if any_plotted:
        ax2.axvline(suggested, color="#333333", linestyle="--", lw=2,
                    label=f"chosen={suggested:.3f}")
        ax2.set_xlabel("Cosine similarity")
        ax2.set_ylabel("Density")
        ax2.set_title("By pair type\n(shows if attn/MLP have different patterns)")
        ax2.legend(fontsize=8)
        ax2.set_xlim(-1.05, 1.05)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Not enough pairs per group\nto plot separately",
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    out = Path(save_dir) / "threshold_analysis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Threshold analysis -> {out}")


# build_head_layout (deprecated)

def build_head_layout(
    n_layers: int,
    n_heads:  int,
    d_model:  int,
    scope:    str = "attn_VO",
) -> List[Dict]:
    """
    Deprecated; use build_unit_layout().
    """
    import warnings
    warnings.warn(
        "build_head_layout is deprecated; use build_unit_layout() instead.",
        DeprecationWarning, stacklevel=2,
    )
    d_head = d_model // n_heads
    params_per_head = 2 * d_model * d_head
    layout = []
    head_id = 0
    for layer in range(n_layers):
        for h in range(n_heads):
            start = head_id * params_per_head
            layout.append({
                "head_id":   head_id, "layer": layer, "head": h,
                "dim_start": start, "dim_end": start + params_per_head,
                "label":     f"L{layer}H{h}",
            })
            head_id += 1
    return layout


# find_blocks_by_layer: one block per layer, no similarity clustering

def find_blocks_by_layer(
    unit_layout: List[UnitInfo],
    strategy:    str = "layer_attn_mlp",
    save_dir:    Optional[str] = None,
) -> BlockResult:
    """
    Layer-stratified blocking, without gradient-similarity clustering.

    strategy="layer_attn_mlp" : per layer, one attn block + one MLP block
    strategy="layer_combined" : per layer, a single block (attn + MLP)

    Returns the same BlockResult as find_blocks(). unit_sim is a
    block-diagonal indicator (1 within a block, 0 across) so that
    plot_unit_heatmap still produces a meaningful figure.
    """
    from collections import defaultdict

    n_units = len(unit_layout)
    type_order = {"attn": 0, "mlp": 1}

    groups: Dict[tuple, List[int]] = defaultdict(list)
    for uid, u in enumerate(unit_layout):
        if strategy == "layer_attn_mlp":
            g = "attn" if u.unit_type == "attn_head" else "mlp"
            key = (u.layer, g)
        else:
            key = (u.layer,)
        groups[key].append(uid)

    # sort by (layer, attn before mlp)
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (k[0], type_order.get(k[1], 0) if len(k) > 1 else 0),
    )
    blocks: List[List[int]] = [sorted(groups[k]) for k in sorted_keys]

    block_indices = [
        torch.cat([unit_layout[uid].get_indices() for uid in uids])
        for uids in blocks
    ]

    # block-diagonal unit_sim (for plotting)
    sim = torch.zeros(n_units, n_units)
    uid_to_block = {uid: k for k, blk in enumerate(blocks) for uid in blk}
    for i in range(n_units):
        for j in range(n_units):
            if uid_to_block[i] == uid_to_block[j]:
                sim[i, j] = 1.0

    result = BlockResult(
        unit_sim         = sim,
        blocks           = blocks,
        block_indices    = block_indices,
        unit_layout      = unit_layout,
        n_units          = n_units,
        K                = len(blocks),
        threshold_used   = 0.0,
        threshold_source = strategy,
    )
    print(result.summary())
    return result


# Plotting

def plot_unit_heatmap(
    result:   BlockResult,
    save_dir: Optional[str] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    sim = result.unit_sim.numpy()
    n   = result.n_units
    ul  = result.unit_layout

    # n x n unit similarity heatmap, square single-column figure
    from utils.paper_style import paper_size
    fig, ax = plt.subplots(figsize=paper_size("single", aspect=1.0))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # block boundaries
    pos = 0
    for block in result.blocks:
        pos += len(block)
        if pos < n:
            ax.axhline(pos - 0.5, color="white", lw=1.5)
            ax.axvline(pos - 0.5, color="white", lw=1.5)

    # layer tick labels (one group per n_heads)
    attn_units = [u for u in ul if u.unit_type == "attn_head"]
    if attn_units:
        n_heads_per_layer = max(u.head for u in attn_units) + 1
        layer_ticks = [i * n_heads_per_layer + n_heads_per_layer // 2
                       for i in range(ul[-1].layer + 1)]
        layer_labels = [f"L{i}" for i in range(ul[-1].layer + 1)]
        ax.set_xticks(layer_ticks); ax.set_xticklabels(layer_labels, fontsize=7)
        ax.set_yticks(layer_ticks); ax.set_yticklabels(layer_labels, fontsize=7)

    plt.colorbar(im, ax=ax, label="cosine similarity (grad_avg)")
    ax.set_title(
        f"Unit similarity ({n}×{n})\n"
        f"K={result.K} blocks  |  threshold={0.4}"
    )

    if save_dir:
        out = Path(save_dir) / "unit_heatmap.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=250, bbox_inches="tight")
        print(f"  [plot] {out}")
