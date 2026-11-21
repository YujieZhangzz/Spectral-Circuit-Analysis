"""
Validate the cross-task unit classification (universal / family-specific) by
parameter-subspace knockout, against size-matched random unit sets.

For each unit set from classify/v1_min_annotated.csv (Universal, GT/SVA/COUNTRY
family sets, a mixed control), knocks out every unit (projecting its subspace
out of its parameters), evaluates all 12 tasks and restores the weights.
Lift_S(t) = (L_clean(t) - L_S(t)) / |L_clean(t)| minus the mean random-set value.
Writes lift_matrix.csv, raw_metrics.csv, concentration.csv and summary.txt to
cross_task/run_gpt_12tasks/<variant>/knockout_validation_<mode>/.

Usage:
    python cross_task/knockout_validate.py --knockout_mode subspace
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.model_utils import load_model_and_tokenizer, get_param_filter
from spectral.block_finder import build_unit_layout, detect_arch
from spectral.verify_knockout import apply_knockout, build_param_seg_map
from cross_task.weighted_unit_svd import load_task_subspaces
from circuits_discovery.faithfulness.data import load_ioi
from circuits_discovery.faithfulness.patching import (
    get_year_token_ids,
    build_sva_verb_token_ids,
)


# Task manifest
EAP_DATA = Path("eap_data")
VERB_LIST_CSV = EAP_DATA / "sva" / "combined_verb_list.csv"

TASKS: Dict[str, Dict] = {
    "col_obj":         dict(csv=EAP_DATA / "colored-objects/sampled_all_gpt2.csv",        schema="logit_diff"),
    "country_cap":     dict(csv=EAP_DATA / "fact-retrieval-comma/sampled_all_gpt2.csv",   schema="logit_diff"),
    "gender_bias":     dict(csv=EAP_DATA / "gender-bias/sampled_all_gpt2_200.csv",        schema="logit_diff"),
    "gp":              dict(csv=EAP_DATA / "gender-pronoun/sampled_all_gpt2_200.csv",     schema="logit_diff"),
    "gt":              dict(csv=EAP_DATA / "greater-than/sampled_all_gpt2_200.csv",       schema="greater_than"),
    "gt_price":        dict(csv=EAP_DATA / "greater-than-price/sampled_all_gpt2_200.csv", schema="greater_than"),
    "ioi_strong":      dict(csv=EAP_DATA / "ioi/sampled_all_gpt2_200.csv",                 schema="logit_diff"),
    "person_country":  dict(csv=EAP_DATA / "person-country/sampled_all_gpt2.csv",          schema="logit_diff"),
    "person_pro":      dict(csv=EAP_DATA / "person-profession/sampled_all_gpt2.csv",       schema="logit_diff"),
    "sva":             dict(csv=EAP_DATA / "sva/sampled_all_gpt2_200_extended-2.csv",      schema="sva_verbset"),
    "sva_blimp":       dict(csv=EAP_DATA / "sva-blimp/gpt2_blimp.csv",                     schema="sva_verbset"),
    "sva_isare":       dict(csv=EAP_DATA / "sva/sampled_all_gpt2_200_extended-2.csv",      schema="sva_isare"),
}

# Per-task results dir (experiment.name in the task config) holding that
# task's own U_u.
TASK_NPZ_DIRS = {
    "col_obj":         "gpt2_small_col_N100_M5",
    "country_cap":     "gpt2_small_country_cap_N100_M5",
    "gender_bias":     "gpt2_small_gb_N100_M5",
    "gp":              "gpt2_small_gp_N100_M5",
    "gt":              "gpt2_small_gt_N100_M5",
    "gt_price":        "gpt2_small_gt_price_N100_M5_linear",
    "ioi_strong":      "gpt2_small_ioi_strong_N100_M5_linear",
    "person_country":  "gpt2_small_per_country_N100_M5",
    "person_pro":      "gpt2_small_per_pro_N100_M5",
    "sva":             "gpt2_small_sva_N100_M5_ori",
    "sva_blimp":       "gpt2_small_sva_blimp_N100_M5",
    "sva_isare":       "gpt2_small_sva_N100_M5_isare_back100",
}

# Task -> family (for the concentration metric)
TASK_TO_FAMILY = {
    "col_obj":         None,                # singleton orphan in v1
    "country_cap":     "COUNTRY",
    "gender_bias":     "PRONOUN",
    "gp":              "PRONOUN",
    "gt":              "GT",
    "gt_price":        "GT",
    "ioi_strong":      None,                # singleton orphan in v1
    "person_country":  "COUNTRY",
    "person_pro":      "PRONOUN",
    "sva":             "SVA",
    "sva_blimp":       "SVA",
    "sva_isare":       "SVA",
}


# Metric builders (HF logits)
def build_metric_fn(schema, batch, model, tokenizer, device):
    """Return callable(logits, input_lengths) -> scalar tensor."""
    if schema == "logit_diff":
        labels = batch.labels.to(device)                        # (B, 2)
        def metric(logits, input_lengths):
            B = logits.size(0)
            last = logits[torch.arange(B, device=device), input_lengths - 1]
            picked = torch.gather(last, -1, labels)             # (B, 2)
            return (picked[:, 0] - picked[:, 1]).mean()
        return metric

    if schema == "greater_than":
        year_ids = get_year_token_ids(tokenizer).to(device)     # (100,)
        labels   = batch.labels.to(device)                      # (B,) start year YY
        YY = torch.arange(100, device=device).unsqueeze(0)      # (1, 100)
        def metric(logits, input_lengths):
            B = logits.size(0)
            last  = logits[torch.arange(B, device=device), input_lengths - 1]
            probs = F.softmax(last.float(), dim=-1)
            yp    = probs[:, year_ids]                          # (B, 100)
            gt_mask = (YY > labels.unsqueeze(1)).float()
            le_mask = 1.0 - gt_mask
            return ((yp * gt_mask).sum(-1) - (yp * le_mask).sum(-1)).mean()
        return metric

    if schema == "sva_verbset":
        sing_ids, plur_ids = build_sva_verb_token_ids(tokenizer, str(VERB_LIST_CSV))
        sing_ids = sing_ids.to(device); plur_ids = plur_ids.to(device)
        labels   = batch.labels.to(device)                      # (B,) plural in {0,1}
        def metric(logits, input_lengths):
            B = logits.size(0)
            last  = logits[torch.arange(B, device=device), input_lengths - 1]
            probs = F.softmax(last.float(), dim=-1)
            sing_p = probs[:, sing_ids].sum(-1)
            plur_p = probs[:, plur_ids].sum(-1)
            return torch.where(labels == 0, sing_p - plur_p, plur_p - sing_p).mean()
        return metric

    if schema == "sva_isare":
        is_id  = tokenizer(" is",  add_special_tokens=False)["input_ids"][0]
        are_id = tokenizer(" are", add_special_tokens=False)["input_ids"][0]
        labels = batch.labels.to(device)
        def metric(logits, input_lengths):
            B = logits.size(0)
            last = logits[torch.arange(B, device=device), input_lengths - 1]
            d = last[:, is_id] - last[:, are_id]
            return torch.where(labels == 0, d, -d).mean()
        return metric

    raise ValueError(f"unknown schema: {schema}")


# Eval one task (single forward pass; n=100 fits easily for gpt2-small)
def run_eval(model, tokenizer, batch, metric_fn, device, max_len=128) -> float:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.pad_token_id = pad_id
    tokenizer.padding_side = "right"
    enc = tokenizer(batch.clean, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    input_lengths = enc["attention_mask"].sum(1)
    return float(metric_fn(logits, input_lengths).item())


# Knockout / restore for a set of units
def save_affected_params(model, param_segs, all_global_idx):
    """Snapshot the param values that the upcoming knockout will modify."""
    seg_dict   = {name: (s, e) for name, s, e in param_segs}
    saved      = {}
    pdict      = dict(model.named_parameters())
    flat = torch.tensor(all_global_idx, dtype=torch.long)
    for name, fs, fe in param_segs:
        m = (flat >= fs) & (flat < fe)
        if not m.any():
            continue
        ppos = (flat[m] - fs).long()
        param = pdict[name]
        # store position + values
        saved[name] = (ppos.clone(), param.data.view(-1)[ppos.to(param.device)].clone())
    return saved


def restore_affected_params(model, saved):
    pdict = dict(model.named_parameters())
    for name, (ppos, vals) in saved.items():
        p = pdict[name]
        p.data.view(-1)[ppos.to(p.device)] = vals.to(p.dtype).to(p.device)


def build_unit_param_map(unit_by_id, param_segs):
    """For each unit_id, pre-compute [(param_name, positions_in_param, U_rows), ...].

    Avoids the per-call Python loop in apply_knockout(). Computed once per run.
    """
    out: Dict[int, List[Tuple[str, torch.Tensor, torch.Tensor]]] = {}
    for uid, unit in unit_by_id.items():
        idx = torch.tensor(np.asarray(unit.get_indices(), dtype=np.int64).ravel(),
                           dtype=torch.long)
        groups: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for name, fs, fe in param_segs:
            mask = (idx >= fs) & (idx < fe)
            if not mask.any():
                continue
            positions = (idx[mask] - fs).long()
            u_rows    = torch.where(mask)[0].long()
            groups.append((name, positions, u_rows))
        out[uid] = groups
    return out


def compute_block_derived_U(task_dir: Path, unit_subs: dict, unit_by_id: dict,
                              relevant_uids: List[int]) -> dict:
    """
    Build per-unit block-derived orthonormal basis Q_u ∈ R^(d_u, ρ_B).

    For each unit u with parent block B in this task:
        load U_B (block's left singular basis, d_block × ρ_B)
        slice rows of u's params: M = U_B[local_idx, :]  (d_u, ρ_B)
        QR: Q, _ = qr(M)  (d_u, min(d_u, ρ_B))
    Returns a unit_subs-shaped dict where 'U' is the block-derived Q.

    Memory: blocks are loaded then freed; only Q is retained per unit.
    """
    out = {}
    block_cache: Dict[int, Tuple[torch.Tensor, dict]] = {}
    for uid in relevant_uids:
        if uid not in unit_subs or uid not in unit_by_id:
            continue
        entry = unit_subs[uid]
        block_id = int(entry["meta"]["parent_block"])
        if block_id not in block_cache:
            U_B_path = task_dir / f"U_rho_block{block_id}.pt"
            gidx_path = task_dir / f"global_indices_block{block_id}.pt"
            if not U_B_path.exists() or not gidx_path.exists():
                print(f"    [warn] missing block files for block {block_id} in {task_dir.name}, skipping")
                continue
            U_B  = torch.load(U_B_path, map_location="cpu", weights_only=True).float()
            gidx = torch.load(gidx_path, map_location="cpu", weights_only=True).long().numpy()
            gidx_to_pos = {int(g): i for i, g in enumerate(gidx)}
            block_cache[block_id] = (U_B, gidx_to_pos)
        U_B, gidx_to_pos = block_cache[block_id]

        unit_global = np.asarray(unit_by_id[uid].get_indices(), dtype=np.int64).ravel()
        try:
            local_idx = np.array([gidx_to_pos[int(g)] for g in unit_global], dtype=np.int64)
        except KeyError as e:
            print(f"    [warn] unit {uid} param idx {e.args[0]} not in block {block_id}, skipping")
            continue
        M = U_B[torch.tensor(local_idx, dtype=torch.long), :]    # (d_u, rho_B)
        Q, _ = torch.linalg.qr(M, mode="reduced")                # (d_u, min(d_u, rho_B))
        new_entry = {**entry, "meta": {**entry["meta"]}}
        new_entry["U"]   = Q.numpy().astype(np.float32)
        new_entry["rho"] = int(Q.shape[1])
        out[uid] = new_entry

    # Free block caches before returning
    block_cache.clear()
    return out


def apply_knockout_fast(model_named, unit_param_groups, U_torch, mode="subspace"):
    """
    Apply per-unit knockout. mode controls how aggressive:
      - "subspace": project out U_torch (rho_u dims); default
      - "head_zero": zero all d_u params of the unit
    """
    for name, positions, u_rows in unit_param_groups:
        param  = model_named[name]
        device = param.device
        pos    = positions.to(device)
        if mode == "head_zero":
            param.data.view(-1)[pos] = 0
            continue
        # subspace projection
        U_loc  = U_torch[u_rows.to(U_torch.device)].to(device).float()    # (d_partial, rho)
        w      = param.data.view(-1)[pos].float()
        proj   = U_loc @ (U_loc.T @ w)
        param.data.view(-1)[pos] = (w - proj).to(param.dtype)


def knockout_set(model, param_segs, unit_by_id, unit_subs, unit_ids,
                 unit_param_map=None, model_named=None, mode="subspace"):
    """
    Apply per-unit knockout for every uid in unit_ids; returns (saved, n_knocked).
    Uses fast vectorized path when unit_param_map + model_named are provided.
    `mode` is forwarded to apply_knockout_fast: "subspace" or "head_zero".
    """
    # Save affected params (single pass)
    all_idx = []
    for uid in unit_ids:
        if uid not in unit_by_id or uid not in unit_subs:
            continue
        idx = np.asarray(unit_by_id[uid].get_indices(), dtype=np.int64).ravel()
        all_idx.append(idx)
    if not all_idx:
        return {}, 0
    all_idx_t = np.concatenate(all_idx)
    saved = save_affected_params(model, param_segs, all_idx_t)

    # Apply knockout per unit (vectorized)
    use_fast = (unit_param_map is not None) and (model_named is not None)
    n = 0
    for uid in unit_ids:
        if uid not in unit_by_id or uid not in unit_subs:
            continue
        if use_fast:
            if mode == "head_zero":
                U = None    # ignored
                apply_knockout_fast(model_named, unit_param_map[uid], None, mode="head_zero")
            else:
                U = torch.from_numpy(unit_subs[uid]["U"].astype(np.float32))
                apply_knockout_fast(model_named, unit_param_map[uid], U, mode=mode)
        else:
            idx = torch.tensor(np.asarray(unit_by_id[uid].get_indices(), dtype=np.int64).ravel(),
                               dtype=torch.long)
            U = torch.tensor(unit_subs[uid]["U"].astype(np.float32))
            apply_knockout(model, param_segs, idx, U)
        n += 1
    return saved, n


# Unit set selection
def parse_classification(csv_path: Path, allowed_uids: set | None = None):
    rows = list(csv.DictReader(open(csv_path)))
    if allowed_uids is not None:
        rows = [r for r in rows if int(r["unit_id"]) in allowed_uids]
    sets: Dict[str, List[int]] = {}

    # universal
    sets["Universal"] = [int(r["unit_id"]) for r in rows
                          if r["category_key"] == "universal"]

    # GT specialized: full set + top-4 by within_GT_min
    gt_spec = [r for r in rows if r["category_key"] == "family_GT_specialized"]
    gt_spec.sort(key=lambda r: -float(r["within_GT_min"]))
    sets[f"GT_specialized_{len(gt_spec)}"] = [int(r["unit_id"]) for r in gt_spec]
    sets["GT_specialized_top4"]            = [int(r["unit_id"]) for r in gt_spec[:4]]

    sets["SVA_specialized_4"] = [int(r["unit_id"]) for r in rows
                                  if r["category_key"] == "family_SVA_specialized"]

    sets["COUNTRY_broad_2"]   = [int(r["unit_id"]) for r in rows
                                  if r["category_key"] == "family_COUNTRY_broad"]

    # Control: random mixed units (size-matched to GT)
    mixed = [int(r["unit_id"]) for r in rows if r["category_key"] == "mixed"]
    n_ctrl = min(22, len(mixed))
    rng = np.random.default_rng(0)
    sets[f"Mixed_{n_ctrl}_control"] = sorted(rng.choice(mixed, n_ctrl, replace=False).tolist()) if mixed else []

    return sets


def make_random_sets(n_random: int, sizes: List[int], all_unit_ids: List[int],
                     base_seed: int = 100):
    """For each unique size, generate n_random random subsets of that size."""
    out = {}
    for size in sizes:
        out[size] = []
        for s in range(n_random):
            rng = np.random.default_rng(base_seed + size * 1000 + s)
            out[size].append(sorted(rng.choice(all_unit_ids, size, replace=False).tolist()))
    return out


# Main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=100, help="prompts per task")
    ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--variant", default="consistency_share",
                    choices=["consistency_share", "sigma_weighted"])
    ap.add_argument("--scope", default="attn_VO_mlp")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out_subdir", default=None,
                    help="Override output subdir name. Default: knockout_validation_<mode>")
    ap.add_argument("--knockout_mode", default="subspace",
                    choices=["subspace", "block_full", "head_zero"],
                    help="subspace: project out U_u (rho_u dims, ~0.005% of d_u). "
                         "block_full: project out parent block's full subspace (rho_B dims). "
                         "head_zero: zero entire unit params (extreme).")
    ap.add_argument("--attn_only", action="store_true", default=True,
                    help="Restrict all unit sets + random baselines to attn_head units (skip MLPs)")
    ap.add_argument("--include_mlp", dest="attn_only", action="store_false",
                    help="Override --attn_only to also include MLP units")
    ap.add_argument("--per_task_u", action="store_true", default=True,
                    help="Use each task's own U_u when knocking out (avoid donor-task bias).")
    ap.add_argument("--single_donor_npz", default=None,
                    help="Override per_task_u: use this single npz as donor U for all tasks.")
    ap.add_argument("--sanity", action="store_true",
                    help="Sanity mode: skip random baselines, restrict to 3 main sets.")
    return ap.parse_args()


def main():
    args = parse_args()
    base = REPO_ROOT / "cross_task" / "run_gpt_12tasks" / args.variant
    # Reference npz (ioi_strong); only printed. The knockout U_u comes from
    # each task's own npz, or from --single_donor_npz.
    npz_dir_name = "gpt2_small_ioi_strong_N100_M5_linear"
    npz  = REPO_ROOT / "results" / npz_dir_name / (
               "weighted_unit_subspaces.npz" if args.variant == "consistency_share"
               else "weighted_unit_subspaces_sigma.npz"
           )
    classify_csv = base / "classify" / "v1_min_annotated.csv"
    out_subdir = args.out_subdir or f"knockout_validation_{args.knockout_mode}"
    out_dir = base / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"variant       : {args.variant}")
    print(f"npz (any task): {npz}")
    print(f"classify csv  : {classify_csv}")
    print(f"out dir       : {out_dir}")

    print("\n[1] Loading model + unit_layout")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device)
    model.eval()
    arch = detect_arch(model)
    n_layers = getattr(model.config, "num_hidden_layers",
                       getattr(model.config, "n_layer", None))
    param_filter = get_param_filter(args.scope, n_layers)
    unit_layout = build_unit_layout(model, param_filter=param_filter,
                                    arch=arch, include_mlp=True)
    unit_by_id  = {u.unit_id: u for u in unit_layout}
    param_segs  = build_param_seg_map(model, param_filter)

    # attn-only filter (default)
    if args.attn_only:
        attn_unit_ids = sorted([uid for uid, u in unit_by_id.items()
                                 if u.unit_type == "attn_head"])
        all_unit_ids  = attn_unit_ids
        print(f"   attn_only=True  -> restricting to {len(attn_unit_ids)} attn heads "
              f"(of {len(unit_by_id)} total)")
    else:
        all_unit_ids = sorted(unit_by_id.keys())
        print(f"   attn_only=False -> using all {len(all_unit_ids)} units (incl. MLP)")

    # Pre-compute unit -> param map (vectorized knockout)
    print("   pre-computing unit->param map for vectorized knockout...")
    unit_param_map = build_unit_param_map(unit_by_id, param_segs)
    model_named    = dict(model.named_parameters())
    print(f"   unit_layout: {len(unit_layout)} units, scope={args.scope}")

    # Load per-unit subspaces:
    #   per-task (default): load 12 npz files; each task is evaluated with
    #     its own U_u, which avoids donor-task bias.
    #   single donor (--single_donor_npz): one npz used for every task.
    if args.single_donor_npz:
        per_task_u = False
        print(f"\n[2] (single donor mode) loading {args.single_donor_npz}")
        donor_subs = load_task_subspaces(args.single_donor_npz)
        task_unit_subs = {t: donor_subs for t in TASKS}
        print(f"   {len(donor_subs)} units (donor), used for all 12 tasks")
    else:
        per_task_u = True
        print(f"\n[2] (per-task U mode) loading 12 task npz files")
        npz_filename = ("weighted_unit_subspaces.npz" if args.variant == "consistency_share"
                        else "weighted_unit_subspaces_sigma.npz")
        task_unit_subs = {}
        for tname, subdir in TASK_NPZ_DIRS.items():
            p = REPO_ROOT / "results" / subdir / npz_filename
            if not p.exists():
                sys.exit(f"[ERROR] missing {p}")
            task_unit_subs[tname] = load_task_subspaces(str(p))
            print(f"   {tname:<16s} <- {subdir}/{npz_filename} ({len(task_unit_subs[tname])} units)")

    # Parse classification
    print(f"\n[3] Parsing unit sets from {classify_csv.name}")
    sets = parse_classification(classify_csv, allowed_uids=set(all_unit_ids))
    if args.sanity:
        # restrict to a small subset for fast verification
        keep = ["Universal", "GT_specialized_top4", "SVA_specialized_4"]
        sets = {k: v for k, v in sets.items() if k in keep}
        print(f"   --sanity: restricted to sets {list(sets.keys())}")
    for k, v in sets.items():
        print(f"   {k:<26s} : n={len(v)}    {v}")

    # Random sets
    sizes = sorted({len(v) for v in sets.values()})
    print(f"\n[4] Generating random baselines for sizes {sizes} x n_random={args.n_random}")
    random_sets = make_random_sets(args.n_random, sizes, all_unit_ids)

    # Load all task batches + build metric_fns
    print(f"\n[5] Loading 12 task batches (n={args.n} each)")
    task_batches = {}
    metric_fns   = {}
    for name, info in TASKS.items():
        gt_mode = "prob_diff" if info["schema"] == "greater_than" else "prob_diff"
        batch = load_ioi(info["csv"], n=args.n, gt_mode=gt_mode, tokenizer=tokenizer)
        task_batches[name] = batch
        metric_fns[name]   = build_metric_fn(info["schema"], batch, model, tokenizer, args.device)
        print(f"   {name:<16s} n={len(batch.clean):3d}  task={batch.task}")

    # Clean baseline
    print(f"\n[6] Computing clean (no knockout) baselines")
    L_clean = {}
    for name in TASKS:
        L_clean[name] = run_eval(model, tokenizer, task_batches[name],
                                  metric_fns[name], args.device)
    print("   clean metrics:", {k: f"{v:.3f}" for k, v in L_clean.items()})

    # Helper: knock out uid set with task-specific U, eval task, restore
    def eval_set_on_task(unit_ids, task_name, u_subs=None):
        if u_subs is None:
            u_subs = task_unit_subs[task_name]
        saved, n_ko = knockout_set(model, param_segs, unit_by_id, u_subs, unit_ids,
                                    unit_param_map=unit_param_map, model_named=model_named,
                                    mode=args.knockout_mode)
        L = run_eval(model, tokenizer, task_batches[task_name],
                      metric_fns[task_name], args.device)
        restore_affected_params(model, saved)
        return L, n_ko

    raw_results: Dict[str, Dict[str, float]] = {"_clean": L_clean}
    if args.sanity:
        random_results: Dict[int, List[Dict[str, float]]] = {}
    else:
        random_results = {size: [{} for _ in range(args.n_random)] for size in sizes}
    t0 = time.time()

    if args.knockout_mode == "block_full":
        # Task-major loop: load blocks once per task, compute Q per unit, eval all sets
        print(f"\n[7] block_full mode: task-major loop ({len(TASKS)} tasks x ({len(sets)} sets + {len(sizes)*args.n_random if not args.sanity else 0} random))")
        # Initialize empty result dicts
        for set_name in sets:
            raw_results[set_name] = {}
        for tname in TASKS:
            print(f"\n   -- task: {tname} -- computing block-derived U for relevant units...")
            task_dir = REPO_ROOT / "results" / TASK_NPZ_DIRS[tname]
            # Collect relevant uids (union of all sets + all random)
            all_uids: set = set()
            for uid_list in sets.values():
                all_uids.update(uid_list)
            if not args.sanity:
                for size in sizes:
                    for draw in random_sets[size]:
                        all_uids.update(draw)
            block_subs = compute_block_derived_U(task_dir, task_unit_subs[tname],
                                                   unit_by_id, sorted(all_uids))
            print(f"      built block-derived U for {len(block_subs)} units; running evals...")

            # Sets
            for set_name, unit_ids in sets.items():
                L, n_ko = eval_set_on_task(unit_ids, tname, u_subs=block_subs)
                raw_results[set_name][tname] = L

            # Random
            if not args.sanity:
                for size in sizes:
                    for s, unit_ids in enumerate(random_sets[size]):
                        L, _ = eval_set_on_task(unit_ids, tname, u_subs=block_subs)
                        random_results[size][s][tname] = L

            del block_subs
            torch.cuda.empty_cache()
            print(f"      done. cumulative={time.time() - t0:.1f}s")

    else:
        # Set-major loop (subspace, head_zero)
        print(f"\n[7] Knockout per (unit set x task), per_task_u={per_task_u}, mode={args.knockout_mode}")
        for set_name, unit_ids in sets.items():
            L_S = {}
            for tname in TASKS:
                L_S[tname], n_ko = eval_set_on_task(unit_ids, tname)
            raw_results[set_name] = L_S
            print(f"   {set_name:<26s} n_ko={n_ko:>3}  cumulative={time.time() - t0:.1f}s")

        if args.sanity:
            print("\n[8] (sanity) skipping random baselines")
        else:
            print(f"\n[8] Random baselines ({len(sizes)} sizes x {args.n_random} draws x 12 tasks)")
            for size in sizes:
                for s, unit_ids in enumerate(random_sets[size]):
                    for tname in TASKS:
                        L, _ = eval_set_on_task(unit_ids, tname)
                        random_results[size][s][tname] = L
                    print(f"   size={size:>3}  draw={s+1}/{args.n_random}  cumulative={time.time()-t0:.1f}s")

    # Compute lift
    print(f"\n[9] Computing Lift = F_S - F_random (relative drop)")

    def faith(L_clean_t, L_t):
        eps = 1e-6
        return (L_clean_t - L_t) / max(abs(L_clean_t), eps)

    lift_rows = []
    for set_name, unit_ids in sets.items():
        size = len(unit_ids)
        F_S = {t: faith(L_clean[t], raw_results[set_name][t]) for t in TASKS}
        if random_results.get(size):
            F_R_mean = {t: float(np.mean([faith(L_clean[t], r[t]) for r in random_results[size]])) for t in TASKS}
            F_R_std  = {t: float(np.std ([faith(L_clean[t], r[t]) for r in random_results[size]])) for t in TASKS}
        else:
            F_R_mean = {t: 0.0 for t in TASKS}
            F_R_std  = {t: 0.0 for t in TASKS}
        lift = {t: F_S[t] - F_R_mean[t] for t in TASKS}
        lift_rows.append({"set": set_name, "size": size,
                          **{f"lift_{t}": lift[t] for t in TASKS},
                          **{f"F_S_{t}":  F_S[t]  for t in TASKS},
                          **{f"F_R_{t}":  F_R_mean[t] for t in TASKS},
                          **{f"F_Rstd_{t}": F_R_std[t] for t in TASKS}})

    # Save lift_matrix.csv
    lift_csv = out_dir / "lift_matrix.csv"
    cols = ["set", "size"] + [f"lift_{t}" for t in TASKS]
    with open(lift_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in lift_rows:
            w.writerow([r["set"], r["size"]] + [f"{r[f'lift_{t}']:.4f}" for t in TASKS])
    print(f"   [saved] {lift_csv}")

    # raw_metrics.csv
    raw_csv = out_dir / "raw_metrics.csv"
    cols_raw = ["condition"] + list(TASKS)
    with open(raw_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols_raw)
        for cond, L in raw_results.items():
            w.writerow([cond] + [f"{L[t]:.4f}" for t in TASKS])
        for size, drawlist in random_results.items():
            for s, L in enumerate(drawlist):
                w.writerow([f"random_size{size}_draw{s}"] + [f"{L[t]:.4f}" for t in TASKS])
    print(f"   [saved] {raw_csv}")

    # concentration.csv: per-set, mean Lift on family tasks vs non-family
    fam_for_set = {
        "Universal":            None,        # universal: should be uniform
        "GT_specialized_22":    "GT",
        "GT_specialized_top4":  "GT",
        "SVA_specialized_4":    "SVA",
        "COUNTRY_broad_2":      "COUNTRY",
        "Mixed_22_control":     None,
    }
    conc_rows = []
    for r in lift_rows:
        fam = fam_for_set.get(r["set"])
        in_fam, out_fam = [], []
        for t in TASKS:
            l = r[f"lift_{t}"]
            if fam is not None and TASK_TO_FAMILY.get(t) == fam:
                in_fam.append(l)
            else:
                out_fam.append(l)
        in_mean  = float(np.mean(in_fam)) if in_fam  else float("nan")
        out_mean = float(np.mean(out_fam)) if out_fam else float("nan")
        ratio = (in_mean / out_mean) if (out_mean and out_mean > 1e-6) else float("nan")
        conc_rows.append({
            "set": r["set"], "size": r["size"], "expected_family": fam or "—",
            "lift_in_family":  f"{in_mean:.4f}",
            "lift_out_family": f"{out_mean:.4f}",
            "concentration":   f"{ratio:.3f}" if not np.isnan(ratio) else "—",
        })
    conc_csv = out_dir / "concentration.csv"
    with open(conc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conc_rows[0].keys()))
        w.writeheader(); w.writerows(conc_rows)
    print(f"   [saved] {conc_csv}")

    # summary.txt
    summary = out_dir / "summary.txt"
    with open(summary, "w") as f:
        f.write(f"Cross-task knockout validation\n{'='*60}\n\n")
        f.write(f"variant:    {args.variant}\n")
        f.write(f"n_prompts:  {args.n}\n")
        f.write(f"n_random:   {args.n_random}\n")
        f.write(f"device:     {args.device}\n\n")
        f.write("Lift matrix (rows = set, cols = task):\n")
        f.write(f"{'set':<26s} | " + " ".join(f"{t[:8]:>8s}" for t in TASKS) + "\n")
        f.write("-" * (28 + 9 * len(TASKS)) + "\n")
        for r in lift_rows:
            f.write(f"{r['set']:<26s} | " + " ".join(f"{r[f'lift_{t}']:>8.3f}" for t in TASKS) + "\n")
        f.write("\nConcentration (lift_in_family / lift_out_family):\n")
        for c in conc_rows:
            f.write(f"  {c['set']:<26s}  fam={c['expected_family']:<8s}  "
                    f"in={c['lift_in_family']}  out={c['lift_out_family']}  ratio={c['concentration']}\n")
    print(f"   [saved] {summary}")

    print(f"\n[done] total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
