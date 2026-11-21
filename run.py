"""Main SCA pipeline for one task.

Phases: 0-2 gradient collection, NSR diagnostics and sign consistency (SC);
3 unit layout and block finding; 4 block-wise SVD; 5b block-derived weighted
unit subspaces and the ranked_units.csv unit ranking.

Usage:
    python run.py --config configs/gpt2_small_ioi_strong.yaml
    python run.py --config configs/gpt2_small_ioi_strong.yaml --stop_after blocks
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from utils.io_utils     import load_config, get_output_dir, PhaseCache
from utils.model_utils  import load_model_and_tokenizer, get_param_filter, count_params
from utils.paper_style  import apply_paper_style_globally

# Paper figure style for all plots.
apply_paper_style_globally("single")
from data.loader       import (
    load_records, load_records_cap, load_records_gp,
    load_records_ioi_v3, load_records_ioi_strong, load_records_ioi_sampled,
    load_records_gb_sampled, load_records_gp_sampled,
    load_records_person_country,
    load_records_country_cap, load_records_gt,
    load_records_sva, build_sva_verb_ids, make_sva_prob_diff_loss_fn,
    build_sva_is_are_ids, make_sva_is_are_logit_diff_loss_fn,
    build_gt_year_ids, make_gt_prob_diff_loss_fn,
    make_gt_boundary_logit_diff_loss_fn,
)

from spectral.gradient_collector import collect, CollectionResult
from spectral.diagnostics        import run_diagnostics, plot_all
from spectral.block_finder       import (
    detect_arch, build_unit_layout, find_blocks, find_blocks_by_layer,
    plot_unit_heatmap, BlockResult, UnitInfo,
)
from spectral.svd_analysis       import run_block_svd, plot_svd_spectra
from spectral.test_gradient_sanity import run_sanity_checks
from spectral.gradient_heatmap     import run_gradient_heatmap
from spectral.plot_sigma_dist      import plot_sigma_distributions
from spectral.plot_block_dist      import plot_block_importance_dist
from cross_task.weighted_unit_svd  import (
    compute_task_subspaces_from_blocks,
    save_task_subspaces as save_weighted_subspaces,
)


# Layer consistency helper

def compute_layer_consistency(
    grad_avg: "torch.Tensor",
    unit_layout,
) -> dict:
    """
    Per-layer gradient consistency from grad_avg (N, D), computed separately
    for attn units and MLP units to avoid MLP dominating due to larger d_u.

    Returns dict[(layer, unit_type_group) -> float]
      where unit_type_group is "attn_head" or "mlp".
    """
    import torch.nn.functional as F
    from collections import defaultdict

    ul = unit_layout if isinstance(unit_layout, list) else list(unit_layout.values())

    layer_type_indices: dict = defaultdict(list)
    for unit in ul:
        utype = "attn_head" if unit.unit_type == "attn_head" else "mlp"
        layer_type_indices[(unit.layer, utype)].extend(unit.get_indices().tolist())

    G = grad_avg.float()   # (N, D)
    result = {}
    for (layer, utype), idxs in sorted(layer_type_indices.items()):
        idx_t  = torch.tensor(idxs, dtype=torch.long)
        G_l    = G[:, idx_t]
        G_l_n  = F.normalize(G_l, p=2, dim=1)
        mean_l = F.normalize(G_l_n.mean(dim=0, keepdim=True), p=2, dim=1)
        cosim  = (G_l_n @ mean_l.T).squeeze(1)
        result[(layer, utype)] = float(cosim.mean().item())
    return result


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--stop_after",
        choices=["collect", "diagnostics", "blocks", "svd", "unit_svd"],
        default=None,
    )
    p.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run gradient sanity checks (subset of records) before main pipeline.",
    )
    p.add_argument(
        "--sanity_only",
        action="store_true",
        help="After sanity check, exit immediately (implies --sanity_check).",
    )
    return p.parse_args()


# Main pipeline

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    outdir = get_output_dir(cfg)
    cache  = PhaseCache(outdir)
    do_plot = cfg["output"].get("plot", True)

    print(f"\n{'='*60}")
    print(f"  Spectral Circuit Analysis")
    print(f"  config : {args.config}")
    print(f"  output : {outdir}")
    print(f"{'='*60}\n")

    # Model
    print("[Setup] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name_or_path"],
        cfg["model"]["torch_dtype"],
        cfg["model"]["device"],
    )

    # Detect architecture
    arch     = detect_arch(model)
    n_layers = getattr(model.config, "num_hidden_layers",
                       getattr(model.config, "n_layer", None))
    print(f"[Setup] arch={arch.name}, n_layers={n_layers}, "
          f"n_heads={arch.n_heads}, n_kv_heads={arch.n_kv_heads}, "
          f"d_model={arch.d_model}, GQA={arch.uses_gqa}")

    scope        = cfg["params"]["scope"]
    include_mlp  = cfg["params"].get("include_mlp", True)
    param_filter = get_param_filter(scope, n_layers)
    param_info   = count_params(model, param_filter)
    D            = param_info["total_D"]
    print(f"[Setup] scope={scope}, include_mlp={include_mlp}, D={D:,}")

    # Data
    print("[Setup] Loading records...")
    data_format = cfg["data"].get("format", "ioi")
    _M = cfg["data"].get("M", None)   # None = use all paraphrases per instance
    if data_format == "cap":
        records = load_records_cap(
            cfg["data"]["path"],
            cfg["data"]["source"],
            cfg["data"]["N"],
        )
    elif data_format == "gp":
        records = load_records_gp(
            cfg["data"]["path"],
            cfg["data"]["source"],
            cfg["data"]["N"],
        )
    elif data_format == "ioi_v3":
        records = load_records_ioi_v3(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "ioi_strong":
        records = load_records_ioi_strong(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "ioi_sampled":
        records = load_records_ioi_sampled(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "gb_sampled":
        records = load_records_gb_sampled(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "gp_sampled":
        records = load_records_gp_sampled(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "person_country":
        records = load_records_person_country(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "country_cap":
        records = load_records_country_cap(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "gt":
        records = load_records_gt(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
        )
    elif data_format == "sva":
        records = load_records_sva(
            cfg["data"]["path"],
            N=cfg["data"]["N"],
            M=_M,
            start=int(cfg["data"].get("start", 0)),
        )
    else:
        records = load_records(
            cfg["data"]["path"],
            cfg["data"]["source"],
            cfg["data"]["N"],
        )
    N = len(records)
    M = len(records[0]["paraphrase_queries"])
    print(f"[Setup] N={N} instances, M={M} paraphrases each")

    # Optional sanity check
    if args.sanity_check or args.sanity_only:
        san_max_n = min(N, cfg.get("sanity", {}).get("max_n", 10))
        sanity_res = run_sanity_checks(
            model, tokenizer, records, param_filter, cfg, max_n=san_max_n,
        )
        if sanity_res["failed"]:
            print(f"[Sanity] {len(sanity_res['failed'])} test(s) failed; "
                  "continuing to main pipeline (use --sanity_only to abort)")
        if args.sanity_only:
            print("\nStopped after sanity check.")
            return

    # Phase 0/1/2: gradient collection + diagnostics
    print(f"\n{'-'*50}")
    print("Phase 0/1/2: Gradient Collection + Diagnostics")
    print(f"{'-'*50}")

    # Task-specific loss_fn. Built outside the cache branch because Phase 4
    # recomputes gradients with it when grad_avg was not cached.
    loss_fn = None
    if data_format == "gt":
        # cfg.data.loss_variant: "prob_diff" (default) or "boundary_logit_diff"
        #   prob_diff:           -(Σ logsumexp p(YY > YY₀) − Σ logsumexp p(YY ≤ YY₀))
        #                        over all 100 YY tokens; matches eap-ig-faithfulness
        #                        prob_diff_greater_than metric.
        #   boundary_logit_diff: -(logit[YY₀+1] − logit[YY₀]) on the 2-token boundary
        #                        pair; sharper per-instance signal. Use with knockout
        #                        gt_mode=boundary_logit_diff and EAP task=greater-than-boundary.
        year_ids = build_gt_year_ids(tokenizer)
        loss_variant = cfg["data"].get("loss_variant", "prob_diff")
        if loss_variant == "prob_diff":
            loss_fn = make_gt_prob_diff_loss_fn(year_ids)
            print(f"[Loss] gt prob-diff over {year_ids.numel()} YY tokens "
                  f"(min_id={int(year_ids.min())}, max_id={int(year_ids.max())})")
        elif loss_variant == "boundary_logit_diff":
            loss_fn = make_gt_boundary_logit_diff_loss_fn(year_ids)
            print(f"[Loss] gt boundary logit-diff: -(logit[YY₀+1] - logit[YY₀]) "
                  f"over {year_ids.numel()} YY token ids")
        else:
            raise ValueError(
                f"Unknown gt loss_variant={loss_variant!r}; "
                "expected 'prob_diff' or 'boundary_logit_diff'"
            )

    elif data_format == "sva":
        # cfg.data.loss_variant: "verb_set" (default) or "is_are"
        #   verb_set: -(Σ logsumexp(target verb-set) − Σ logsumexp(other set))
        #             over all single-token sing/plur verbs from verb_list_csv;
        #             matches eap-ig-faithfulness prob_diff_sva_at_last_pos.
        #   is_are:   -(logit[target] − logit[other]) restricted to (' is', ' are').
        #             Sharper but narrower signal; useful when the verb-set
        #             SVD spectrum collapses to one dominant direction.
        loss_variant = cfg["data"].get("loss_variant", "verb_set")
        if loss_variant == "is_are":
            is_id, are_id = build_sva_is_are_ids(tokenizer)
            loss_fn = make_sva_is_are_logit_diff_loss_fn(is_id, are_id)
            print(f"[Loss] sva is/are logit-diff: is_id={is_id}, are_id={are_id}")
        elif loss_variant == "verb_set":
            verb_list_csv = cfg["data"].get(
                "verb_list_csv",
                "eap_data/sva/combined_verb_list.csv",
            )
            sing_ids, plur_ids = build_sva_verb_ids(tokenizer, verb_list_csv)
            loss_fn = make_sva_prob_diff_loss_fn(sing_ids, plur_ids)
            print(f"[Loss] sva verb-set prob-diff: {sing_ids.numel()} singular tokens, "
                  f"{plur_ids.numel()} plural tokens (verb_list={verb_list_csv})")
        else:
            raise ValueError(
                f"Unknown sva loss_variant={loss_variant!r}; "
                "expected 'verb_set' or 'is_are'"
            )

    if cache.exists("gram_matrix") and cache.exists("sign_accum"):
        print("[Cache HIT] Loading gram_matrix, sign_accum...")
        gram     = cache.load_tensor("gram_matrix")
        sign_acc = cache.load_tensor("sign_accum")
        # grad_avg may not be cached
        grad_avg = cache.load_tensor("grad_avg") if cache.exists("grad_avg") else None

        col_result = CollectionResult(
            gram_matrix=gram, sign_accum=sign_acc,
            grad_avg=grad_avg,
            M=M, N=N, D=D, elapsed_sec=0,
            grad_norms=torch.zeros(N, M),
        )
        if grad_avg is None:
            print("  [Note] grad_avg not cached; block finding will fall back to sc")
    else:
        print("[Cache MISS] Running gradient collection...")
        col_result = collect(model, tokenizer, records, param_filter, cfg,
                             loss_fn=loss_fn)

        if cfg["output"]["save_intermediate"]:
            cache.save_tensor(col_result.gram_matrix, "gram_matrix")
            cache.save_tensor(col_result.sign_accum,  "sign_accum")
            cache.save_tensor(col_result.grad_norms,  "grad_norms")
            if col_result.grad_avg is not None:
                cache.save_tensor(col_result.grad_avg, "grad_avg")
                print("  [Saved] grad_avg")

    # Gradient heatmap (on unless cfg.heatmap.enabled is false). Reruns one
    # forward+backward per paraphrase, so it is expensive for large N, M.
    if cfg.get("heatmap", {}).get("enabled", True) and do_plot:
        print(f"\n{'-'*50}")
        print("Gradient heatmap (per-layer norm + cosine sim)")
        print(f"{'-'*50}")
        run_gradient_heatmap(
            model, tokenizer, records, param_filter, cfg,
            n_layers=n_layers,
            arch_name=arch.name,
            save_dir=str(outdir / "plots"),
            split_consistency=cfg["heatmap"].get("split_consistency", True),
            split_preset=cfg["heatmap"].get("split_preset", "single"),
            loss_fn=loss_fn,
        )

    if args.stop_after == "collect":
        print("\nStopped after collection."); return

    # Diagnostics
    if cache.exists("sc") and cache.exists("diagnostics_summary"):
        print("[Cache HIT] Loading diagnostics...")
        sc          = cache.load_tensor("sc")
        diag_json   = cache.load_json("diagnostics_summary")
        from spectral.diagnostics import DiagnosticsResult
        # Use the threshold that was actually computed (accounts for top_pct)
        actual_threshold = diag_json["sc_threshold"]
        semantic_mask = sc >= actual_threshold
        diag_result = DiagnosticsResult(
            nsr=diag_json["nsr"],
            d_intra_mean=diag_json["d_intra_mean"],
            d_inter_mean=diag_json["d_inter_mean"],
            gram_matrix=col_result.gram_matrix,
            sc=sc, sc_mean=diag_json["sc_mean"],
            sc_threshold=actual_threshold,
            n_semantic_dims=int(semantic_mask.sum()),
            n_noise_dims=D - int(semantic_mask.sum()),
            semantic_mask=semantic_mask,
            N=N, M=M, D=D,
        )
        print(f"  NSR={diag_result.nsr:.4f}  SC_mean={diag_result.sc_mean:.4f}")
    else:
        print("[Cache MISS] Running diagnostics...")
        diag_result = run_diagnostics(
            col_result,
            sc_threshold = cfg["sc"]["threshold"],
            sc_top_pct   = cfg["sc"].get("top_pct", None),
        )
        if cfg["output"]["save_intermediate"]:
            cache.save_tensor(diag_result.sc,           "sc")
            cache.save_tensor(diag_result.semantic_mask,"semantic_mask")
            cache.save_json({
                "nsr":             diag_result.nsr,
                "d_intra_mean":    diag_result.d_intra_mean,
                "d_inter_mean":    diag_result.d_inter_mean,
                "sc_mean":         diag_result.sc_mean,
                "sc_threshold":    diag_result.sc_threshold,
                "n_semantic_dims": diag_result.n_semantic_dims,
                "n_noise_dims":    diag_result.n_noise_dims,
            }, "diagnostics_summary")
    if do_plot:
        plot_all(diag_result, save_dir=str(outdir / "plots"))

    if args.stop_after == "diagnostics":
        print("\nStopped after diagnostics."); return

    # Phase 3: unit layout + block finding
    print(f"\n{'-'*50}")
    print("Phase 3: Unit Layout + Block Finding")
    print(f"{'-'*50}")

    # The unit layout is cheap, so it is always rebuilt rather than cached.
    unit_layout = build_unit_layout(
        model, param_filter, arch,
        include_mlp=include_mlp,
    )

    if cache.exists("blocks") and cache.exists("unit_sim"):
        print("[Cache HIT] Loading block result...")
        blocks_data = [b for b in cache.load_json("blocks")
                       if "unit_ids" in b]   # skip the trailing _meta entry
        unit_sim    = cache.load_tensor("unit_sim")
        blocks      = [b["unit_ids"] for b in blocks_data]
        block_indices = [
            torch.cat([unit_layout[uid].get_indices() for uid in uids])
            for uids in blocks
        ]
        # threshold is stored in the _meta entry
        meta = next((b for b in cache.load_json("blocks")
                     if "_meta" in b), {}).get("_meta", {})
        block_result = BlockResult(
            unit_sim         = unit_sim,
            blocks           = blocks,
            block_indices    = block_indices,
            unit_layout      = unit_layout,
            n_units          = len(unit_layout),
            K                = len(blocks),
            threshold_used   = meta.get("threshold_used",   0.0),
            threshold_source = meta.get("threshold_source", "cached"),
        )
    else:
        print("[Cache MISS] Running block finding...")
        block_strategy = cfg["block"].get("strategy", "similarity")
        if block_strategy in ("layer_attn_mlp", "layer_combined"):
            print(f"[BlockFinder] strategy={block_strategy} (layer-stratified, no similarity)")
            block_result = find_blocks_by_layer(
                unit_layout = unit_layout,
                strategy    = block_strategy,
                save_dir    = str(outdir / "plots") if do_plot else None,
            )
        else:
            block_result = find_blocks(
                unit_layout   = unit_layout,
                grad_avg      = col_result.grad_avg,
                sc            = diag_result.sc,
                sign_accum    = col_result.sign_accum,
                M             = col_result.M,
                semantic_mask = diag_result.semantic_mask,
                sim_threshold = cfg["block"].get("sim_threshold", None),
                min_block_size= cfg["block"]["min_block_size"],
                min_dims      = cfg["block"].get("min_dims", 100),
                save_dir      = str(outdir / "plots") if do_plot else None,
            )

        # blocks.json is small and needed by downstream plots, so it is saved
        # regardless of save_intermediate.
        cache.save_json(
            [{
                "block_id":       k,
                "unit_ids":       b,
                "unit_labels":    [unit_layout[uid].label     for uid in b],
                "unit_types":     list({unit_layout[uid].unit_type for uid in b}),
                "layers":         sorted({unit_layout[uid].layer   for uid in b}),
                "d_k":            int(block_result.block_indices[k].shape[0]),
            } for k, b in enumerate(block_result.blocks)] +
            [{"_meta": {
                "threshold_used":   block_result.threshold_used,
                "threshold_source": block_result.threshold_source,
            }}],
            "blocks",
        )
        if cfg["output"]["save_intermediate"]:
            cache.save_tensor(block_result.unit_sim, "unit_sim")

    if do_plot:
        plot_unit_heatmap(block_result, save_dir=str(outdir / "plots"))

    if args.stop_after == "blocks":
        print("\nStopped after block finding."); return

    # Phase 4: block-wise SVD
    print(f"\n{'-'*50}")
    print(f"Phase 4: Block-wise SVD (K={block_result.K}, η={cfg['svd']['eta']})")
    print(f"{'-'*50}")

    _mean_center_now = bool(cfg["svd"].get("mean_center", False))
    _svd_cached = (
        cache.exists("svd_summary")
        and all(cache.exists(f"U_rho_block{k}")        for k in range(block_result.K))
        and all(cache.exists(f"sigma_block{k}")         for k in range(block_result.K))
        and all(cache.exists(f"global_indices_block{k}") for k in range(block_result.K))
    )
    if _svd_cached:
        _cached_mc = bool(cache.load_json("svd_summary")[0].get("mean_center", False))
        if _cached_mc != _mean_center_now:
            print(f"[Cache] svd_summary mean_center={_cached_mc} != "
                  f"cfg.svd.mean_center={_mean_center_now}; invalidating Phase 4 cache")
            _svd_cached = False

    if _svd_cached:
        print("[Cache HIT] Loading SVD results...")
        svd_summary = cache.load_json("svd_summary")
        svd_results = []
        ul = unit_layout
        for entry in svd_summary:
            k        = entry["block_id"]
            unit_ids = entry["unit_ids"]
            U_rho    = cache.load_tensor(f"U_rho_block{k}")
            sigma    = cache.load_tensor(f"sigma_block{k}")
            g_idx    = cache.load_tensor(f"global_indices_block{k}")
            rho      = entry["effective_rank"]
            cumvar   = torch.cumsum(sigma**2, dim=0) / (sigma**2).sum().clamp(min=1e-12)
            from spectral.svd_analysis import BlockSVDResult
            svd_results.append(BlockSVDResult(
                block_id        = k,
                unit_ids        = unit_ids,
                unit_labels     = [ul[uid].label     for uid in unit_ids],
                unit_types      = [ul[uid].unit_type for uid in unit_ids],
                layers          = sorted({ul[uid].layer for uid in unit_ids}),
                d_k             = entry["d_k"],
                N               = entry["N"],
                singular_values = sigma,
                explained_var   = cumvar,
                effective_rank  = rho,
                eta             = entry["eta"],
                U_rho           = U_rho,
                redundancy_ratio= entry["redundancy_ratio"],
                global_indices  = g_idx,
            ))
        print(f"  Loaded {len(svd_results)} blocks from cache.")
    else:
        print("[Cache MISS] Running block SVD...")
        svd_results = run_block_svd(
            block_result = block_result,
            records      = records,
            cfg          = cfg,
            grad_avg     = col_result.grad_avg,
            model        = model,
            tokenizer    = tokenizer,
            param_filter = param_filter,
            loss_fn      = loss_fn,
        )

        for res in svd_results:
            cache.save_tensor(res.U_rho,           f"U_rho_block{res.block_id}")
            cache.save_tensor(res.singular_values,  f"sigma_block{res.block_id}")
            cache.save_tensor(res.global_indices,   f"global_indices_block{res.block_id}")

        cache.save_json(
            [{
                "block_id":         r.block_id,
                "unit_ids":         block_result.blocks[r.block_id],
                "d_k":              r.d_k,
                "N":                r.N,
                "effective_rank":   r.effective_rank,
                "eta":              r.eta,
                "redundancy_ratio": r.redundancy_ratio,
                "mean_center":      _mean_center_now,
            } for r in svd_results],
            "svd_summary",
        )

    if do_plot:
        plot_svd_spectra(svd_results, save_dir=str(outdir / "plots"))
        plot_sigma_distributions(outdir, n_layers=n_layers)

    if args.stop_after == "svd":
        print("\nStopped after block SVD."); return


    # Phase 5b: block-derived weighted unit subspaces
    weighted_cfg = cfg.get("weighted_unit_svd", {})
    if weighted_cfg.get("enabled", True):
        print(f"\n{'-'*50}")
        print(f"Phase 5b: Block-derived weighted unit subspaces")
        print(f"{'-'*50}")

        weighted_path = outdir / "weighted_unit_subspaces.npz"
        sigma_path    = outdir / "weighted_unit_subspaces_sigma.npz"  # sibling: sigma_weighted view
        csv_path = outdir / "ranked_units.csv"
        imp_mode = weighted_cfg.get("importance_mode", "sigma_weighted")
        _use_cache = (
            imp_mode == "sigma_weighted" and   # non-default mode always recomputes
            weighted_path.exists() and
            cache.exists("weighted_unit_svd_summary") and
            csv_path.exists() and
            _svd_cached                       # rerun if Phase 4 was recomputed
        )
        if _use_cache:
            print("[Cache HIT] Loading weighted_unit_subspaces...")
            from cross_task.weighted_unit_svd import (
                load_task_subspaces as load_weighted,
                derive_sigma_subs,
            )
            weighted_subs = load_weighted(str(weighted_path))
            print(f"  Loaded {len(weighted_subs)} units from cache.")
            # Derive the sigma_weighted sibling if missing.
            if not sigma_path.exists():
                print("  [sigma sibling] missing; deriving from primary npz...")
                save_weighted_subspaces(derive_sigma_subs(weighted_subs), str(sigma_path))
                print(f"  [saved] {sigma_path}")
        else:
            print("[Cache MISS] Running Phase 5b...")
            # Convert BlockSVDResults to the dict format used by weighted_unit_svd.
            block_results_dict = {
                r.block_id: {
                    'U':              r.U_rho.numpy(),
                    'sigma':          r.singular_values[:r.effective_rank].numpy(),
                    'unit_ids':       block_result.blocks[r.block_id],
                    'global_indices': r.global_indices.numpy(),
                }
                for r in svd_results
            }

            weighted_subs = compute_task_subspaces_from_blocks(
                block_results = block_results_dict,
                unit_layout   = list(unit_layout.values())
                                if isinstance(unit_layout, dict)
                                else unit_layout,
                eta_inner     = weighted_cfg.get("eta_inner", cfg["svd"]["eta"]),
                verbose       = True,
            )

            # Snapshot the sigma_weighted importances (||M_u||_F²) before any
            # override, for the sibling npz used by cross-task analysis.
            from cross_task.weighted_unit_svd import snapshot_sigma_subs
            weighted_subs_sigma = snapshot_sigma_subs(weighted_subs)

            # Optional importance override.
            # residual_topk: drop the top-k singular directions, which for
            # prob_diff tasks (GT, SVA) align with the common gradient direction
            # (close to the magnitude baseline). Importance then measures the
            # remaining anisotropy, decoupled from per-unit gradient norm.
            if imp_mode == "residual_topk":
                top_k_skip = int(weighted_cfg.get("top_k_skip", 1))
                print(f"\n[Phase 5b] importance_mode=residual_topk  "
                      f"(top_k_skip={top_k_skip})")
                from cross_task.weighted_unit_svd import _build_unit_to_block_index
                ul_list = (list(unit_layout.values())
                           if isinstance(unit_layout, dict) else unit_layout)
                unit_to_blk = _build_unit_to_block_index(block_results_dict, ul_list)

                n_overridden = 0
                n_skipped    = 0    # blocks with rho_B <= top_k_skip have no residual
                for uid, sub in weighted_subs.items():
                    if uid not in unit_to_blk:
                        continue
                    bid, local_idx = unit_to_blk[uid]
                    U_B   = block_results_dict[bid]['U']      # (d_block, rho_B)
                    sigma = block_results_dict[bid]['sigma']  # (rho_B,)
                    d_block = U_B.shape[0]
                    rho_B   = U_B.shape[1]
                    if rho_B <= top_k_skip:
                        new_imp = 0.0
                        n_skipped += 1
                    else:
                        # M_res[i, j] = U_B[local_idx[i], top_k_skip+j] *
                        #               sigma[top_k_skip+j] / sqrt(d_block)
                        sigma_norm = sigma[top_k_skip:] / np.sqrt(d_block)
                        M_res = (U_B[local_idx, top_k_skip:].astype(np.float32)
                                 * sigma_norm[None, :].astype(np.float32))
                        new_imp = float((M_res ** 2).sum())
                    sub['importance']         = new_imp
                    sub['meta']['importance'] = new_imp
                    n_overridden += 1
                if n_skipped:
                    print(f"  overrode {n_overridden} units; "
                          f"{n_skipped} units in low-rank blocks (rho<={top_k_skip}) "
                          f"have importance=0")
                else:
                    print(f"  overrode {n_overridden} units (all blocks rho > {top_k_skip})")

            elif imp_mode == "consistency_share":
                print(f"\n[Phase 5b] importance_mode=consistency_share")
                if col_result.grad_avg is None:
                    print("  [WARN] grad_avg not available; falling back to sigma_weighted")
                    imp_mode = "sigma_weighted"
                else:
                    from cross_task.weighted_unit_svd import _build_unit_to_block_index
                    ul_list = (list(unit_layout.values())
                               if isinstance(unit_layout, dict) else unit_layout)

                    layer_cons = compute_layer_consistency(col_result.grad_avg, ul_list)
                    print("  layer consistency (attn | mlp):")
                    for l in sorted({k[0] for k in layer_cons}):
                        a = layer_cons.get((l, "attn_head"), float("nan"))
                        m = layer_cons.get((l, "mlp"),       float("nan"))
                        print(f"    L{l}: attn={a:.3f}  mlp={m:.3f}")

                    unit_to_blk = _build_unit_to_block_index(block_results_dict, ul_list)
                    n_overridden = 0
                    for uid, sub in weighted_subs.items():
                        if uid not in unit_to_blk:
                            continue
                        bid, local_idx = unit_to_blk[uid]
                        U_B   = block_results_dict[bid]['U']               # (d_block, rho_B)
                        d_u   = len(local_idx)
                        unit_share = float((U_B[local_idx, :] ** 2).sum()) / d_u
                        layer  = sub['meta']['layer']
                        utype  = sub['meta'].get('unit_type', '')
                        ctype  = "attn_head" if utype == "attn_head" else "mlp"
                        cons   = layer_cons.get((layer, ctype), 0.0)
                        new_imp = cons * unit_share
                        sub['importance']         = new_imp
                        sub['meta']['importance'] = new_imp
                        n_overridden += 1
                    print(f"  overrode importance for {n_overridden} units")

            # Check that unit importances sum to the block total.
            print("\n[sanity] importance conservation per block:")
            conservation = []
            all_ok = True
            for r in svd_results:
                bid    = r.block_id
                uids_b = block_result.blocks[bid]
                unit_total = float(sum(
                    weighted_subs[uid]['importance']
                    for uid in uids_b
                    if uid in weighted_subs
                ))

                if imp_mode == "residual_topk":
                    # Σ_u importance = (1/d_block) · Σ_{j>k} σ_j² · ||U_B[:,j]||²
                    #                ≈ (1/d_block) · Σ_{j>k} σ_j²    (column orthonormality)
                    sigma_b = block_results_dict[bid]['sigma']
                    rho_B_b = len(sigma_b)
                    d_u_b   = len(unit_to_blk[uids_b[0]][1])
                    d_block = block_results_dict[bid]['U'].shape[0]
                    if rho_B_b <= top_k_skip:
                        expected = 0.0
                    else:
                        expected = float((sigma_b[top_k_skip:] ** 2).sum()) / d_block
                    ratio = (unit_total / expected) if expected > 0 else (
                        1.0 if unit_total == 0.0 else float("inf"))
                    ok = (expected == 0.0 and unit_total == 0.0) or abs(ratio - 1.0) < 0.01
                    all_ok = all_ok and ok
                    conservation.append({
                        "block_id":     bid,
                        "sigma2_residual_total": round(expected, 6),
                        "unit_sum":     round(unit_total, 6),
                        "ratio":        round(ratio, 6),
                        "ok":           ok,
                    })
                    mark = "OK  " if ok else "FAIL"
                    print(f"  {mark} block {bid}: "
                          f"Σσ²[k={top_k_skip}:]/d_block={expected:.4f}  "
                          f"unit_sum={unit_total:.4f}  ratio={ratio:.4f}")
                elif imp_mode == "consistency_share":
                    # importance(u) = cons(layer,type) × ||U_B[idx_u,:]||² / d_u
                    # Σ_u importance = cons × Σ_u ||U_B[idx_u,:]||²/d_u
                    #                = cons × ||U_B||²_F / d_u   (uniform d_u in block)
                    #                ≈ cons × ρ_B / d_u          (column orthonormality)
                    first_unit = unit_layout[uids_b[0]]
                    layer_b    = first_unit.layer
                    ctype_b    = "attn_head" if first_unit.unit_type == "attn_head" else "mlp"
                    cons_b     = layer_cons.get((layer_b, ctype_b), 0.0)
                    U_B_b      = block_results_dict[bid]['U']
                    rho_B_b    = U_B_b.shape[1]
                    d_u_b      = len(unit_to_blk[uids_b[0]][1])
                    expected   = cons_b * rho_B_b / d_u_b
                    ratio      = unit_total / max(expected, 1e-9)
                    ok         = abs(ratio - 1.0) < 0.02
                    all_ok     = all_ok and ok
                    conservation.append({
                        "block_id": bid, "expected": round(expected, 8),
                        "unit_sum": round(unit_total, 8), "ratio": round(ratio, 6), "ok": ok,
                    })
                    mark = "OK  " if ok else "FAIL"
                    print(f"  {mark} block {bid} (L{layer_b} {ctype_b}): "
                          f"expected(cons×ρ/d_u)={expected:.6e}  "
                          f"unit_sum={unit_total:.6e}  ratio={ratio:.4f}")
                else:
                    block_total = float((r.singular_values[:r.effective_rank] ** 2).sum()) / r.d_k
                    ratio = unit_total / max(block_total, 1e-9)
                    ok    = abs(ratio - 1.0) < 0.01
                    all_ok = all_ok and ok
                    conservation.append({
                        "block_id":     bid,
                        "sigma2_total": round(block_total, 6),
                        "unit_sum":     round(unit_total, 6),
                        "ratio":        round(ratio, 6),
                        "ok":           ok,
                    })
                    mark = "OK  " if ok else "FAIL"
                    print(f"  {mark} block {bid}: σ²={block_total:.2f}  "
                          f"unit_sum={unit_total:.2f}  ratio={ratio:.4f}")
            if not all_ok:
                print("  WARNING: importance conservation failed for some blocks!")

            # Importance distribution and ranking
            sorted_units = sorted(
                weighted_subs.items(),
                key=lambda kv: kv[1]['importance'],
                reverse=True,
            )
            importance_vals = [v['importance'] for _, v in sorted_units]
            ranked_units = [
                {
                    "rank":       rank,
                    "unit_id":    int(uid),
                    "label":      unit_layout[uid].label,
                    "unit_type":  unit_layout[uid].unit_type,
                    "layer":      unit_layout[uid].layer,
                    "importance": round(float(v['importance']), 6),
                    "rho":        int(v['rho']),
                }
                for rank, (uid, v) in enumerate(sorted_units)
            ]

            print(f"\n[importance distribution] n_units={len(importance_vals)}")
            print(f"  min={min(importance_vals):.3e}  "
                  f"median={float(np.median(importance_vals)):.3e}  "
                  f"max={max(importance_vals):.3e}")

            # weighted_unit_subspaces.npz (configured importance mode)
            save_weighted_subspaces(weighted_subs, str(weighted_path))
            print(f"\n[saved] {weighted_path}")

            # weighted_unit_subspaces_sigma.npz (sigma_weighted)
            save_weighted_subspaces(weighted_subs_sigma, str(sigma_path))
            print(f"[saved] {sigma_path}")

            # weighted_unit_svd_summary.json
            cache.save_json({
                "conservation": conservation,
                "all_ok":       all_ok,
                "importance_stats": {
                    "n_units":       len(importance_vals),
                    "min":           round(float(np.min(importance_vals)), 6),
                    "median":        round(float(np.median(importance_vals)), 6),
                    "max":           round(float(np.max(importance_vals)), 6),
                    "ratio_max_min": round(
                        float(np.max(importance_vals)) /
                        max(float(np.min(importance_vals)), 1e-12), 2
                    ),
                },
                "ranked_units": ranked_units,
            }, "weighted_unit_svd_summary")
            print(f"  [saved] weighted_unit_svd_summary.json  "
                  f"(top unit: {ranked_units[0]['label']}  "
                  f"importance={ranked_units[0]['importance']:.4f})")

            # ranked_units.csv
            import csv
            csv_path = outdir / "ranked_units.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "rank", "unit_id", "label", "unit_type", "layer",
                    "importance", "rho",
                ])
                writer.writeheader()
                writer.writerows(ranked_units)
            print(f"  [saved] {csv_path}")

            # ranked_units_by_block.csv
            uid_to_block = {
                uid: k
                for k, uids in enumerate(block_result.blocks)
                for uid in uids
            }
            rows_by_block = sorted(
                ranked_units,
                key=lambda r: (uid_to_block.get(r["unit_id"], -1),
                               -r["importance"]),
            )
            block_counters: dict = {}
            for row in rows_by_block:
                bid = uid_to_block.get(row["unit_id"], -1)
                row["block_id"]      = bid
                row["rank_in_block"] = block_counters.get(bid, 0)
                block_counters[bid]  = row["rank_in_block"] + 1
            by_block_path = outdir / "ranked_units_by_block.csv"
            with open(by_block_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "rank", "unit_id", "label", "unit_type", "layer",
                    "importance", "rho", "block_id", "rank_in_block",
                ])
                writer.writeheader()
                writer.writerows(rows_by_block)
            print(f"  [saved] {by_block_path}")

    # Block importance distribution plot
    if do_plot:
        plot_block_importance_dist(outdir)

    # Summary
    print(f"\n{'='*60}")
    print("  Experiment Complete")
    print(f"  Output: {outdir}/")
    print(f"{'='*60}")
    print(f"  NSR      = {diag_result.nsr:.4f}")
    print(f"  SC_mean  = {diag_result.sc_mean:.4f}")
    print(f"  Blocks   = {block_result.K}")
    for r in svd_results:
        layers = sorted({unit_layout[uid].layer
                         for uid in block_result.blocks[r.block_id]})
        print(f"  Block {r.block_id}: ρ={r.effective_rank}  "
              f"redundancy={r.redundancy_ratio:.1%}  layers={layers}")


if __name__ == "__main__":
    main()