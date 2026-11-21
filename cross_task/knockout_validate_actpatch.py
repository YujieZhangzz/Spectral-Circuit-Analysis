"""Cross-task knockout validation by activation patching.

For each (unit set, task) pair, corrupted-prompt activations are patched into
the set's units on clean prompts (attn heads at hook_z, MLPs at hook_mlp_out)
and the task metric is recorded. Each set is compared against random sets with
the same attn:MLP composition. Reads the unit classification CSV from
cross_task/<run>/<variant>/classify/ and writes lift_matrix.csv,
raw_metrics.csv, concentration.csv and summary.txt to <variant>/<out_subdir>/.

Usage: CT_RUN=run_gpt_12tasks python cross_task/knockout_validate_actpatch.py --model gpt2
"""
from __future__ import annotations
import argparse, csv, os, pickle, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformer_lens import HookedTransformer
from circuits_discovery.faithfulness.data import load_ioi
from circuits_discovery.faithfulness.patching import (
    tokenize_pair, make_metric_fn,
)


def cache_corrupted_full(model, corrupted_tokens):
    """Cache both hook_z (attn) and hook_mlp_out (MLP) for each layer."""
    def f(name):
        return name.endswith(".attn.hook_z") or name.endswith(".hook_mlp_out")
    with torch.no_grad():
        _, cache = model.run_with_cache(corrupted_tokens, names_filter=f, return_type=None)
    z_cache   = {l: cache[f"blocks.{l}.attn.hook_z"].detach()  for l in range(model.cfg.n_layers)}
    mlp_cache = {l: cache[f"blocks.{l}.hook_mlp_out"].detach() for l in range(model.cfg.n_layers)}
    return z_cache, mlp_cache


def run_with_patches_full(model, clean_tokens, corrupted_z, corrupted_mlp, targets):
    """
    targets: list of ('attn', layer, head) or ('mlp', layer)
    """
    head_by_layer: Dict[int, List[int]] = {}
    mlp_layers: set = set()
    for t in targets:
        if t[0] == "attn":
            _, l, h = t
            head_by_layer.setdefault(l, []).append(h)
        elif t[0] == "mlp":
            mlp_layers.add(t[1])

    fwd_hooks = []
    # Attn z hooks, one per layer
    for l, heads in head_by_layer.items():
        heads_t = torch.tensor(sorted(set(heads)), dtype=torch.long,
                                device=corrupted_z[l].device)
        cz = corrupted_z[l]
        def make_z_hook(cz_tensor, head_idx):
            def hook(z, hook):
                z[:, :, head_idx, :] = cz_tensor[:, :, head_idx, :]
                return z
            return hook
        fwd_hooks.append((f"blocks.{l}.attn.hook_z", make_z_hook(cz, heads_t)))

    # MLP hooks replace the whole MLP output
    for l in mlp_layers:
        cm = corrupted_mlp[l]
        def make_mlp_hook(cm_tensor):
            def hook(mlp_out, hook):
                return cm_tensor
            return hook
        fwd_hooks.append((f"blocks.{l}.hook_mlp_out", make_mlp_hook(cm)))

    with torch.no_grad():
        logits = model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks, return_type="logits")
    return logits


def units_to_targets(unit_ids):
    """Convert unit_id list to patching targets list (deduping mlp layers)."""
    targets = []
    seen_mlp = set()
    for uid in unit_ids:
        layer = uid // 14
        intra = uid % 14
        if intra < 12:
            targets.append(("attn", layer, intra))
        else:
            if layer not in seen_mlp:
                targets.append(("mlp", layer))
                seen_mlp.add(layer)
    return targets

EAP_DATA      = Path("eap_data")
VERB_LIST_CSV = str(EAP_DATA / "sva" / "combined_verb_list.csv")

TASKS_GPT2: Dict[str, dict] = {
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

# Mirrors per-task pythia configs under configs/pythia_*.yaml.
TASKS_PYTHIA: Dict[str, dict] = {
    "col_obj":         dict(csv=EAP_DATA / "colored-objects/sampled_all_pythia.csv",        schema="logit_diff"),
    "country_cap":     dict(csv=EAP_DATA / "fact-retrieval-comma/sampled_all_pythia.csv",   schema="logit_diff"),
    "gender_bias":     dict(csv=EAP_DATA / "gender-bias/sampled_all_pythia_200.csv",        schema="logit_diff"),
    "gp":              dict(csv=EAP_DATA / "gender-pronoun/sampled_all_pythia_200.csv",     schema="logit_diff"),
    "gt":              dict(csv=EAP_DATA / "greater-than/sampled_all_pythia_200.csv",       schema="greater_than"),
    "gt_price":        dict(csv=EAP_DATA / "greater-than-price/sampled_all_pythia_200.csv", schema="greater_than"),
    "ioi_strong":      dict(csv=EAP_DATA / "ioi/sampled_all_pythia_200.csv",                 schema="logit_diff"),
    "person_country":  dict(csv=EAP_DATA / "person-country/sampled_all_pythia.csv",          schema="logit_diff"),
    "person_pro":      dict(csv=EAP_DATA / "person-profession/sampled_all_pythia.csv",       schema="logit_diff"),
    "sva":             dict(csv=EAP_DATA / "sva/sampled_all_pythia_200_extended-2.csv",      schema="sva_verbset"),
    "sva_blimp":       dict(csv=EAP_DATA / "sva-blimp/sampled_all_pythia_200.csv",           schema="sva_verbset"),
    "sva_isare":       dict(csv=EAP_DATA / "sva/sampled_all_pythia_200_extended-2.csv",      schema="sva_isare"),
}

# Picked at main() based on --model (overridable with --task_set).
TASKS: Dict[str, dict] = TASKS_GPT2

TASK_TO_FAMILY = {
    "col_obj":         None,
    "country_cap":     "COUNTRY",
    "gender_bias":     "PRONOUN",
    "gp":              "PRONOUN",
    "gt":              "GT",
    "gt_price":        "GT",
    "ioi_strong":      None,
    "person_country":  "COUNTRY",
    "person_pro":      "PRONOUN",
    "sva":             "SVA",
    "sva_blimp":       "SVA",
    "sva_isare":       "SVA",
}


def _category_key(r):
    """Derive category_key from (category, primary_label, family_tier).
    Uses an explicit category_key column if present.
    """
    if "category_key" in r and r["category_key"]:
        return r["category_key"]
    cat   = r.get("category", "")
    label = r.get("primary_label", "")
    tier  = r.get("family_tier", "-")
    if cat == "universal":
        return "universal"
    if cat == "task":
        return "task_specific"
    if cat == "mixed":
        return "mixed"
    if cat == "family":
        # label is "family_<F>" -> key = "family_<F>_<tier>"
        return f"{label}_{tier}"        # e.g., "family_GT_specialized"
    return "mixed"


def parse_classification(csv_path: Path) -> Dict[str, List[int]]:
    rows = list(csv.DictReader(open(csv_path)))
    # Annotate category_key in place
    for r in rows:
        r["_ck"] = _category_key(r)

    sets: Dict[str, List[int]] = {}
    sets["Universal"] = [int(r["unit_id"]) for r in rows if r["_ck"] == "universal"]

    gt_spec = [r for r in rows if r["_ck"] == "family_GT_specialized"]
    gt_spec.sort(key=lambda r: -float(r["within_GT_min"]))
    sets[f"GT_specialized_{len(gt_spec)}"] = [int(r["unit_id"]) for r in gt_spec]
    sets["GT_specialized_top4"]            = [int(r["unit_id"]) for r in gt_spec[:4]]

    # GT broad sets, ranked by within_GT_min (descending)
    gt_broad = [r for r in rows if r["_ck"] == "family_GT_broad"]
    gt_broad.sort(key=lambda r: -float(r["within_GT_min"]))
    sets["GT_broad_top4"]   = sorted([int(r["unit_id"]) for r in gt_broad[:4]])
    sets["GT_broad_top9"]   = sorted([int(r["unit_id"]) for r in gt_broad[:9]])
    sets["GT_broad_all"]    = sorted([int(r["unit_id"]) for r in gt_broad])
    gt_broad_mlps_sorted = [r for r in gt_broad if r["unit_type"] in ("mlp_gate", "mlp_down")]
    sets["GT_broad_mlps_top4"] = sorted([int(r["unit_id"]) for r in gt_broad_mlps_sorted[:4]])
    sets["GT_broad_mlps_all"]  = sorted([int(r["unit_id"]) for r in gt_broad_mlps_sorted])

    # Family GT all (specialized + broad)
    sets["GT_all"] = sorted(sets[f"GT_specialized_{len(gt_spec)}"] + sets["GT_broad_all"])

    sets["SVA_specialized_4"] = [int(r["unit_id"]) for r in rows
                                  if r["_ck"] == "family_SVA_specialized"]
    sva_broad = [r for r in rows if r["_ck"] == "family_SVA_broad"]
    sets["SVA_broad_all"] = sorted([int(r["unit_id"]) for r in sva_broad])
    sets["SVA_all"] = sorted(sets["SVA_specialized_4"] + sets["SVA_broad_all"])

    sets["COUNTRY_broad_2"]   = [int(r["unit_id"]) for r in rows
                                  if r["_ck"].startswith("family_COUNTRY")]
    sets["PRONOUN_all"]       = [int(r["unit_id"]) for r in rows
                                  if r["_ck"].startswith("family_PRONOUN")]
    sets["IOI_COL_all"]       = [int(r["unit_id"]) for r in rows
                                  if r["_ck"].startswith("family_IOI_COL")]

    # Task-specific units: one set per task with at least one such unit.
    # Ablating these should hurt only the target task.
    task_groups: Dict[str, List[int]] = {}
    for r in rows:
        if r["category"] != "task": continue
        tname = r["primary_label"].split("_", 1)[1]   # e.g. "ioi_strong"
        task_groups.setdefault(tname, []).append(int(r["unit_id"]))
    for tname, uids in task_groups.items():
        sets[f"TaskSpec_{tname}_{len(uids)}"] = sorted(uids)

    mixed = [int(r["unit_id"]) for r in rows if r["_ck"] == "mixed"]
    rng = np.random.default_rng(0)
    n_ctrl = min(22, len(mixed))
    sets[f"Mixed_{n_ctrl}_control"] = sorted(rng.choice(mixed, n_ctrl, replace=False).tolist()) if mixed else []
    # Drop empty sets (so things like SVA_broad_all=[] don't pollute output)
    sets = {k: v for k, v in sets.items() if v}
    return sets


def _is_attn_uid(uid: int) -> bool:
    return (uid % 14) < 12


def make_random_unit_sets(n_random, compositions, all_unit_ids, base_seed=100):
    """For each unique (n_attn, n_mlp) composition, generate n_random random
    subsets that preserve the attn:MLP composition of the target set.

    `compositions` is an iterable of (n_attn, n_mlp) tuples.
    Returns dict keyed by composition -> list of unit-id lists.

    Type-matching matters because attn and MLP knockouts have very different
    average effects; a composition-mismatched random baseline biases lift.
    """
    attn_pool = [u for u in all_unit_ids if _is_attn_uid(u)]
    mlp_pool  = [u for u in all_unit_ids if not _is_attn_uid(u)]
    out = {}
    for (n_attn, n_mlp) in compositions:
        out[(n_attn, n_mlp)] = []
        for s in range(n_random):
            rng = np.random.default_rng(base_seed + (n_attn * 13 + n_mlp) * 1000 + s)
            picks = []
            if n_attn > 0:
                picks += list(rng.choice(attn_pool, n_attn, replace=False))
            if n_mlp > 0:
                picks += list(rng.choice(mlp_pool,  n_mlp,  replace=False))
            out[(n_attn, n_mlp)].append(sorted(int(u) for u in picks))
    return out


def composition_of(unit_ids):
    n_attn = sum(1 for u in unit_ids if _is_attn_uid(u))
    return (n_attn, len(unit_ids) - n_attn)


def build_metric_for_task(model: HookedTransformer, batch, schema: str, input_lengths, labels):
    """Dispatch make_metric_fn to the right task type."""
    if schema == "logit_diff":
        return make_metric_fn("logit_diff", model, input_lengths, labels)
    if schema == "greater_than":
        return make_metric_fn("greater_than", model, input_lengths, labels)
    if schema == "sva_verbset":
        return make_metric_fn("sva", model, input_lengths, labels,
                              verb_list_csv=VERB_LIST_CSV, sva_metric_variant="verb_set")
    if schema == "sva_isare":
        return make_metric_fn("sva", model, input_lengths, labels,
                              sva_metric_variant="is_are")
    raise ValueError(f"unknown schema {schema}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",  default="cuda:0")
    p.add_argument("--n",       type=int, default=100)
    p.add_argument("--n_random", type=int, default=10)
    p.add_argument("--variant", default="consistency_share",
                   choices=["consistency_share", "sigma_weighted"])
    p.add_argument("--model",   default="gpt2")
    p.add_argument("--out_subdir", default="knockout_validation_act_full",
                   help="output subdir name (act_full = attn z + mlp_out patching)")
    p.add_argument("--classify_subdir", default="classify",
                   choices=["classify", "classify_weighted"],
                   help="which classification dir to read sets from")
    p.add_argument("--classify_csv_name", default=None,
                   help="basename of the classification csv to read "
                        "(default: v1_min.csv if classify_weighted, v1_min_annotated.csv otherwise)")
    p.add_argument("--sanity",  action="store_true")
    p.add_argument("--task_set", default="auto", choices=["auto", "gpt2", "pythia"],
                   help="which TASKS dict to use; 'auto' picks pythia if --model "
                        "contains 'pythia', else gpt2")
    p.add_argument("--run_dir", default=None,
                   help="run dir name under cross_task/ (default: from CT_RUN env, "
                        "else 'run_gpt_12tasks')")
    return p.parse_args()


def main():
    args = parse_args()
    global TASKS
    if args.task_set == "auto":
        TASKS = TASKS_PYTHIA if "pythia" in args.model.lower() else TASKS_GPT2
    else:
        TASKS = TASKS_PYTHIA if args.task_set == "pythia" else TASKS_GPT2
    run_dir_name = args.run_dir or os.environ.get("CT_RUN", "run_gpt_12tasks")
    base = REPO_ROOT / "cross_task" / run_dir_name / args.variant
    csv_name = args.classify_csv_name or (
        "v1_min.csv" if args.classify_subdir == "classify_weighted" else "v1_min_annotated.csv"
    )
    classify_csv = base / args.classify_subdir / csv_name
    out_dir = base / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  Cross-task knockout validation - activation patching")
    print(f"{'='*72}")
    print(f"  variant     : {args.variant}")
    print(f"  out         : {out_dir}")
    print(f"  n / n_random: {args.n} / {args.n_random}")

    print(f"\n[1] Loading HookedTransformer({args.model}) on {args.device}")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model = model.to(args.device)   # TL sometimes leaves embeds on cuda:0; force.
    model.eval()
    tokenizer = model.tokenizer

    print(f"\n[2] Parsing unit sets (attn + MLP) from {classify_csv.name}")
    sets_uid = parse_classification(classify_csv)
    if args.sanity:
        keep = ["Universal", "GT_specialized_top4",
                "GT_broad_top4", "GT_broad_top9",
                "GT_broad_mlps_top4", "GT_broad_mlps_all",
                "SVA_specialized_4"]
        sets_uid = {k: v for k, v in sets_uid.items() if k in keep}
        print(f"   --sanity: restricted to {list(sets_uid.keys())}")
    for name, uids in sets_uid.items():
        print(f"   {name:<26s} n={len(uids):>3d}  unit_ids={uids[:6]}{'...' if len(uids)>6 else ''}")

    # All 168 unit_ids; type-matched random baseline preserves (n_attn, n_mlp).
    all_unit_ids = list(range(168))   # 12 layers x 14 (12 attn + 2 mlp) = 168
    compositions = sorted({composition_of(uids) for uids in sets_uid.values()}) or [(1, 0)]
    random_units = make_random_unit_sets(args.n_random, compositions, all_unit_ids, base_seed=100)
    print(f"   random baseline compositions (n_attn, n_mlp), type-matched: {compositions}")

    # Load tasks, cache corrupted z + mlp_out, build metrics
    print(f"\n[3] Loading 12 tasks (n={args.n}) and caching corrupted z + mlp_out")
    task_pack: Dict[str, dict] = {}
    for tname, info in TASKS.items():
        batch = load_ioi(info["csv"], n=args.n, gt_mode="prob_diff", tokenizer=tokenizer)
        clean_t, corrupted_t, input_lengths, _n_pos = tokenize_pair(model, batch.clean, batch.corrupted)
        corrupted_z, corrupted_mlp = cache_corrupted_full(model, corrupted_t)
        metric_fn = build_metric_for_task(model, batch, info["schema"], input_lengths, batch.labels)
        task_pack[tname] = dict(
            clean_t=clean_t, corrupted_z=corrupted_z, corrupted_mlp=corrupted_mlp,
            metric_fn=metric_fn, input_lengths=input_lengths,
        )
        print(f"   {tname:<16s} n={len(batch.clean):3d}  task={batch.task}")

    # Clean baseline
    print(f"\n[4] Clean baselines (no patching)")
    L_clean = {}
    with torch.no_grad():
        for tname, p in task_pack.items():
            logits = model(p["clean_t"])
            L_clean[tname] = float(p["metric_fn"](logits).item())
            print(f"   {tname:<16s}  L_clean = {L_clean[tname]:+.4f}")

    # Eval helper (attn z + mlp_out patches)
    def eval_set_on_task(unit_ids, tname):
        p = task_pack[tname]
        targets = units_to_targets(unit_ids)
        if not targets:
            with torch.no_grad():
                logits = model(p["clean_t"])
        else:
            logits = run_with_patches_full(model, p["clean_t"],
                                            p["corrupted_z"], p["corrupted_mlp"],
                                            targets)
        return float(p["metric_fn"](logits).item())

    # Per-set knockout
    raw_results: Dict[str, Dict[str, float]] = {"_clean": L_clean}
    if args.sanity:
        random_results: Dict[tuple, List[Dict[str, float]]] = {}
    else:
        random_results = {comp: [{} for _ in range(args.n_random)] for comp in compositions}

    print(f"\n[5] Knockout per (set x task): {len(sets_uid)} sets x 12 tasks")
    t0 = time.time()
    for set_name, unit_ids in sets_uid.items():
        L = {tname: eval_set_on_task(unit_ids, tname) for tname in TASKS}
        raw_results[set_name] = L
        elapsed = time.time() - t0
        n_attn = sum(1 for u in unit_ids if u % 14 < 12)
        n_mlp_layers = len({u // 14 for u in unit_ids if u % 14 >= 12})
        print(f"   {set_name:<26s} n_units={len(unit_ids):>3}  "
              f"(attn={n_attn}, mlp_layers={n_mlp_layers})  cumulative={elapsed:.1f}s")

    # Random baselines (type-matched)
    if not args.sanity:
        print(f"\n[6] Random baselines ({len(compositions)} compositions x {args.n_random} draws x 12 tasks)")
        for comp in compositions:
            for s, uids in enumerate(random_units[comp]):
                for tname in TASKS:
                    random_results[comp][s][tname] = eval_set_on_task(uids, tname)
                print(f"   (n_attn={comp[0]:>2}, n_mlp={comp[1]:>2})  draw={s+1}/{args.n_random}  cumulative={time.time()-t0:.1f}s")
    else:
        print(f"\n[6] (sanity) skipping random baselines")

    # Lift + z-score
    def faith(L_clean_t, L_t):
        return (L_clean_t - L_t) / max(abs(L_clean_t), 1e-6)

    print(f"\n[7] Computing Lift + z-score")
    lift_rows = []
    for set_name, unit_ids in sets_uid.items():
        size = len(unit_ids)
        comp = composition_of(unit_ids)
        F_S = {t: faith(L_clean[t], raw_results[set_name][t]) for t in TASKS}
        if random_results.get(comp):
            F_R_mean = {t: float(np.mean([faith(L_clean[t], r[t]) for r in random_results[comp]])) for t in TASKS}
            F_R_std  = {t: float(np.std ([faith(L_clean[t], r[t]) for r in random_results[comp]])) for t in TASKS}
        else:
            F_R_mean = {t: 0.0 for t in TASKS}
            F_R_std  = {t: 0.0 for t in TASKS}
        # z-score: how many random-baseline std the set's faithfulness drop is above the random mean
        z = {t: ((F_S[t] - F_R_mean[t]) / F_R_std[t]) if F_R_std[t] > 1e-9 else float("nan")
             for t in TASKS}
        lift_rows.append({"set": set_name, "size": size,
                          "n_attn": comp[0], "n_mlp": comp[1],
                          **{f"lift_{t}": F_S[t] - F_R_mean[t] for t in TASKS},
                          **{f"z_{t}":   z[t] for t in TASKS},
                          **{f"F_S_{t}":  F_S[t] for t in TASKS},
                          **{f"F_R_{t}":  F_R_mean[t] for t in TASKS},
                          **{f"F_Rstd_{t}": F_R_std[t] for t in TASKS}})

    # Save
    lift_csv = out_dir / "lift_matrix.csv"
    cols = (["set", "size", "n_attn", "n_mlp"]
            + [f"lift_{t}" for t in TASKS]
            + [f"z_{t}"    for t in TASKS])
    with open(lift_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in lift_rows:
            row = [r["set"], r["size"], r["n_attn"], r["n_mlp"]]
            row += [f"{r[f'lift_{t}']:.4f}" for t in TASKS]
            row += [f"{r[f'z_{t}']:.2f}" if not np.isnan(r[f'z_{t}']) else "" for t in TASKS]
            w.writerow(row)
    print(f"   [saved] {lift_csv}")

    raw_csv = out_dir / "raw_metrics.csv"
    with open(raw_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["condition"] + list(TASKS))
        for cond, L in raw_results.items():
            w.writerow([cond] + [f"{L[t]:.4f}" for t in TASKS])
        for comp, draws in random_results.items():
            for s, L in enumerate(draws):
                tag = f"random_attn{comp[0]}_mlp{comp[1]}_draw{s}"
                w.writerow([tag] + [f"{L[t]:.4f}" for t in TASKS])
    print(f"   [saved] {raw_csv}")

    # Concentration
    fam_for_set = {
        "Universal":              None,
        "Mixed_22_control":       None,
    }
    # Auto-assign family by set name prefix
    for set_name in lift_rows[0].keys() if lift_rows else []:
        pass
    for r in lift_rows:
        nm = r["set"]
        if nm in fam_for_set:
            continue
        if nm.startswith("GT"):
            fam_for_set[nm] = "GT"
        elif nm.startswith("SVA"):
            fam_for_set[nm] = "SVA"
        elif nm.startswith("COUNTRY"):
            fam_for_set[nm] = "COUNTRY"
        elif nm.startswith("PRONOUN"):
            fam_for_set[nm] = "PRONOUN"
        elif nm.startswith("Mixed"):
            fam_for_set[nm] = None
        else:
            fam_for_set[nm] = None
    conc_rows = []
    for r in lift_rows:
        nm = r["set"]
        # Task-specific sets: in = single target task only
        if nm.startswith("TaskSpec_"):
            # Extract task name: TaskSpec_<task>_<n>
            tname = nm[len("TaskSpec_"):].rsplit("_", 1)[0]
            in_fam = [r[f"lift_{tname}"]] if tname in TASKS else []
            out_fam = [r[f"lift_{t}"] for t in TASKS if t != tname]
            fam_label = f"task:{tname}"
        else:
            fam = fam_for_set.get(nm)
            in_fam, out_fam = [], []
            for t in TASKS:
                l = r[f"lift_{t}"]
                if fam is not None and TASK_TO_FAMILY.get(t) == fam:
                    in_fam.append(l)
                else:
                    out_fam.append(l)
            fam_label = fam or "—"
        in_mean = float(np.mean(in_fam)) if in_fam else float("nan")
        out_mean = float(np.mean(out_fam)) if out_fam else float("nan")
        ratio = (in_mean / out_mean) if (out_mean and abs(out_mean) > 1e-6) else float("nan")
        # Also compute |ratio| for direction-agnostic reporting
        abs_ratio = (abs(in_mean) / abs(out_mean)) if (out_mean and abs(out_mean) > 1e-6) else float("nan")
        conc_rows.append({
            "set": nm, "size": r["size"], "expected_family": fam_label,
            "lift_in_family":  f"{in_mean:.4f}",
            "lift_out_family": f"{out_mean:.4f}",
            "concentration":   f"{ratio:.3f}" if not np.isnan(ratio) else "—",
            "abs_ratio":       f"{abs_ratio:.3f}" if not np.isnan(abs_ratio) else "—",
        })
    conc_csv = out_dir / "concentration.csv"
    with open(conc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conc_rows[0].keys()))
        w.writeheader(); w.writerows(conc_rows)
    print(f"   [saved] {conc_csv}")

    summary = out_dir / "summary.txt"
    with open(summary, "w") as f:
        f.write(f"Cross-task knockout validation - activation patching\n{'='*60}\n\n")
        f.write(f"variant: {args.variant}\nn_prompts: {args.n}\nn_random: {args.n_random}\n")
        f.write(f"random baseline: type-matched (preserves attn:MLP composition per set)\n\n")
        f.write("Lift matrix (signed; rows=set, cols=task):\n")
        f.write(f"{'set':<28s} | " + " ".join(f"{t[:8]:>8s}" for t in TASKS) + "\n")
        f.write("-" * (30 + 9 * len(TASKS)) + "\n")
        for r in lift_rows:
            f.write(f"{r['set']:<28s} | " + " ".join(f"{r[f'lift_{t}']:>8.3f}" for t in TASKS) + "\n")
        f.write("\nz-score (lift / random-baseline std; |z| > 2 ≈ significant):\n")
        f.write(f"{'set':<28s} | " + " ".join(f"{t[:8]:>8s}" for t in TASKS) + "\n")
        f.write("-" * (30 + 9 * len(TASKS)) + "\n")
        for r in lift_rows:
            cells = []
            for t in TASKS:
                z = r[f"z_{t}"]
                cells.append(f"{z:>8.2f}" if not np.isnan(z) else f"{'—':>8s}")
            f.write(f"{r['set']:<28s} | " + " ".join(cells) + "\n")
        f.write("\nConcentration (in-fam / out-fam mean lift; |ratio| for sign-agnostic):\n")
        for c in conc_rows:
            f.write(f"  {c['set']:<28s}  fam={c['expected_family']:<14s}  "
                    f"in={c['lift_in_family']}  out={c['lift_out_family']}  "
                    f"ratio={c['concentration']}  |ratio|={c['abs_ratio']}\n")
    print(f"   [saved] {summary}")
    print(f"\n[done] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
