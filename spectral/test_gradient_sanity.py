"""
End-to-end sanity checks for make_batch and gradient_collector.

  [T1]  make_batch: comp_start matches the offset_mapping boundary
  [T2]  make_batch: completion tokens are not masked (n_valid > 0)
  [T2b] make_batch: n_valid ≈ completion token count (±1 for BPE merges)
  [T3]  make_batch: single-sample loss is finite
  [T4]  make_batch: no NaN loss over all N×M samples
  [T5]  gradient norms are positive and finite
  [T5b] every filtered parameter has a non-zero gradient
  [T6]  sign_accum lies in [-M, M] (no int8 overflow)
  [T7]  grad_avg has shape (N, D), no NaN/inf
  [T8]  Gram diagonal ≈ 1 (atol=0.05 for fp16)
  [T9]  NSR: d_intra < d_inter
  [T10] gradients are deterministic for the same input
  [T11] param_filter dimension == collector.D
  [T12] label shift: manual logprob matches the HF loss
  [D1]  diagnostic only: per-instance CV of gradient norms over paraphrases

Called from run.py (run_sanity_checks) or standalone from the repo root:
    python spectral/test_gradient_sanity.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import numpy as np

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer)

from data.loader       import make_batch, load_records_ioi_v3
from utils.model_utils import get_param_filter, obtain_gradients
from spectral.gradient_collector import collect


PASS = "\033[92mOK\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94m-\033[0m"


# Sanity checks

def run_sanity_checks(
    model:        PreTrainedModel,
    tokenizer:    PreTrainedTokenizer,
    records:      list,
    param_filter,
    cfg:          dict,
    max_n:        int = 10,
) -> dict:
    """
    Run all sanity checks on an already loaded model/tokenizer/records.

    Only the first max_n records are used to keep collect() cheap.
    Returns {passed, total, failed: List[str]}.
    """
    device  = cfg["model"]["device"]
    max_seq = cfg["data"]["max_seq_len"]

    use_records = records[:max_n]
    N = len(use_records)
    M = len(use_records[0]["paraphrase_queries"])

    print(f"\n{'='*60}")
    print(f"  Gradient Sanity Check  N={N} (subset of {len(records)}), M={M}")
    print(f"{'='*60}\n")

    # T7 needs store_grad_avg=True
    san_cfg = dict(cfg)
    san_cfg["collector"] = dict(cfg.get("collector", {}))
    san_cfg["collector"]["store_grad_avg"] = True
    # smaller chunk_size for the Gram computation
    san_cfg["heatmap"] = dict(cfg.get("heatmap", {}))
    san_cfg["heatmap"].setdefault("chunk_size", 500_000)

    results: list = []

    def check(name, ok, detail=""):
        results.append((name, ok))
        mark = PASS if ok else FAIL
        line = f"  {mark} {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        return ok

    def info(name, detail=""):
        line = f"  {INFO} {name}"
        if detail:
            line += f"  ({detail})"
        print(line)

    # T1-T4: make_batch
    print("[make_batch tests]")

    q0 = use_records[0]["paraphrase_queries"][0]
    c0 = use_records[0]["paraphrase_completions"][0]
    batch0 = make_batch(q0, c0, tokenizer, max_seq, device)
    ids0    = batch0["input_ids"][0]
    labels0 = batch0["labels"][0]

    # T1
    char_boundary = len(q0)
    enc0 = tokenizer(q0 + c0, add_special_tokens=True, return_offsets_mapping=True)
    comp_start_ref = next(
        (i for i, (_, e) in enumerate(enc0["offset_mapping"]) if e > char_boundary),
        len(ids0)
    )
    comp_start_actual = int((labels0 != -100).nonzero(as_tuple=True)[0][0])
    check("T1 comp_start via offset_mapping",
          comp_start_actual == comp_start_ref,
          f"expected={comp_start_ref}, got={comp_start_actual}")

    # T2
    n_valid = int((labels0 != -100).sum())
    check("T2 completion tokens not masked",
          n_valid > 0,
          f"valid_labels={n_valid}, seq_len={len(ids0)}")

    # T2b
    full_len   = len(tokenizer(q0 + c0, add_special_tokens=True).input_ids)
    query_len  = len(tokenizer(q0,      add_special_tokens=True).input_ids)
    n_comp_est = full_len - query_len
    check("T2b n_valid ≈ completion token count (±1 BPE)",
          abs(n_valid - n_comp_est) <= 1,
          f"n_valid={n_valid}, completion_est={n_comp_est}")

    # T3
    with torch.no_grad():
        out0 = model(**batch0)
    loss_val = out0.loss.item()
    check("T3 single-sample loss finite",
          np.isfinite(loss_val),
          f"loss={loss_val:.4f}")

    # T4
    print(f"\n  scanning all {N}×{M} samples for NaN loss...")
    nan_count = 0
    for rec in use_records:
        for q, c in zip(rec["paraphrase_queries"], rec["paraphrase_completions"]):
            b = make_batch(q, c, tokenizer, max_seq, device)
            with torch.no_grad():
                out = model(**b)
            if not np.isfinite(out.loss.item()):
                nan_count += 1
    check("T4 no NaN/inf loss across all N×M",
          nan_count == 0,
          f"nan_count={nan_count}/{N*M}")

    # T5b: per-parameter non-zero gradients
    print("\n[gradient_collector tests]")
    print("  [T5b] checking per-param gradients on first sample...")
    batch_t5b = make_batch(q0, c0, tokenizer, max_seq, device)
    model.zero_grad()
    out_t5b = model(**batch_t5b)
    out_t5b.loss.backward()

    zero_params = []
    nonzero_count = 0
    for name, p in model.named_parameters():
        if not param_filter(name):
            continue
        if p.grad is None:
            zero_params.append(f"{name} [no grad]")
        elif p.grad.norm().item() == 0.0:
            zero_params.append(f"{name} [norm=0]")
        else:
            nonzero_count += 1
    model.zero_grad()
    check("T5b per-param non-zero gradients",
          len(zero_params) == 0,
          f"nonzero={nonzero_count}, zero={len(zero_params)}")

    # Run collect()
    print(f"\n  Running collect() on N={N}, M={M}...")
    col = collect(model, tokenizer, use_records, param_filter, san_cfg)

    # T5
    norms = col.grad_norms.numpy()
    check("T5 grad_norms all positive & finite",
          bool(np.all(np.isfinite(norms)) and np.all(norms > 0)),
          f"min={norms.min():.3e}  max={norms.max():.3e}  mean={norms.mean():.3e}")

    # T6
    sa = col.sign_accum.numpy()
    check("T6 sign_accum in [-M, M]",
          bool(sa.min() >= -M and sa.max() <= M),
          f"min={int(sa.min())}  max={int(sa.max())}  expected=[{-M},{M}]")

    # T7
    if col.grad_avg is not None:
        ga = col.grad_avg.float().numpy()
        check("T7 grad_avg shape & finite",
              col.grad_avg.shape == (N, col.D) and bool(np.all(np.isfinite(ga))),
              f"shape={tuple(col.grad_avg.shape)}  nan={np.isnan(ga).sum()}  inf={np.isinf(ga).sum()}")
    else:
        check("T7 grad_avg", False, "grad_avg is None (store_grad_avg=False?)")

    # T8
    gram = col.gram_matrix.numpy()
    diag = np.diag(gram)
    check("T8 Gram diagonal ≈ 1  (atol=0.05 for fp16 memmap)",
          bool(np.allclose(diag, 1.0, atol=0.05)),
          f"diag mean={diag.mean():.6f}  std={diag.std():.2e}")

    # T9
    intra, inter = [], []
    for i in range(N):
        s, e = i * M, i * M + M
        block = gram[s:e, s:e]
        mask  = np.triu(np.ones((M, M), dtype=bool), k=1)
        intra.extend(block[mask].tolist())
        for j in range(N):
            if j == i:
                continue
            sj, ej = j * M, j * M + M
            inter.extend(gram[s:e, sj:ej].flatten().tolist())
    d_intra = 1 - np.mean(intra)
    d_inter = 1 - np.mean(inter)
    nsr     = d_intra / (d_inter + 1e-8)
    check("T9 NSR < 1  (d_intra < d_inter)",
          nsr < 1.0,
          f"NSR={nsr:.4f}  d_intra={d_intra:.4f}  d_inter={d_inter:.4f}")

    # T10: determinism
    print("\n[determinism test]")
    g1 = obtain_gradients(model, batch0, param_filter)
    g2 = obtain_gradients(model, batch0, param_filter)
    max_diff = (g1 - g2).abs().max().item()
    check("T10 gradient deterministic (same input × 2)",
          max_diff == 0.0,
          f"max_diff={max_diff:.2e}")

    # T11: D consistency
    print("\n[param_filter D consistency]")
    n_tensors_kept = sum(1 for name, _ in model.named_parameters() if param_filter(name))
    D_kept = sum(p.numel() for name, p in model.named_parameters() if param_filter(name))
    info(f"param_filter keeps {n_tensors_kept} tensors, D_filter={D_kept:,}")
    check("T11 D_filter == collector.D",
          D_kept == col.D,
          f"filter={D_kept:,}, collector={col.D:,}")

    # T12: label shift
    print("\n[label shift correctness]")
    q_t12 = "The capital of France is "
    c_t12 = "Paris"
    b_t12  = make_batch(q_t12, c_t12, tokenizer, max_seq, device)
    labels_t12 = b_t12["labels"][0]
    n_valid_t12 = int((labels_t12 != -100).sum())

    with torch.no_grad():
        out_t12 = model(**b_t12)
    hf_loss   = out_t12.loss.item()
    comp_tok  = int((labels_t12 != -100).nonzero(as_tuple=True)[0][0])
    paris_id  = b_t12["input_ids"][0, comp_tok].item()
    logprob_manual = F.log_softmax(
        out_t12.logits[0, comp_tok - 1, :], dim=-1
    )[paris_id].item()

    info(f"HF loss={hf_loss:.6f}  -manual={-logprob_manual:.6f}  "
         f"n_valid={n_valid_t12}")

    if n_valid_t12 == 1:
        diff_t12 = abs(hf_loss - (-logprob_manual))
        check("T12 HF loss == -manual_logprob (single completion token)",
              diff_t12 < 1e-4,
              f"|diff|={diff_t12:.2e}")
    else:
        check("T12 manual logprob finite (multi-token fallback)",
              np.isfinite(logprob_manual), f"logprob={logprob_manual:.4f}")

    # D1: per-instance grad_norm CV
    print("\n[diagnostic: per-instance grad_norm CV]")
    high_cv = []
    for i in range(N):
        ns = norms[i, :]
        cv = ns.std() / (ns.mean() + 1e-12)
        if cv > 0.5:
            high_cv.append((i, cv))
    if high_cv:
        info(f"instances with CV > 0.5: "
             + ", ".join(f"i{i}({cv:.2f})" for i, cv in high_cv))
    else:
        info("all instances CV ≤ 0.5")

    # Summary
    passed = sum(ok for _, ok in results)
    total  = len(results)
    failed = [n for n, ok in results if not ok]
    print(f"\n{'='*60}")
    print(f"  Sanity result: {passed}/{total} passed")
    if failed:
        print("  FAILED tests:")
        for n in failed:
            print(f"    - {n}")
    else:
        print("  All sanity checks passed")
    print(f"{'='*60}\n")

    return {"passed": passed, "total": total, "failed": failed}


# Standalone entry point

DATA_PATH  = "data/ioi_v3_strong_semantic_structural_strict_prefix.json"
MODEL_NAME = "gpt2"
DEVICE     = "cpu"
N          = 10
MAX_SEQ    = 256
SCOPE      = "attn_VO_mlp"


def _standalone():
    print("[Setup] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers     = model.config.n_layer
    param_filter = get_param_filter(SCOPE, n_layers)

    print(f"[Setup] Loading data (N={N})...")
    records = load_records_ioi_v3(DATA_PATH, N=N)

    cfg = {
        "model":   {"name_or_path": MODEL_NAME, "device": DEVICE,
                    "torch_dtype": "float32"},
        "data":    {"max_seq_len": MAX_SEQ},
        "params":  {"scope": SCOPE},
        "heatmap": {"chunk_size": 500_000},
        "sc":      {"threshold": 0.7},
        "collector": {"store_grad_avg": True},
        "output":  {"save_intermediate": False},
    }

    res = run_sanity_checks(model, tokenizer, records, param_filter, cfg, max_n=N)
    sys.exit(0 if res["passed"] == res["total"] else 1)


if __name__ == "__main__":
    _standalone()
