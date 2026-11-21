"""
Streaming per-example gradient collection.

Each gradient g_{i,j} (instance i, paraphrase j) is folded into the Gram
matrix, the int8 sign accumulator and grad_sum, then freed, so the full
(N, M, D) tensor is never materialised. grad_sum is converted to fp16 at the
end. For large models grad_sum can be big; set collector.store_grad_avg: false
to disable it.
"""

from __future__ import annotations
import gc
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from data.loader import make_batch
from utils.model_utils import obtain_gradients


# Result container

@dataclass
class CollectionResult:
    # Phase 1
    gram_matrix:  torch.Tensor          # (NM, NM)  float32, cosine similarity

    # Phase 2 (SC diagnostic)
    sign_accum:   torch.Tensor          # (N, D)  int8, sum of signs over the M paraphrases

    # Phase 3 (block finding)
    # grad_avg[i, d] = (1/M) * Σ_j grad_{i,j,d}
    # stored as float16; None if store_grad_avg=False
    grad_avg:     Optional[torch.Tensor]  # (N, D)  float16 | None

    M:            int
    N:            int
    D:            int
    elapsed_sec:  float
    grad_norms:   torch.Tensor          # (N, M)  float32


# Main entry point

def collect(
    model:        PreTrainedModel,
    tokenizer:    PreTrainedTokenizer,
    records:      List[Dict],
    param_filter: Callable[[str], bool],
    cfg:          dict,
    loss_fn:      Optional[Callable] = None,
) -> CollectionResult:
    """
    Single pass over all (i, j) pairs, maintaining the Gram matrix,
    sign_accum and grad_sum.

    cfg["collector"]["store_grad_avg"] (bool, default True): set False or
    reduce N for large models (D > 200M) to save memory.

    loss_fn=None uses the HF cross-entropy loss (IOI/cap/gp/gb). Otherwise
    records[i]["loss_extras_per_paraphrase"][j] is passed through
    obtain_gradients to loss_fn (GT uses this to pass correct_idx).
    """
    N        = len(records)
    M        = len(records[0]["paraphrase_queries"])
    NM       = N * M
    device   = cfg["model"]["device"]
    max_len  = cfg["data"]["max_seq_len"]
    chunk    = cfg["heatmap"]["chunk_size"]
    store_avg = cfg.get("collector", {}).get("store_grad_avg", True)

    metric_name = "ce" if loss_fn is None else "custom"
    print(f"[Collector] N={N}, M={M}, NM={NM}, store_grad_avg={store_avg}, "
          f"metric={metric_name}")

    def _extras(rec, j):
        ex = rec.get("loss_extras_per_paraphrase")
        return ex[j] if ex is not None else None

    # Probe D
    test_batch = make_batch(
        records[0]["paraphrase_queries"][0],
        records[0]["paraphrase_completions"][0],
        tokenizer, max_len, device,
    )
    test_g = obtain_gradients(
        model, test_batch, param_filter,
        loss_fn=loss_fn, loss_extras=_extras(records[0], 0),
    )
    D = test_g.shape[0]
    del test_g

    mem_sign  = N * D * 1 / 1e9
    mem_avg   = N * D * 4 / 1e9   # float32 accumulation
    mem_gram  = NM * NM * 4 / 1e9
    print(f"[Collector] D={D:,}  ({D*4/1e6:.1f} MB per grad vector)")
    print(f"  Memory estimate: sign_accum={mem_sign:.2f}GB  "
          f"grad_sum={mem_avg:.2f}GB  Gram={mem_gram:.2f}GB")
    if mem_avg > 8.0 and store_avg:
        print(f"  [WARN] grad_avg memory ({mem_avg:.1f}GB) is large. "
              f"Set cfg.collector.store_grad_avg=false if OOM.")

    # Accumulators
    sign_accum = torch.zeros(N, D, dtype=torch.int8)
    grad_norms = torch.zeros(N, M, dtype=torch.float32)
    # grad_sum is accumulated in float32 and converted to float16 at the end
    grad_sum   = torch.zeros(N, D, dtype=torch.float32) if store_avg else None

    # Normalised gradients for the Gram matrix: in memory or memmap on disk
    use_streaming = (NM * D * 2) > 4e9
    if use_streaming:
        import numpy as np
        import tempfile, os as _os
        _tmpdir = cfg.get("collector", {}).get("tmpdir", None)
        print(f"[Collector] Using memmap Gram (NM×D×fp16={NM*D*2/1e9:.1f}GB)"
              + (f" -> {_tmpdir}" if _tmpdir else ""))
        _tmp      = tempfile.NamedTemporaryFile(suffix=".bin", delete=False,
                                                dir=_tmpdir)
        _tmp_path = _tmp.name; _tmp.close()
        norm_grads_mm = np.memmap(_tmp_path, dtype="float16",
                                   mode="w+", shape=(NM, D))
        norm_grads = None
    else:
        import numpy as np
        norm_grads_mm = None; _tmp_path = None
        norm_grads = torch.zeros(NM, D, dtype=torch.float16)

    # Main loop
    t0  = time.time()
    idx = 0

    for i, rec in enumerate(tqdm(records, desc="Instances")):
        for j, (q, c) in enumerate(zip(
            rec["paraphrase_queries"],
            rec["paraphrase_completions"],
        )):
            batch = make_batch(q, c, tokenizer, max_len, device)
            g     = obtain_gradients(
                model, batch, param_filter,
                loss_fn=loss_fn, loss_extras=_extras(rec, j),
            )

            # sign_accum
            sign_accum[i] += torch.sign(g).to(torch.int8)

            # grad_sum (for grad_avg)
            if grad_sum is not None:
                grad_sum[i] += g

            # grad norms
            norm = g.norm().item()
            grad_norms[i, j] = norm

            # normalised gradient for the Gram matrix
            g_norm = F.normalize(g.unsqueeze(0), p=2, dim=1).squeeze(0)
            if norm_grads is not None:
                norm_grads[idx] = g_norm.to(torch.float16)
            else:
                norm_grads_mm[idx] = g_norm.to(torch.float16).numpy()

            del g, g_norm
            idx += 1

        if (i + 1) % 50 == 0:
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"[Collector] Gradient loop done in {elapsed:.1f}s")

    # grad_avg: divide by M, cast to float16
    grad_avg = None
    if grad_sum is not None:
        print("[Collector] Computing grad_avg (float16)...")
        grad_avg = (grad_sum / M).to(torch.float16)
        del grad_sum
        gc.collect()
        print(f"  grad_avg shape={tuple(grad_avg.shape)}, "
              f"memory={grad_avg.numel()*2/1e9:.2f}GB")

    # Gram matrix
    gram_K = torch.zeros(NM, NM, dtype=torch.float32)

    if norm_grads is not None:
        print("[Collector] Building Gram matrix (in-memory chunked)...")
        t1 = time.time()
        G  = norm_grads.float()
        for d_s in range(0, D, chunk):
            d_e = min(d_s + chunk, D)
            gram_K += G[:, d_s:d_e] @ G[:, d_s:d_e].T
        del norm_grads, G
        print(f"[Collector] Gram done in {time.time()-t1:.1f}s")

    elif norm_grads_mm is not None:
        print("[Collector] Building Gram matrix (memmap chunked)...")
        t1 = time.time()
        norm_grads_mm.flush()
        for d_s in range(0, D, chunk):
            d_e = min(d_s + chunk, D)
            G_c = torch.from_numpy(
                norm_grads_mm[:, d_s:d_e].astype(np.float32)
            )
            gram_K += G_c @ G_c.T
            del G_c
        del norm_grads_mm
        _os.unlink(_tmp_path)
        print(f"[Collector] Gram done in {time.time()-t1:.1f}s")

    gc.collect()

    return CollectionResult(
        gram_matrix  = gram_K,
        sign_accum   = sign_accum,
        grad_avg     = grad_avg,
        M            = M,
        N            = N,
        D            = D,
        elapsed_sec  = elapsed,
        grad_norms   = grad_norms,
    )