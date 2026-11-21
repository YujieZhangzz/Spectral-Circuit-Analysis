"""Model loading, parameter scope selection, and flat gradient extraction.

param_filter(name) -> bool selects which parameters' gradients are collected
(the slice theta_C of the full parameter vector theta).
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import re

# Parameter scopes: attn_VO, attn_VO_mlp, all_linear.

def get_param_filter(scope: str, n_layers: int):
    def filter_fn(name: str) -> bool:
        if scope == "attn_VO":
            return (
                re.search(r"\.attn\.c_attn\.", name) or   # GPT-2 V (combined QKV)
                re.search(r"\.attn\.c_proj\.", name) or   # GPT-2 O
                re.search(r"\.self_attn\.v_proj\.", name) or  # LLaMA
                re.search(r"\.self_attn\.o_proj\.", name) or  # LLaMA
                re.search(r"\.attention\.query_key_value\.", name) or  # Pythia/NeoX V (combined QKV)
                re.search(r"\.attention\.dense\.", name)               # Pythia/NeoX O
            ) and "bias" not in name

        elif scope == "attn_VO_mlp":
            attn = (
                re.search(r"\.attn\.c_attn\.", name) or
                re.search(r"\.attn\.c_proj\.", name) or
                re.search(r"\.self_attn\.v_proj\.", name) or
                re.search(r"\.self_attn\.o_proj\.", name) or
                re.search(r"\.attention\.query_key_value\.", name) or  # Pythia/NeoX
                re.search(r"\.attention\.dense\.", name)               # Pythia/NeoX
            )
            mlp = (
                re.search(r"\.mlp\.c_fc\.", name) or      # GPT-2
                re.search(r"\.mlp\.c_proj\.", name) or    # GPT-2
                re.search(r"\.mlp\.gate_proj\.", name) or # LLaMA
                re.search(r"\.mlp\.up_proj\.", name) or   # LLaMA
                re.search(r"\.mlp\.down_proj\.", name) or # LLaMA
                re.search(r"\.mlp\.dense_h_to_4h\.", name) or  # Pythia/NeoX
                re.search(r"\.mlp\.dense_4h_to_h\.", name)     # Pythia/NeoX
            )
            return bool(attn or mlp) and "bias" not in name

        elif scope == "all_linear":           # all linear weights
            return "weight" in name and not any(
                x in name for x in ["embed", "ln", "norm", "lm_head"]
            )
        return False
    return filter_fn



# Model loading

def load_model_and_tokenizer(
    name_or_path: str,
    torch_dtype: str = "float32",
    device: str = "cpu",
) -> Tuple[PreTrainedModel, AutoTokenizer]:

    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[torch_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    # Enable gradients on all parameters (no LoRA).
    for param in model.parameters():
        param.requires_grad_(True)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# Parameter counts

def count_params(
    model: PreTrainedModel,
    param_filter: Callable[[str], bool],
) -> Dict[str, int]:
    """
    Returns:
        {
          "total_D":    total number of selected parameters
          "n_tensors":  number of selected parameter tensors
          "by_layer":   {layer_name: param_count}  (first 10 entries)
        }
    """
    total = 0
    n_tensors = 0
    by_layer: Dict[str, int] = {}

    for name, param in model.named_parameters():
        if not param_filter(name):
            continue
        n = param.numel()
        total += n
        n_tensors += 1
        # Use the layer prefix as key.
        layer_key = ".".join(name.split(".")[:3])
        by_layer[layer_key] = by_layer.get(layer_key, 0) + n

    return {
        "total_D":  total,
        "n_tensors": n_tensors,
        "by_layer":  dict(list(by_layer.items())[:10]),
    }


# Flat gradient extraction

def obtain_gradients(
    model: PreTrainedModel,
    batch: dict,
    param_filter: Callable[[str], bool],
    loss_fn: Optional[Callable[[Any, dict, Optional[dict]], torch.Tensor]] = None,
    loss_extras: Optional[dict] = None,
) -> torch.Tensor:
    """
    Forward + backward; returns the selected parameter gradients concatenated
    into a (D,) float32 vector.

    With loss_fn=None, uses outputs.loss (HF cross-entropy on completion tokens;
    query positions have labels=-100). Otherwise backpropagates
    loss_fn(outputs, batch, loss_extras). loss_extras carries task-specific
    data (e.g. the GT correct_idx threshold) kept out of model(**batch).
    """
    model.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss if loss_fn is None else loss_fn(outputs, batch, loss_extras)
    loss.backward()

    grads: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param_filter(name):
            continue
        if param.grad is not None:
            grads.append(param.grad.detach().float().flatten())
        else:
            grads.append(torch.zeros(param.numel(), dtype=torch.float32))

    model.zero_grad()
    return torch.cat(grads).cpu()   # (D,)