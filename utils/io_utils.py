"""File I/O and resume helpers.

Each phase writes its outputs to results/{experiment_name}/. Before running a
phase, run.py checks whether its outputs already exist and loads them if so.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional
import torch
import yaml


# Config

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def get_output_dir(cfg: dict) -> Path:
    d = Path(cfg["output"]["base_dir"]) / cfg["experiment"]["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# .pt files

def save_tensor(tensor: torch.Tensor, path: Path, name: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    mb = tensor.numel() * tensor.element_size() / 1e6
    print(f"  [saved] {name or path.name}  shape={tuple(tensor.shape)}  {mb:.1f} MB")

def load_tensor(path: Path) -> Optional[torch.Tensor]:
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


# .json files

def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [saved] {path.name}")

def load_json(path: Path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# Per-phase cache for resuming runs

class PhaseCache:
    """
    Usage:
        cache = PhaseCache(output_dir)
        if cache.exists("gram_matrix"):
            K = cache.load_tensor("gram_matrix")
        else:
            K = ...compute...
            cache.save_tensor(K, "gram_matrix")
    """
    def __init__(self, output_dir: Path):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _pt(self, key: str) -> Path:
        return self.dir / f"{key}.pt"

    def _json(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def exists(self, key: str) -> bool:
        return self._pt(key).exists() or self._json(key).exists()

    def save_tensor(self, t: torch.Tensor, key: str) -> None:
        save_tensor(t, self._pt(key), name=key)

    def load_tensor(self, key: str) -> Optional[torch.Tensor]:
        return load_tensor(self._pt(key))

    def save_json(self, obj: Any, key: str) -> None:
        save_json(obj, self._json(key))

    def load_json(self, key: str) -> Optional[Any]:
        return load_json(self._json(key))