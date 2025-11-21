"""Shared utilities used throughout the coursework code base."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Tuple

import numpy as np


def get_torch_device(require_gpu: bool = True) -> Tuple["torch.device", str]:
    """Choose a torch device, preferring CUDA, then MPS (Apple), else raise if GPU is required."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if require_gpu:
        raise RuntimeError("未检测到可用的 CUDA 或 MPS 设备（需要 GPU 才能运行实验）")
    return torch.device("cpu"), "cpu"


def ensure_dir(path: Path) -> Path:
    """Creates the directory if necessary and returns it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Persists the provided object as pretty-printed JSON."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))


def set_global_seed(seed: int) -> None:
    """Initialises both Python's and NumPy's RNGs."""
    random.seed(seed)
    np.random.seed(seed)


def seed_torch(seed: int) -> None:
    """同步 Torch 的随机种子，并在可用时覆盖 CUDA/MPS。"""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def empty_device_cache() -> None:
    """在支持 CUDA 时清空显存缓存，其他设备上为空操作。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
