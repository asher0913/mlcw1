"""Shared utilities for the RF coursework package."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np


def get_torch_device(require_gpu: bool = True) -> Tuple["torch.device", str]:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if require_gpu:
        raise RuntimeError("未检测到可用的 CUDA 或 MPS 设备（需要 GPU 才能运行实验）")
    return torch.device("cpu"), "cpu"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class DeviceContext:
    device: "torch.device"
    backend: str
    pin_memory: bool
    non_blocking: bool


def get_device_context(require_gpu: bool = True) -> DeviceContext:
    import torch

    device, backend = get_torch_device(require_gpu)
    pin_memory = backend == "cuda"
    non_blocking = backend in {"cuda", "mps"}
    return DeviceContext(device=device, backend=backend, pin_memory=pin_memory, non_blocking=non_blocking)


def seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def empty_device_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
