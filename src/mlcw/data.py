"""Data loading helpers for the coursework experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from torchvision import datasets, transforms


@dataclass
class DatasetBundle:
    """Container with both flattened and raw CIFAR-10 data."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: Sequence[str]
    train_images: np.ndarray
    test_images: np.ndarray


def _flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    """Converts raw image arrays to float32 vectors in [0, 1]."""
    flat = images.reshape(images.shape[0], -1).astype(np.float32)
    return flat / 255.0


def _take_subset(
    X: np.ndarray, y: np.ndarray, subset: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a (possibly) down-sampled copy of the provided dataset."""
    if subset is None or subset >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=subset, replace=False)
    return X[indices], y[indices]


def load_cifar10(
    data_root: str,
    train_subset: int | None = None,
    test_subset: int | None = None,
    random_state: int = 42,
) -> DatasetBundle:
    """Downloads (if needed) and loads the CIFAR-10 dataset."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train_dataset = datasets.CIFAR10(
        root=str(root), train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        root=str(root), train=False, download=True, transform=transforms.ToTensor()
    )

    train_images = train_dataset.data
    y_train = np.array(train_dataset.targets)
    test_images = test_dataset.data
    y_test = np.array(test_dataset.targets)

    train_images, y_train = _take_subset(train_images, y_train, train_subset, random_state)
    test_images, y_test = _take_subset(test_images, y_test, test_subset, random_state + 1)

    X_train = _flatten_and_normalize(train_images)
    X_test = _flatten_and_normalize(test_images)

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=train_dataset.classes,
        train_images=train_images,
        test_images=test_images,
    )
