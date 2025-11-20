"""Small plotting helpers that store figures under the outputs directory."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from .utils import ensure_dir


def plot_metric_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Plots a simple 2D curve with optional annotations."""
    if not xs:
        return
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_param_bar(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plots a bar chart comparing discrete hyper-parameter settings."""
    if not labels:
        return
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    positions = range(len(labels))
    plt.bar(positions, values)
    plt.xticks(positions, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
