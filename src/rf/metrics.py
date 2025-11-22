"""Evaluation helpers for accuracy/F1 calculations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    label_indices: Sequence[int] | None = None,
) -> dict:
    labels = list(label_indices) if label_indices is not None else list(
        np.unique(np.concatenate([y_true, y_pred]))
    )
    target_names = [str(name) for name in class_names]
    if len(target_names) != len(labels):
        if label_indices is None:
            target_names = [str(label) for label in labels]
        else:
            raise ValueError("class_names 与 label_indices 长度不匹配")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    flat = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {
            label: {
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": int(values["support"]),
            }
            for label, values in report.items()
            if label in target_names
        },
    }
    return flat
