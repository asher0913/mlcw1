"""Helper functions for computing and formatting evaluation metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """Returns accuracy, macro/weighted F1, and per-class metrics."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )
    flat_report = {
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
            if label in class_names
        },
    }
    return flat_report
