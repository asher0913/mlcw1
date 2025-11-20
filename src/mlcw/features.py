"""Feature engineering utilities such as PCA compression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSet:
    """Represents a transformed view of the dataset."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    n_features: int
    explained_variance: float
    description: str


def create_feature_sets(
    X_train: np.ndarray,
    X_test: np.ndarray,
    pca_targets: Sequence[int],
    random_state: int = 42,
) -> List[FeatureSet]:
    """Returns standardized features and PCA-compressed variants."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_sets: List[FeatureSet] = [
        FeatureSet(
            name="original",
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            n_features=X_train_scaled.shape[1],
            explained_variance=1.0,
            description="Standardized flattened RGB pixels.",
        )
    ]

    for pct in sorted(set(pca_targets)):
        label = f"pca_{pct}"
        if pct >= 100:
            feature_sets.append(
                FeatureSet(
                    name=label,
                    X_train=X_train_scaled,
                    X_test=X_test_scaled,
                    n_features=X_train_scaled.shape[1],
                    explained_variance=1.0,
                    description="Reference copy of the full feature space.",
                )
            )
            continue

        keep_ratio = max(min(pct / 100.0, 0.999), 0.01)
        pca = PCA(
            n_components=keep_ratio,
            svd_solver="full",
            random_state=random_state,
            whiten=False,
        )
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        feature_sets.append(
            FeatureSet(
                name=label,
                X_train=X_train_pca,
                X_test=X_test_pca,
                n_features=X_train_pca.shape[1],
                explained_variance=float(np.sum(pca.explained_variance_ratio_)),
                description=f"PCA retaining {pct}% of the training variance.",
            )
        )

    return feature_sets


def dump_feature_metadata(
    feature_sets: Iterable[FeatureSet], output_path: Path
) -> None:
    """Writes the dimensions and explained variance figures to JSON."""
    serializable = []
    for fs in feature_sets:
        entry = {
            "name": fs.name,
            "n_features": fs.n_features,
            "explained_variance": fs.explained_variance,
            "description": fs.description,
        }
        serializable.append(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2))
