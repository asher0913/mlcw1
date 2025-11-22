"""Configuration dataclasses used across the RF coursework experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DataConfig:
    data_root: str = "data"
    train_subset: int | None = 10000
    test_subset: int | None = 2000
    random_state: int = 42


@dataclass
class FeatureConfig:
    pca_targets: Tuple[int, ...] = (10, 30, 50, 70, 100)


@dataclass
class MLPConfig:
    hidden_layer_sizes: Tuple[int, ...] = (512, 256)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4
    batch_size: int = 256
    learning_rate_init: float = 1e-3
    max_iter: int = 80
    early_stopping: bool = True
    n_iter_no_change: int = 10
    tol: float = 1e-4


@dataclass
class RandomForestConfig:
    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_split: int = 2
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    cv_splits: int = 5

    def mlp_hparam_grid(self) -> List[Dict]:
        return [
            {
                "name": "compact",
                "hidden_layer_sizes": (256,),
                "learning_rate": 2e-3,
                "weight_decay": 5e-5,
                "dropout": 0.05,
                "epochs": 60,
            },
            {
                "name": "baseline",
                "hidden_layer_sizes": self.mlp.hidden_layer_sizes,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "dropout": 0.1,
                "epochs": 80,
            },
            {
                "name": "deep",
                "hidden_layer_sizes": (1024, 512, 256),
                "learning_rate": 5e-4,
                "weight_decay": 2e-4,
                "dropout": 0.15,
                "epochs": 110,
            },
        ]

    def rf_hparam_grid(self) -> List[Dict]:
        return [
            {
                "name": "shallow",
                "n_estimators": 200,
                "max_depth": 40,
                "min_samples_split": 2,
            },
            {
                "name": "baseline",
                "n_estimators": self.random_forest.n_estimators,
                "max_depth": self.random_forest.max_depth,
                "min_samples_split": self.random_forest.min_samples_split,
            },
            {
                "name": "regularized",
                "n_estimators": 600,
                "max_depth": 60,
                "min_samples_split": 4,
            },
        ]
