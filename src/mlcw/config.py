"""Configuration dataclasses used across the coursework experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class DataConfig:
    """Holds dataset loading related options."""

    data_root: str = "data"
    train_subset: int | None = 10000
    test_subset: int | None = 2000
    random_state: int = 42


@dataclass
class FeatureConfig:
    """Defines the PCA compression levels expressed as percentages."""

    pca_targets: Tuple[int, ...] = (10, 30, 50, 70, 100)


@dataclass
class MLPConfig:
    """Default MLP hyper parameters shared across experiments."""

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
    """Default Random Forest hyper parameters for the alternative model."""

    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_split: int = 2
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class ExperimentConfig:
    """Groups all experiment knobs for convenient passing between functions."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    cv_splits: int = 5

    def mlp_hparam_grid(self) -> List[Dict]:
        """Preset list of MLP hyper parameter sweeps for task 2.3."""
        return [
            {
                "name": "compact",
                "hidden_layer_sizes": (256,),
                "learning_rate_init": 5e-3,
                "alpha": 5e-4,
                "max_iter": 60,
            },
            {
                "name": "baseline",
                "hidden_layer_sizes": self.mlp.hidden_layer_sizes,
                "learning_rate_init": self.mlp.learning_rate_init,
                "alpha": self.mlp.alpha,
                "max_iter": self.mlp.max_iter,
            },
            {
                "name": "deep",
                "hidden_layer_sizes": (1024, 512, 256),
                "learning_rate_init": 5e-4,
                "alpha": 1e-4,
                "max_iter": 120,
            },
        ]

    def rf_hparam_grid(self) -> List[Dict]:
        """Preset list of Random Forest hyper parameter sweeps for task 3."""
        return [
            {
                "name": "shallow",
                "n_estimators": 200,
                "max_depth": 40,
            },
            {
                "name": "baseline",
                "n_estimators": self.random_forest.n_estimators,
                "max_depth": self.random_forest.max_depth,
            },
            {
                "name": "regularized",
                "n_estimators": 600,
                "max_depth": 60,
                "min_samples_split": 4,
            },
        ]
