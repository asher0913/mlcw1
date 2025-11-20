"""CLI entry point that orchestrates tasks 1-3 end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .data import load_cifar10
from .features import FeatureSet, create_feature_sets, dump_feature_metadata
from .task2_mlp_torch import run_feature_dimension_experiment as mlp_feature_sweep
from .task2_mlp_torch import run_hparam_experiment as mlp_hparam_sweep
from .task3_cnn import (
    run_feature_variant_experiment as cnn_feature_sweep,
)
from .task3_cnn import run_hparam_experiment as cnn_hparam_sweep
from .utils import ensure_dir, set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs all coursework experiments for COMP3055."
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--pca-targets", nargs="+", type=int, default=[10, 30, 50, 70, 100])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--skip-task2", action="store_true")
    parser.add_argument("--skip-task3", action="store_true")

    # MLP specific knobs
    parser.add_argument("--mlp-hidden", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--mlp-activation", type=str, default="relu")
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-max-iter", type=int, default=80)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument("--mlp-no-early-stop", action="store_true")
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-pca-patience", type=int, default=10)

    # Task 3 (only CNN) knobs
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=5e-4)
    parser.add_argument("--cnn-dropout", type=float, default=0.1)
    parser.add_argument("--cnn-momentum", type=float, default=0.9)
    parser.add_argument("--cnn-num-workers", type=int, default=4)
    return parser.parse_args()


def _select_feature_set(feature_sets: Sequence[FeatureSet], name: str) -> FeatureSet:
    for fs in feature_sets:
        if fs.name == name:
            return fs
    raise ValueError(f"Feature set '{name}' not found. Available: {[fs.name for fs in feature_sets]}")


def main() -> None:
    args = _parse_args()
    set_global_seed(args.random_seed)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    print("=== Loading CIFAR-10 dataset ===")
    dataset = load_cifar10(
        data_root=args.data_root,
        train_subset=args.train_subset if args.train_subset > 0 else None,
        test_subset=args.test_subset if args.test_subset > 0 else None,
        random_state=args.random_seed,
    )
    print(f"Train samples: {len(dataset.X_train)} | Test samples: {len(dataset.X_test)}")

    print("=== Building PCA feature sets ===")
    feature_sets = create_feature_sets(
        dataset.X_train,
        dataset.X_test,
        args.pca_targets,
        random_state=args.random_seed,
    )
    dump_feature_metadata(feature_sets, output_root / "task1" / "feature_metadata.json")

    mlp_params = {
        "hidden_layer_sizes": tuple(args.mlp_hidden),
        "activation": args.mlp_activation,
        "learning_rate_init": args.mlp_lr,
        "alpha": args.mlp_alpha,
        "batch_size": args.mlp_batch_size,
        "max_iter": args.mlp_max_iter,
        "early_stopping": not args.mlp_no_early_stop,
        "n_iter_no_change": args.mlp_patience,
        "dropout": args.mlp_dropout,
        "weight_decay": args.mlp_weight_decay,
        "epochs": args.mlp_epochs,
        "patience": args.mlp_pca_patience,
        "learning_rate": args.mlp_lr,
    }
    cnn_params = {
        "epochs": args.cnn_epochs,
        "batch_size": args.cnn_batch_size,
        "learning_rate": args.cnn_lr,
        "weight_decay": args.cnn_weight_decay,
        "dropout": args.cnn_dropout,
        "momentum": args.cnn_momentum,
        "num_workers": args.cnn_num_workers,
    }

    original_features = _select_feature_set(feature_sets, "original")

    if not args.skip_task2:
        print("=== Running Task 2: MLP feature sweep ===")
        mlp_feature_sweep(
            feature_sets=feature_sets,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            class_names=dataset.class_names,
            base_params=mlp_params,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed,
        )
        print("=== Running Task 2: MLP hyper parameter sweep ===")
        mlp_hparam_grid = [
            {
                "name": "compact",
                "hidden_layer_sizes": (256,),
                "learning_rate": args.mlp_lr * 2,
                "weight_decay": args.mlp_weight_decay * 0.5,
                "dropout": args.mlp_dropout * 0.8,
                "epochs": max(40, args.mlp_epochs - 20),
            },
            {
                "name": "baseline",
                "hidden_layer_sizes": tuple(args.mlp_hidden),
                "learning_rate": args.mlp_lr,
                "weight_decay": args.mlp_weight_decay,
                "dropout": args.mlp_dropout,
                "epochs": args.mlp_epochs,
            },
            {
                "name": "deep",
                "hidden_layer_sizes": tuple(list(args.mlp_hidden) + [128]),
                "learning_rate": args.mlp_lr / 2,
                "weight_decay": args.mlp_weight_decay * 2,
                "dropout": args.mlp_dropout + 0.05,
                "epochs": args.mlp_epochs + 30,
            },
        ]
        mlp_hparam_sweep(
            feature_set=original_features,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            class_names=dataset.class_names,
            base_params=mlp_params,
            hparam_grid=mlp_hparam_grid,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed,
        )
    else:
        print("Skipping Task 2 as requested.")

    if not args.skip_task3:
        print("=== Running Task 3 (CNN): augmentation sweep ===")
        cnn_feature_sweep(
            train_images=dataset.train_images,
            train_labels=dataset.y_train,
            test_images=dataset.test_images,
            test_labels=dataset.y_test,
            class_names=dataset.class_names,
            base_params=cnn_params,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed + 101,
        )
        print("=== Running Task 3 (CNN): hyper parameter sweep ===")
        cnn_hparam_grid = [
            {
                "name": "fast",
                "epochs": max(10, args.cnn_epochs - 10),
                "learning_rate": args.cnn_lr * 2,
                "weight_decay": args.cnn_weight_decay / 2,
                "dropout": max(0.0, args.cnn_dropout - 0.05),
            },
            {
                "name": "baseline",
                "epochs": args.cnn_epochs,
                "learning_rate": args.cnn_lr,
                "weight_decay": args.cnn_weight_decay,
                "dropout": args.cnn_dropout,
            },
            {
                "name": "regularized",
                "epochs": args.cnn_epochs + 10,
                "learning_rate": args.cnn_lr * 0.6,
                "weight_decay": args.cnn_weight_decay * 2,
                "dropout": args.cnn_dropout + 0.1,
            },
        ]
        cnn_hparam_sweep(
            train_images=dataset.train_images,
            train_labels=dataset.y_train,
            test_images=dataset.test_images,
            test_labels=dataset.y_test,
            class_names=dataset.class_names,
            base_params=cnn_params,
            hparam_grid=cnn_hparam_grid,
            output_dir=output_root,
            cv_splits=args.cv_splits,
            random_state=args.random_seed + 211,
        )
    else:
        print("Skipping Task 3 as requested.")

    print("All requested experiments have completed.")


if __name__ == "__main__":
    main()
