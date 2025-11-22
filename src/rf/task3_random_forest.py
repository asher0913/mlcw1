"""Task 3: Random Forest experiments mirroring Task 2 structure."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from .features import FeatureSet
from .metrics import classification_metrics
from .plotting import plot_metric_curve, plot_param_bar
from .utils import ensure_dir, save_json


def _train_rf(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: Dict,
    random_state: int,
) -> tuple[float, float]:
    clf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features"),
        max_samples=params.get("max_samples"),
        class_weight=params.get("class_weight"),
        n_jobs=params.get("n_jobs", -1),
        random_state=random_state,
        bootstrap=True,
    )
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[val_idx])
    y_true = y[val_idx]
    labels = sorted(np.unique(y))
    metrics = classification_metrics(
        y_true,
        preds,
        class_names=[str(lbl) for lbl in labels],
        label_indices=labels,
    )
    return float(np.mean(preds == y_true)), float(metrics["macro_f1"])


def _evaluate_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict,
    class_names: Sequence[str],
    random_state: int,
):
    clf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features"),
        max_samples=params.get("max_samples"),
        class_weight=params.get("class_weight"),
        n_jobs=params.get("n_jobs", -1),
        random_state=random_state,
        bootstrap=True,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    labels = sorted(np.unique(np.concatenate([y_train, y_test])))
    metrics = classification_metrics(
        y_test,
        preds,
        class_names=[str(lbl) for lbl in labels],
        label_indices=labels,
    )
    return clf, metrics


def run_feature_dimension_experiment(
    feature_sets: Sequence[FeatureSet],
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    base_params: Dict,
    output_dir: Path,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    sweep_dir = ensure_dir(output_dir / "task3" / "rf_feature_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics_json: Dict[str, Dict] = {}
    plot_points = []

    params = {
        "n_estimators": base_params.get("n_estimators", 400),
        "max_depth": base_params.get("max_depth"),
        "min_samples_split": base_params.get("min_samples_split", 2),
        "n_jobs": base_params.get("n_jobs", -1),
    }

    feature_progress = tqdm(
        enumerate(feature_sets),
        total=len(feature_sets),
        desc="Task3: RF 特征",
        unit="set",
    )
    for idx, feat in feature_progress:
        seed = random_state + idx * 211
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        fold_progress = tqdm(
            total=cv_splits,
            desc=f"[Task3-RF][{feat.name}] Folds",
            leave=False,
            unit="fold",
        )
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(feat.X_train, y_train), start=1):
            val_acc, val_f1 = _train_rf(
                feat.X_train,
                y_train,
                tr_idx,
                val_idx,
                params,
                seed + fold_idx,
            )
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "feature_set": feat.name,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                }
            )
            fold_progress.update(1)
        fold_progress.close()
        cv_rows.extend(fold_metrics)
        mean_val_acc = float(np.mean([r["val_accuracy"] for r in fold_metrics]))
        mean_val_f1 = float(np.mean([r["val_macro_f1"] for r in fold_metrics]))

        _, test_metrics = _evaluate_test(
            feat.X_train,
            y_train,
            feat.X_test,
            y_test,
            params,
            class_names,
            seed + 999,
        )
        test_metrics_json[feat.name] = test_metrics
        save_json(test_metrics, reports_dir / f"{feat.name}_test_metrics.json")
        summary_rows.append(
            {
                "feature_set": feat.name,
                "n_features": feat.n_features,
                "explained_variance": feat.explained_variance,
                "mean_val_accuracy": mean_val_acc,
                "mean_val_macro_f1": mean_val_f1,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
        plot_points.append((feat.n_features, test_metrics["accuracy"], feat.name))

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not cv_df.empty:
        cv_df.to_csv(sweep_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(sweep_dir / "summary.csv", index=False)
    save_json(test_metrics_json, sweep_dir / "test_metrics.json")
    plot_metric_curve(
        xs=[p[0] for p in plot_points],
        ys=[p[1] for p in plot_points],
        labels=[p[2] for p in plot_points],
        xlabel="Feature Dimension",
        ylabel="Test Accuracy",
        title="Random Forest accuracy vs PCA dimensionality",
        output_path=output_dir / "task3" / "plots" / "rf_feature_accuracy.png",
    )
    return summary_df


def run_hparam_experiment(
    feature_set: FeatureSet,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    base_params: Dict,
    hparam_grid: Sequence[Dict],
    output_dir: Path,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    sweep_dir = ensure_dir(output_dir / "task3" / "rf_hparam_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics_json: Dict[str, Dict] = {}
    labels_plot: List[str] = []
    values_plot: List[float] = []

    config_progress = tqdm(
        enumerate(hparam_grid),
        total=len(hparam_grid),
        desc="Task3: RF 超参",
        unit="cfg",
    )
    for idx, cfg in config_progress:
        name = cfg.get("name", f"rf_config_{idx}")
        params = {
            "n_estimators": cfg.get("n_estimators", base_params.get("n_estimators", 400)),
            "max_depth": cfg.get("max_depth", base_params.get("max_depth")),
            "min_samples_split": cfg.get("min_samples_split", base_params.get("min_samples_split", 2)),
            "n_jobs": cfg.get("n_jobs", base_params.get("n_jobs", -1)),
        }
        seed = random_state + idx * 311
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        fold_progress = tqdm(
            total=cv_splits,
            desc=f"[Task3-RF][{name}] Folds",
            leave=False,
            unit="fold",
        )
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(feature_set.X_train, y_train), start=1):
            val_acc, val_f1 = _train_rf(
                feature_set.X_train,
                y_train,
                tr_idx,
                val_idx,
                params,
                seed + fold_idx,
            )
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "config_name": name,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                }
            )
            fold_progress.update(1)
        fold_progress.close()
        cv_rows.extend(fold_metrics)
        mean_val_acc = float(np.mean([r["val_accuracy"] for r in fold_metrics]))
        mean_val_f1 = float(np.mean([r["val_macro_f1"] for r in fold_metrics]))

        _, test_metrics = _evaluate_test(
            feature_set.X_train,
            y_train,
            feature_set.X_test,
            y_test,
            params,
            class_names,
            seed + 777,
        )
        test_metrics_json[name] = test_metrics
        save_json(test_metrics, reports_dir / f"{name}_test_metrics.json")
        summary_rows.append(
            {
                "config_name": name,
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "mean_val_accuracy": mean_val_acc,
                "mean_val_macro_f1": mean_val_f1,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
        labels_plot.append(name)
        values_plot.append(test_metrics["accuracy"])

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not cv_df.empty:
        cv_df.to_csv(sweep_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(sweep_dir / "summary.csv", index=False)
    save_json(test_metrics_json, sweep_dir / "test_metrics.json")
    plot_param_bar(
        labels=labels_plot,
        values=values_plot,
        title="Random Forest test accuracy vs hyper-parameters",
        ylabel="Test Accuracy",
        output_path=output_dir / "task3" / "plots" / "rf_hparam_accuracy.png",
    )
    return summary_df
