"""Task 2: GPU MLP experiments (feature sweep + hyper-parameter sweep)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .features import FeatureSet
from .metrics import classification_metrics
from .plotting import plot_metric_curve, plot_param_bar
from .utils import ensure_dir, save_json, get_torch_device


def _require_gpu() -> torch.device:
    device, backend = get_torch_device(require_gpu=True)
    if backend == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


class MLPClassifier(nn.Module):
    """简单的多层感知机，用于处理 PCA 后的扁平特征。"""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float, num_classes: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainParams:
    hidden_sizes: Sequence[int]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int


def _make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)
    train_ds: Dataset = TensorDataset(X_train, y_train)
    val_ds: Dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def _train_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    params: TrainParams,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    train_loader, val_loader = _make_loaders(X, y, train_idx, val_idx, params.batch_size)
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_sizes=params.hidden_sizes,
        dropout=params.dropout,
        num_classes=len(np.unique(y)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    best_acc = 0.0
    best_f1 = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for _ in range(params.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        val_acc = float(np.mean(y_pred == y_true))
        val_f1 = float(
            classification_metrics(y_true, y_pred, class_names=list(range(len(np.unique(y)))))["macro_f1"]
        )
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= params.patience:
                break

    model.load_state_dict(best_state)
    return best_acc, best_f1


def _evaluate_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: TrainParams,
    device: torch.device,
    class_names: Sequence[str],
    seed: int,
):
    torch.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_sizes=params.hidden_sizes,
        dropout=params.dropout,
        num_classes=len(class_names),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    for _ in range(params.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
        # 简单早停：用训练集准确率跟踪
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
            train_acc = correct / total
        if train_acc > best_acc + 1e-4:
            best_acc = train_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= params.patience:
                break

    model.load_state_dict(best_state)
    # 测试集评估
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            preds = model(xb).argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = classification_metrics(y_true, y_pred, class_names)
    return model, metrics


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
    device = _require_gpu()
    sweep_dir = ensure_dir(output_dir / "task2" / "mlp_feature_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics_json: Dict[str, Dict] = {}
    plot_points = []

    train_params = TrainParams(
        hidden_sizes=tuple(base_params.get("hidden_layer_sizes", (512, 256))),
        dropout=base_params.get("dropout", 0.1),
        lr=base_params.get("learning_rate", 1e-3),
        weight_decay=base_params.get("weight_decay", 1e-4),
        batch_size=base_params.get("batch_size", 256),
        epochs=base_params.get("epochs", 60),
        patience=base_params.get("patience", 10),
    )

    for idx, feat in enumerate(feature_sets):
        seed = random_state + idx * 97
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(feat.X_train, y_train), start=1):
            val_acc, val_f1 = _train_one_fold(
                feat.X_train, y_train, feat.n_features, train_params, tr_idx, val_idx, device
            )
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "feature_set": feat.name,
                    "n_features": feat.n_features,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                }
            )
        cv_rows.extend(fold_metrics)
        mean_val_acc = float(np.mean([r["val_accuracy"] for r in fold_metrics]))
        mean_val_f1 = float(np.mean([r["val_macro_f1"] for r in fold_metrics]))

        _, test_metrics = _evaluate_test(
            feat.X_train,
            y_train,
            feat.X_test,
            y_test,
            train_params,
            device,
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
        title="GPU MLP accuracy vs PCA dimensionality",
        output_path=output_dir / "task2" / "plots" / "mlp_feature_accuracy.png",
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
    device = _require_gpu()
    sweep_dir = ensure_dir(output_dir / "task2" / "mlp_hparam_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics_json: Dict[str, Dict] = {}
    labels_plot = []
    values_plot = []

    for idx, cfg in enumerate(hparam_grid):
        name = cfg.get("name", f"cfg_{idx}")
        params = TrainParams(
            hidden_sizes=cfg.get("hidden_layer_sizes", base_params.get("hidden_layer_sizes", (512, 256))),
            dropout=cfg.get("dropout", base_params.get("dropout", 0.1)),
            lr=cfg.get("learning_rate", base_params.get("learning_rate", 1e-3)),
            weight_decay=cfg.get("weight_decay", base_params.get("weight_decay", 1e-4)),
            batch_size=cfg.get("batch_size", base_params.get("batch_size", 256)),
            epochs=cfg.get("epochs", base_params.get("epochs", 80)),
            patience=cfg.get("patience", base_params.get("patience", 10)),
        )
        seed = random_state + idx * 131
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(feature_set.X_train, y_train), start=1):
            val_acc, val_f1 = _train_one_fold(
                feature_set.X_train,
                y_train,
                feature_set.n_features,
                params,
                tr_idx,
                val_idx,
                device,
            )
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "config_name": name,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                }
            )
        cv_rows.extend(fold_metrics)
        mean_val_acc = float(np.mean([r["val_accuracy"] for r in fold_metrics]))
        mean_val_f1 = float(np.mean([r["val_macro_f1"] for r in fold_metrics]))

        _, test_metrics = _evaluate_test(
            feature_set.X_train,
            y_train,
            feature_set.X_test,
            y_test,
            params,
            device,
            class_names,
            seed + 777,
        )
        test_metrics_json[name] = test_metrics
        save_json(test_metrics, reports_dir / f"{name}_test_metrics.json")
        summary_rows.append(
            {
                "config_name": name,
                "hidden_layer_sizes": str(params.hidden_sizes),
                "learning_rate": params.lr,
                "weight_decay": params.weight_decay,
                "dropout": params.dropout,
                "epochs": params.epochs,
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
        title="GPU MLP test accuracy vs hyper-parameters",
        ylabel="Test Accuracy",
        output_path=output_dir / "task2" / "plots" / "mlp_hparam_accuracy.png",
    )
    return summary_df
