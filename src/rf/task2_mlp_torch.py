"""Task 2: GPU MLP experiments for the RF package."""

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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from .features import FeatureSet
from .metrics import classification_metrics
from .plotting import plot_metric_curve, plot_param_bar
from .utils import ensure_dir, save_json, get_device_context, seed_torch


def _require_gpu():
    ctx = get_device_context(require_gpu=True)
    if ctx.backend == "cuda":
        torch.backends.cudnn.benchmark = True
    return ctx


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        dropout: float,
        num_classes: int,
        use_batchnorm: bool = True,
        use_gelu: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU() if use_gelu else nn.ReLU(inplace=True))
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
    use_batchnorm: bool
    max_grad_norm: float | None = None
    use_gelu: bool = True
    use_sgd: bool = False


def _make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)
    train_ds: Dataset = TensorDataset(X_train, y_train)
    val_ds: Dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return train_loader, val_loader


def _train_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    params: TrainParams,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    non_blocking: bool,
    pin_memory: bool,
    progress_desc: str | None = None,
) -> tuple[float, float]:
    train_loader, val_loader = _make_loaders(
        X, y, train_idx, val_idx, params.batch_size, pin_memory=pin_memory
    )
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_sizes=params.hidden_sizes,
        dropout=params.dropout,
        num_classes=len(np.unique(y)),
        use_batchnorm=params.use_batchnorm,
        use_gelu=params.use_gelu,
    ).to(device)
    if params.use_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=params.lr, momentum=0.9, weight_decay=params.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.lr, weight_decay=params.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, params.patience // 3),
    )
    best_acc = 0.0
    best_f1 = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    epoch_bar = tqdm(
        range(params.epochs),
        desc=progress_desc or "MLP Epochs",
        leave=False,
        unit="epoch",
    )
    class_labels = list(np.unique(y))
    for _ in epoch_bar:
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            if params.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()

        model.eval()
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        val_acc = float(np.mean(y_pred == y_true))
        val_f1 = float(
            classification_metrics(
                y_true,
                y_pred,
                class_names=class_labels,
                label_indices=class_labels,
            )["macro_f1"]
        )
        scheduler.step(val_acc)
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
    non_blocking: bool,
    pin_memory: bool,
    progress_desc: str | None = None,
):
    seed_torch(seed)
    # 创建内部验证集用于早停/调度（10%）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    (train_idx, val_idx) = next(sss.split(X_train, y_train))

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train[train_idx], dtype=torch.float32), torch.tensor(y_train[train_idx], dtype=torch.long)),
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_train[val_idx], dtype=torch.float32), torch.tensor(y_train[val_idx], dtype=torch.long)),
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_sizes=params.hidden_sizes,
        dropout=params.dropout,
        num_classes=len(class_names),
        use_batchnorm=params.use_batchnorm,
        use_gelu=params.use_gelu,
    ).to(device)
    if params.use_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=params.lr, momentum=0.9, weight_decay=params.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.lr, weight_decay=params.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, params.patience // 2),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    epoch_bar = tqdm(
        range(params.epochs),
        desc=progress_desc or "MLP Full Train",
        leave=False,
        unit="epoch",
    )
    for _ in epoch_bar:
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            if params.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()

        # 验证集评估
        model.eval()
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
                preds = model(xb).argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        y_pred_val = np.concatenate(all_preds)
        y_true_val = np.concatenate(all_labels)
        val_acc = float(np.mean(y_pred_val == y_true_val))
        scheduler.step(val_acc)
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= params.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking)
            preds = model(xb).argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = classification_metrics(
        y_true,
        y_pred,
        class_names,
        label_indices=list(range(len(class_names))),
    )
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
    ctx = _require_gpu()
    device = ctx.device
    sweep_dir = ensure_dir(output_dir / "task2" / "mlp_feature_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics_json: Dict[str, Dict] = {}
    plot_points = []

    train_params = TrainParams(
        hidden_sizes=tuple(base_params.get("hidden_layer_sizes", (1024, 512))),
        dropout=base_params.get("dropout", 0.3),
        lr=base_params.get("learning_rate", 1e-3),
        weight_decay=base_params.get("weight_decay", 1e-4),
        batch_size=base_params.get("batch_size", 256),
        epochs=base_params.get("epochs", 80),
        patience=base_params.get("patience", 10),
        use_batchnorm=base_params.get("use_batchnorm", True),
        max_grad_norm=base_params.get("max_grad_norm", 1.0),
        use_gelu=base_params.get("use_gelu", True),
        use_sgd=base_params.get("use_sgd", False),
    )

    feature_progress = tqdm(
        enumerate(feature_sets),
        total=len(feature_sets),
        desc="Task2: MLP 特征",
        unit="set",
    )
    for idx, feat in feature_progress:
        seed = random_state + idx * 97
        seed_torch(seed)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        fold_progress = tqdm(
            total=cv_splits,
            desc=f"[Task2][{feat.name}] Folds",
            leave=False,
            unit="fold",
        )
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(feat.X_train, y_train), start=1):
            val_acc, val_f1 = _train_one_fold(
                feat.X_train,
                y_train,
                feat.n_features,
                train_params,
                tr_idx,
                val_idx,
                device,
                ctx.non_blocking,
                ctx.pin_memory,
                progress_desc=f"[Task2][{feat.name}] Fold {fold_idx}",
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
            train_params,
            device,
            class_names,
            seed + 999,
            ctx.non_blocking,
            ctx.pin_memory,
            progress_desc=f"[Task2][{feat.name}] Test Fit",
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
    ctx = _require_gpu()
    device = ctx.device
    sweep_dir = ensure_dir(output_dir / "task2" / "mlp_hparam_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics_json: Dict[str, Dict] = {}
    labels_plot = []
    values_plot = []

    config_progress = tqdm(
        enumerate(hparam_grid),
        total=len(hparam_grid),
        desc="Task2: MLP 超参",
        unit="cfg",
    )
    for idx, cfg in config_progress:
        name = cfg.get("name", f"cfg_{idx}")
        params = TrainParams(
            hidden_sizes=cfg.get("hidden_layer_sizes", base_params.get("hidden_layer_sizes", (1024, 512))),
            dropout=cfg.get("dropout", base_params.get("dropout", 0.3)),
            lr=cfg.get("learning_rate", base_params.get("learning_rate", 1e-3)),
            weight_decay=cfg.get("weight_decay", base_params.get("weight_decay", 1e-4)),
            batch_size=cfg.get("batch_size", base_params.get("batch_size", 256)),
            epochs=cfg.get("epochs", base_params.get("epochs", 80)),
            patience=cfg.get("patience", base_params.get("patience", 10)),
            use_batchnorm=cfg.get("use_batchnorm", base_params.get("use_batchnorm", True)),
            max_grad_norm=cfg.get("max_grad_norm", base_params.get("max_grad_norm", 1.0)),
            use_gelu=cfg.get("use_gelu", base_params.get("use_gelu", True)),
            use_sgd=cfg.get("use_sgd", base_params.get("use_sgd", False)),
        )
        seed = random_state + idx * 131
        seed_torch(seed)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        fold_progress = tqdm(
            total=cv_splits,
            desc=f"[Task2][{name}] Folds",
            leave=False,
            unit="fold",
        )
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(feature_set.X_train, y_train), start=1):
            val_acc, val_f1 = _train_one_fold(
                feature_set.X_train,
                y_train,
                feature_set.n_features,
                params,
                tr_idx,
                val_idx,
                device,
                ctx.non_blocking,
                ctx.pin_memory,
                progress_desc=f"[Task2][{name}] Fold {fold_idx}",
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
            device,
            class_names,
            seed + 777,
            ctx.non_blocking,
            ctx.pin_memory,
            progress_desc=f"[Task2][{name}] Test Fit",
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
