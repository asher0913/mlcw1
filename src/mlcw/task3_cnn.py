"""Implements Task 3 using a CNN (ResNet18) trained on raw CIFAR-10 images."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

from .metrics import classification_metrics
from .plotting import plot_param_bar
from .utils import ensure_dir, save_json, get_torch_device, seed_torch, empty_device_cache

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class NumpyCIFARDataset(Dataset):
    """Dataset wrapper that applies torchvision transforms on demand."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[self.indices[idx]])
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.labels[self.indices[idx]])
        return image, label


def _build_dataloader(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = NumpyCIFARDataset(images, labels, indices, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def _build_model(num_classes: int, dropout: float = 0.0) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )  # better for 32x32
    model.maxpool = nn.Identity()
    if dropout > 0:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    preds: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            labels = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds.append(outputs.argmax(dim=1).cpu().numpy())
            targets_all.append(labels.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets_all, axis=0)
    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return running_loss / total, accuracy, macro_f1


def _train_full_model(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    transform_train,
    params: Dict,
    device: torch.device,
    progress_desc: str | None = None,
) -> nn.Module:
    train_indices = np.arange(len(train_labels))
    train_loader = _build_dataloader(
        train_images,
        train_labels,
        train_indices,
        transform_train,
        params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )
    model = _build_model(num_classes=10, dropout=params.get("dropout", 0.0)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=params["learning_rate"],
        momentum=params.get("momentum", 0.9),
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
    epoch_bar = tqdm(
        range(params["epochs"]),
        desc=progress_desc or "CNN Full Train",
        leave=False,
        unit="epoch",
    )
    for _ in epoch_bar:
        _train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
    return model


def _require_gpu() -> torch.device:
    device, backend = get_torch_device(require_gpu=True)
    if backend == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def _feature_variants():
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    standard_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    strong_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return [
        {
            "name": "standard",
            "train_transform": standard_train,
            "eval_transform": eval_transform,
            "description": "Random crop + horizontal flip.",
        },
        {
            "name": "strong_aug",
            "train_transform": strong_train,
            "eval_transform": eval_transform,
            "description": "Adds color jitter and random erasing.",
        },
    ]


def run_feature_variant_experiment(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: Sequence[str],
    base_params: Dict,
    output_dir: Path,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    device = _require_gpu()
    variant_dir = ensure_dir(output_dir / "task3" / "cnn_feature_sweep")
    reports_dir = ensure_dir(variant_dir / "reports")
    feature_variants = _feature_variants()
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    metrics_json: Dict[str, Dict] = {}
    plot_labels: List[str] = []
    plot_values: List[float] = []

    variant_progress = tqdm(
        enumerate(feature_variants),
        total=len(feature_variants),
        desc="Task3: CNN 增广",
        unit="variant",
    )
    for idx, variant in variant_progress:
        seed = random_state + idx * 503
        seed_torch(seed)
        transformer_train = variant["train_transform"]
        transformer_eval = variant["eval_transform"]

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        fold_progress = tqdm(
            total=cv_splits,
            desc=f"[Task3][{variant['name']}] Folds",
            leave=False,
            unit="fold",
        )
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels), start=1):
            seed_torch(seed + fold_idx)
            train_loader = _build_dataloader(
                train_images,
                train_labels,
                train_idx,
                transformer_train,
                base_params["batch_size"],
                shuffle=True,
                num_workers=base_params["num_workers"],
            )
            val_loader = _build_dataloader(
                train_images,
                train_labels,
                val_idx,
                transformer_eval,
                base_params["batch_size"],
                shuffle=False,
                num_workers=base_params["num_workers"],
            )
            model = _build_model(num_classes=len(class_names), dropout=base_params.get("dropout", 0.0)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=base_params["learning_rate"],
                momentum=base_params.get("momentum", 0.9),
                weight_decay=base_params["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=base_params["epochs"]
            )
            best_state = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            best_macro_f1 = 0.0
            epoch_bar = tqdm(
                range(base_params["epochs"]),
                desc=f"[Task3][{variant['name']}][Fold {fold_idx}]",
                leave=False,
                unit="epoch",
            )
            for _ in epoch_bar:
                _train_one_epoch(model, train_loader, criterion, optimizer, device)
                _, val_acc, val_macro_f1 = _evaluate(model, val_loader, criterion, device)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_macro_f1 = val_macro_f1
                    best_state = copy.deepcopy(model.state_dict())
                scheduler.step()
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "feature_variant": variant["name"],
                    "val_accuracy": best_acc,
                    "val_macro_f1": best_macro_f1,
                }
            )
            del model
            empty_device_cache()
            fold_progress.update(1)
        fold_progress.close()

        cv_rows.extend(fold_metrics)
        mean_val_acc = float(np.mean([row["val_accuracy"] for row in fold_metrics]))
        mean_val_macro_f1 = float(np.mean([row["val_macro_f1"] for row in fold_metrics]))

        final_model = _train_full_model(
            train_images,
            train_labels,
            transformer_train,
            base_params,
            device,
            progress_desc=f"[Task3][{variant['name']}] Test Fit",
        )
        test_loader = _build_dataloader(
            test_images,
            test_labels,
            np.arange(len(test_labels)),
            transformer_eval,
            base_params["batch_size"],
            shuffle=False,
            num_workers=base_params["num_workers"],
        )
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, _ = _evaluate(final_model, test_loader, criterion, device)
        preds, truths = _collect_predictions(final_model, test_loader, device)
        test_metrics = classification_metrics(truths, preds, class_names)
        metrics_json[variant["name"]] = test_metrics
        save_json(test_metrics, reports_dir / f"{variant['name']}_test_metrics.json")
        summary_rows.append(
            {
                "feature_variant": variant["name"],
                "description": variant["description"],
                "mean_val_accuracy": mean_val_acc,
                "mean_val_macro_f1": mean_val_macro_f1,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_loss": test_loss,
            }
        )
        plot_labels.append(variant["name"])
        plot_values.append(test_metrics["accuracy"])
        del final_model
        empty_device_cache()

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not cv_df.empty:
        cv_df.to_csv(variant_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(variant_dir / "summary.csv", index=False)
    save_json(metrics_json, variant_dir / "test_metrics.json")
    plot_param_bar(
        labels=plot_labels,
        values=plot_values,
        title="CNN test accuracy vs augmentation strength",
        ylabel="Test Accuracy",
        output_path=output_dir / "task3" / "plots" / "cnn_feature_accuracy.png",
    )
    return summary_df


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            preds.append(outputs.argmax(dim=1).cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


def run_hparam_experiment(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: Sequence[str],
    base_params: Dict,
    hparam_grid: Sequence[Dict],
    output_dir: Path,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    device = _require_gpu()
    sweep_dir = ensure_dir(output_dir / "task3" / "cnn_hparam_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    base_variants = _feature_variants()
    baseline_transform = base_variants[0]["train_transform"]
    eval_transform = base_variants[0]["eval_transform"]

    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    metrics_json: Dict[str, Dict] = {}
    plot_labels: List[str] = []
    plot_values: List[float] = []

    config_progress = tqdm(
        enumerate(hparam_grid),
        total=len(hparam_grid),
        desc="Task3: CNN 超参",
        unit="cfg",
    )
    for idx, config in config_progress:
        name = config.get("name", f"cnn_config_{idx}")
        params = {**base_params, **{k: v for k, v in config.items() if k != "name"}}
        seed = random_state + idx * 701
        seed_torch(seed)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        fold_metrics: List[Dict] = []
        fold_progress = tqdm(
            total=cv_splits,
            desc=f"[Task3][{name}] Folds",
            leave=False,
            unit="fold",
        )
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels), start=1):
            seed_torch(seed + fold_idx)
            train_loader = _build_dataloader(
                train_images,
                train_labels,
                train_idx,
                baseline_transform,
                params["batch_size"],
                shuffle=True,
                num_workers=params["num_workers"],
            )
            val_loader = _build_dataloader(
                train_images,
                train_labels,
                val_idx,
                eval_transform,
                params["batch_size"],
                shuffle=False,
                num_workers=params["num_workers"],
            )
            model = _build_model(
                num_classes=len(class_names), dropout=params.get("dropout", 0.0)
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                momentum=params.get("momentum", 0.9),
                weight_decay=params["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])
            best_state = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            best_macro_f1 = 0.0
            epoch_bar = tqdm(
                range(params["epochs"]),
                desc=f"[Task3][{name}][Fold {fold_idx}]",
                leave=False,
                unit="epoch",
            )
            for _ in epoch_bar:
                _train_one_epoch(model, train_loader, criterion, optimizer, device)
                _, val_acc, val_macro_f1 = _evaluate(model, val_loader, criterion, device)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_macro_f1 = val_macro_f1
                    best_state = copy.deepcopy(model.state_dict())
                scheduler.step()
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "config_name": name,
                    "val_accuracy": best_acc,
                    "val_macro_f1": best_macro_f1,
                }
            )
            del model
            empty_device_cache()
            fold_progress.update(1)
        fold_progress.close()
        cv_rows.extend(fold_metrics)
        mean_val_acc = float(np.mean([row["val_accuracy"] for row in fold_metrics]))
        mean_val_macro_f1 = float(np.mean([row["val_macro_f1"] for row in fold_metrics]))

        final_model = _train_full_model(
            train_images,
            train_labels,
            baseline_transform,
            params,
            device,
            progress_desc=f"[Task3][{name}] Test Fit",
        )
        test_loader = _build_dataloader(
            test_images,
            test_labels,
            np.arange(len(test_labels)),
            eval_transform,
            params["batch_size"],
            shuffle=False,
            num_workers=params["num_workers"],
        )
        criterion = nn.CrossEntropyLoss()
        test_loss, _, _ = _evaluate(final_model, test_loader, criterion, device)
        preds, truths = _collect_predictions(final_model, test_loader, device)
        test_metrics = classification_metrics(truths, preds, class_names)
        metrics_json[name] = test_metrics
        save_json(test_metrics, reports_dir / f"{name}_test_metrics.json")
        summary_rows.append(
            {
                "config_name": name,
                "epochs": params["epochs"],
                "learning_rate": params["learning_rate"],
                "weight_decay": params["weight_decay"],
                "dropout": params.get("dropout", 0.0),
                "mean_val_accuracy": mean_val_acc,
                "mean_val_macro_f1": mean_val_macro_f1,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_loss": test_loss,
            }
        )
        plot_labels.append(name)
        plot_values.append(test_metrics["accuracy"])
        del final_model
        empty_device_cache()

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not cv_df.empty:
        cv_df.to_csv(sweep_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(sweep_dir / "summary.csv", index=False)
    save_json(metrics_json, sweep_dir / "test_metrics.json")
    plot_param_bar(
        labels=plot_labels,
        values=plot_values,
        title="CNN test accuracy vs hyper-parameters",
        ylabel="Test Accuracy",
        output_path=output_dir / "task3" / "plots" / "cnn_hparam_accuracy.png",
    )
    return summary_df
