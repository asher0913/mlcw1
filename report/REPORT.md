# COMP3055 Machine Learning Coursework Report

> **Note:** This report is written as a reproducible template. Once you run `python main.py` on the server (GPU), replace the highlighted placeholders (✅ sections) with the actual numbers, tables, and figures from `outputs/`.

---

## 1. Problem Statement & Dataset Overview

The coursework evaluates end‑to‑end supervised learning workflows on the CIFAR‑10 dataset. CIFAR‑10 contains 60 000 RGB images (32×32) split over 10 mutually exclusive classes. Each class contributes exactly 6 000 images: 50 000 for training and 10 000 for testing. A randomly shuffled subset was used to keep local experimentation tractable; default settings sample **10 000** training and **2 000** testing images, although every experiment can be rerun with the full dataset by passing `--train-subset 0 --test-subset 0` to the runner.

Key dataset processing steps implemented in `src/mlcw/data.py`:

1. **Download & Cache** – `torchvision.datasets.CIFAR10` downloads the batches once and reuses the cached files under `data/`.
2. **Flattening & Normalisation** – Each image is converted to a 3 072‑dimensional vector (3 channels × 32 × 32) and linearly scaled to `[0, 1]` so that features share a comparable magnitude before standardisation.
3. **Subsampling (optional)** – Deterministic random sampling (`numpy.random.default_rng`) selects the requested number of samples to guarantee reproducibility across runs via the `--random-seed` flag.

The rest of the pipeline builds and evaluates two families of classifiers on the processed data. All experiments share the same random seed, 5‑fold stratified cross validation, and identical train/test splits to ensure apples‑to‑apples comparisons.

---

## 2. Task 1 – PCA Feature Engineering

### 2.1 Procedure

1. **Standardisation** – Both train and test matrices are standardised (`StandardScaler`) to zero mean and unit variance per feature.
2. **PCA Compression** – The scaler output feeds into PCA configured via percentages of retained variance: 10 %, 30 %, 50 %, 70 %, and 100 %. `n_components` is derived by multiplying the percentage by the total variance ratio (`svd_solver="full"`).
3. **Artifacts** – For each feature set we store:
   - Transformed train/test matrices (kept in memory for downstream experiments).
   - `outputs/task1/feature_metadata.json` summarising dimensionality and explained variance—useful for Task 4 comparisons.

### 2.2 Results (fill after running)

| Feature Set | Dimensionality | Explained Variance | Notes |
| --- | --- | --- | --- |
| original | 3 072 | 100 % | Standardised flattened RGB |
| pca_70 | ✅ | ✅ | PCA retains 70 % variance |
| pca_50 | ✅ | ✅ | 50 % variance |
| pca_30 | ✅ | ✅ | 30 % variance |
| pca_10 | ✅ | ✅ | 10 % variance |

*Numbers can be copied from `outputs/task1/feature_metadata.json`. Dimensionality is determined automatically by PCA and varies with the percentage retained; expect ~2 000 features for 70 % and ~300 for 10 %.*

### 2.3 Observations

- PCA dramatically reduces feature count without re‑computing the dataset for each experiment.
- Lower percentages accelerate training and mitigate overfitting risk but may lose discriminative information, which will be visible in downstream accuracy/F1.
- Using variance ratios rather than fixed components makes the script dataset‑agnostic (works for both subsets and full CIFAR‑10).

---

## 3. Task 2 – MLP Object Recognition System

### 3.1 Experimental Setup

- **Model** – `sklearn.neural_network.MLPClassifier` with default hidden layers `[512, 256]`, ReLU activations, Adam optimiser, `learning_rate_init=1e-3`, `alpha=1e-4`, batch size 256, early stopping patience 10 epochs, and maximum 80 iterations.
- **Cross Validation** – 5‑fold stratified CV on the training split for both feature‑dimension and hyper‑parameter sweeps.
- **Evaluation** – After CV, the model is retrained on the full training data and evaluated on the held‑out test subset。产物包括：
  - `fold_metrics.csv`
  - `summary.csv`
  - `reports/*.json`
  - `plots/mlp_feature_accuracy.png` / `mlp_hparam_accuracy.png`

### 3.2 Feature Dimension Sweep

**Files:** `outputs/task2/mlp_feature_sweep/summary.csv`, `.../reports/*.json`, `outputs/task2/plots/mlp_feature_accuracy.png`.

| Feature Set | Mean CV Accuracy | Mean CV Macro F1 | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- |
| original | ✅ | ✅ | ✅ | ✅ |
| pca_70 | ✅ | ✅ | ✅ | ✅ |
| pca_50 | ✅ | ✅ | ✅ | ✅ |
| pca_30 | ✅ | ✅ | ✅ | ✅ |
| pca_10 | ✅ | ✅ | ✅ | ✅ |

**How to populate:** Run `python main.py`, open `summary.csv`, and paste the corresponding numbers. The accuracy vs. dimensionality plot (`mlp_feature_accuracy.png`) provides a visual summary that can be embedded in the final PDF report.

**Expected trend (justify with your numbers):**

- Accuracy typically improves monotonically with more retained variance. Expect a significant drop at 10 % due to aggressive compression.
- CV macro F1 closely follows test macro F1, indicating low overfitting in the default configuration.
- If using the full dataset, the gaps may shrink because the MLP benefits from more data even with fewer features.

### 3.3 Hyper‑Parameter Sweep

**Configs implemented:**

1. **compact** – `(256,)` hidden layers, higher LR (`5e-3`), stronger regularisation (alpha `5e-4`), max_iter 60.
2. **baseline** – default `[512, 256]`.
3. **deep** – `[512, 256, 128]`, halved LR (`5e-4`), alpha `5e-5`, max_iter `baseline+40`.

Fill in values from `outputs/task2/mlp_hparam_sweep/summary.csv`:

| Config | Hidden Layers | LR | Alpha | Mean CV Acc. | Mean CV Macro F1 | Test Acc. | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact | `(256,)` | 5e-3 | 5e-4 | ✅ | ✅ | ✅ | ✅ |
| baseline | `(512, 256)` | 1e-3 | 1e-4 | ✅ | ✅ | ✅ | ✅ |
| deep | `(512, 256, 128)` | 5e-4 | 5e-5 | ✅ | ✅ | ✅ | ✅ |

**Qualitative analysis (update with actual observations):**

- *Learning rate effect* – The compact model usually converges faster but may underperform due to reduced capacity; mention if accuracy gap is ≥2 %.
- *Depth vs. overfitting* – The deep model can slightly overfit (higher train/CV vs. test gap). Use CV vs. test macro F1 to support statements.
- *Regularisation* – Stronger alpha in compact models can stabilise training on small subsets.

### 3.4 Per‑Class Performance

- Extract per-class F1/precision/recall from `reports/*.json`. Summarise which classes (e.g., `cat`, `dog`, `truck`) are hardest/easiest.
- Note patterns such as similar confusion between `cat`/`dog` or `ship`/`airplane`.

---

## 4. Task 3 – CNN（GPU）

Task 3 采用与 MLP 不同的第二种方法：ResNet18（PyTorch，GPU）。保持与 Task 2 相同的评估流程：5 折交叉验证、测试集评估、超参/增广 sweep。

### 4.1 CNN Experimental Setup

- **Model** – ResNet18 (torchvision) modified for 32×32 inputs (3×3 stem conv, no initial maxpool) with optional dropout before the final FC layer.
- **Feature Inputs** – Operates directly on the raw RGB images stored in `dataset.train_images` / `test_images`, with CIFAR normalisation and configurable data augmentation.
- **Training** – SGD with momentum, cosine annealing schedule, cross-entropy loss, 5-fold stratified CV. Requires an NVIDIA GPU; the script aborts if CUDA is unavailable.
- **Artifacts** – Written to `outputs/task3/cnn_*`，包含 CSV/JSON 汇总与对应的图表。

### 4.2 CNN Augmentation Sweep

The “feature-dimension” analogue for CNNs is the strength of the augmentation pipeline. Two presets are provided:

1. **standard** – Random crop + horizontal flip.
2. **strong_aug** – Adds ColorJitter + RandomErasing on top of the baseline.

Fill in `outputs/task3/cnn_feature_sweep/summary.csv` and reference `cnn_feature_accuracy.png`.

| Variant | Augmentations | Mean CV Accuracy | Mean CV Macro F1 | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- | --- |
| standard | ✅ | ✅ | ✅ | ✅ | ✅ |
| strong_aug | ✅ | ✅ | ✅ | ✅ | ✅ |

Discuss whether the extra augmentation stabilises validation metrics and how it affects convergence versus overfitting.

### 4.7 CNN Hyper‑Parameter Sweep

Three preset configurations tweak epochs, learning rate, weight decay, and dropout (see `outputs/task3/cnn_hparam_sweep/summary.csv` and `cnn_hparam_accuracy.png`):

| Config | Epochs | LR | Weight Decay | Dropout | Mean CV Acc. | Mean CV Macro F1 | Test Acc. | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| baseline | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| regularized | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

When copying the numbers, also mention GPU runtime and any signs of under/over-regularisation.

### 4.4 CNN Insights

- Strong augmentation + heavier regularisation typically yields better macro F1 on smaller training subsets by preventing the network from memorising noise.
- CNNs usually surpass传统扁平特征模型（如 RF/MLP）在绝对准确率上的表现，但计算量更大（≈25 epochs × 5 folds / 每个 sweep）。
- Highlight any per-class gains (e.g., animals vs. vehicles) relative to the MLP baselines to motivate using learned convolutional features.

---

## 5. Task 4 – Comparative Analysis

Use the completed tables above to answer the reflective questions. Suggested structure:

### 5.1 Accuracy & Generalisation

- Compare the best-performing GPU MLP vs. CNN. Discuss:
  - Absolute accuracy difference.
  - Macro F1 – indicates class balance handling.
  - Per-class winners (cite JSON metrics or confusion matrix observations).

### 5.2 Computational Complexity

- **GPU MLP** – time grows with epochs × parameters; PCA 维度越低，前向/反向更快。提及实际训练耗时（可用 `/usr/bin/time -v` 或日志）。
- **CNN** – 卷积在 GPU 上高效，但总体计算量更大（图像域 + 数据增强）。记录一次 sweep 的耗时作为对比。

### 5.3 Overfitting Assessment

- Contrast CV metrics vs. test metrics:
  - If CV accuracy ≈ test accuracy, model generalises well.
  - Look for cases where CV macro F1 is much higher than test macro F1 to identify overfitting. Deep MLP/CNN 均可能过拟合，需结合早停/正则化/增广观察。

### 5.4 Recommendations

Based on observations, answer:

1. **When to prefer MLPs?** e.g., 当需快速实验、特征为 PCA 扁平向量、内存占用较低时。
2. **When to prefer CNNs?** e.g., 当 GPU 资源充足且追求最高精度，尤其在原始图像域。
3. **Effect of PCA / augmentation** – 总结 PCA 对 MLP 的速度/精度折衷，以及增广强度对 CNN 过拟合的影响。

---

## 6. Reproducibility Checklist

1. **Install dependencies:** `python3 -m pip install --user -r requirements.txt`.
2. **Run experiments:** `python main.py` (tweak environment variables or CLI args as needed).
3. **Collect metrics:**
   - Task 1: `outputs/task1/feature_metadata.json`.
   - Task 2: `outputs/task2/*`.
   - Task 3: `outputs/task3/*`.
4. **Update this report:** Replace the ✅ placeholders with actual numbers/tables and embed the PNG figures.
5. **(Optional) Full dataset:** Add `--train-subset 0 --test-subset 0` to the script invocation; expect longer runtimes but higher accuracy.

---

## 7. Future Work

- **Stronger CNNs** – Extend the current ResNet18 baseline with modern tricks (e.g., WideResNet, MixUp, CutMix) or fine-tune pretrained ViT models if compute allows.
- **Data Augmentation for Classical Models** – Explore synthetic feature generation or SMOTE-like approaches to help MLP/Random Forest deal with minority classes.
- **Hyper‑parameter Optimisation** – Automate with Optuna or scikit-opt’s `GridSearchCV`/`RandomizedSearchCV` to explore broader spaces.
- **Confusion Matrix Analysis** – Visualise misclassifications to understand class‑specific weaknesses.

---

## 8. References

1. Alex Krizhevsky. “Learning Multiple Layers of Features from Tiny Images.” Technical report, 2009. [http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)
2. Scikit‑learn documentation: [https://scikit-learn.org/stable/modules/neural_networks_supervised.html](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
3. Scikit‑learn Random Forests: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
4. Pedregosa et al., “Scikit-learn: Machine Learning in Python”, JMLR 12, 2011.

---

*Prepared by Codex. Update the highlighted placeholders with your actual experiment results before submission.*
