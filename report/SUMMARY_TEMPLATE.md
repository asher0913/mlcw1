# 运行结果摘要（填充后使用）

> 请在 GPU 上运行 `python main.py` 后，将关键指标填入下表；若使用不同子集或额外超参，请注明。

## Task 1 (PCA)
摘自 `outputs/task1/feature_metadata.json`：

| 特征集 | 维度 | 方差保留 |
| --- | --- | --- |
| original | 3072 | 100% |
| pca_70 |  |  |
| pca_50 |  |  |
| pca_30 |  |  |
| pca_10 |  |  |

## Task 2 (GPU MLP) – 特征维度 sweep
来自 `outputs/task2/mlp_feature_sweep/summary.csv`：

| 特征集 | CV Acc | CV Macro F1 | Test Acc | Test Macro F1 |
| --- | --- | --- | --- | --- |
| original |  |  |  |  |
| pca_70 |  |  |  |  |
| pca_50 |  |  |  |  |
| pca_30 |  |  |  |  |
| pca_10 |  |  |  |  |

## Task 2 (GPU MLP) – 超参 sweep（原始特征）
来自 `outputs/task2/mlp_hparam_sweep/summary.csv`：

| 配置 | 结构 | LR | WD | Dropout | Epochs | CV Acc | CV Macro F1 | Test Acc | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast |  |  |  |  |  |  |  |  |  |
| baseline |  |  |  |  |  |  |  |  |  |
| regularized |  |  |  |  |  |  |  |  |  |

## Task 3 (CNN) – 增广 sweep
来自 `outputs/task3/cnn_feature_sweep/summary.csv`：

| 方案 | CV Acc | CV Macro F1 | Test Acc | Test Macro F1 |
| --- | --- | --- | --- | --- |
| standard |  |  |  |  |
| strong_aug |  |  |  |  |

## Task 3 (CNN) – 超参 sweep
来自 `outputs/task3/cnn_hparam_sweep/summary.csv`：

| 配置 | Epochs | LR | WD | Dropout | CV Acc | CV Macro F1 | Test Acc | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast |  |  |  |  |  |  |  |  |
| baseline |  |  |  |  |  |  |  |  |
| regularized |  |  |  |  |  |  |  |  |

## 观察与结论（简述）
- MLP vs PCA 维度：_______
- MLP 超参主要影响：_______
- CNN 增广/超参效果：_______
- 最佳模型与理由：_______
