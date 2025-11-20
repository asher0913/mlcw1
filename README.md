## COMP3055 Coursework Experiment Suite (GPU)

本仓库提供满足课设要求的可复现实验代码：

1. **Task 1** – 下载/加载 CIFAR-10，构建 PCA 特征集（10%、30%、50%、70%、100% 方差保留），输出到 `outputs/task1/feature_metadata.json`。
2. **Task 2** – 使用 **PyTorch** 实现的 GPU MLP，在 PCA 特征与原始特征上执行 5 折交叉验证的特征维度 sweep 和超参 sweep，产出 CSV/JSON/图表。
3. **Task 3** – 仅保留 **CNN (ResNet18)**，在原始图像上做数据增强 sweep 和超参 sweep，同样输出 CSV/JSON/图表。

> 所有训练（Task 2/3）均依赖 NVIDIA GPU，脚本会用 `nvidia-smi` 检查 CUDA。

### 依赖安装

```bash
python3 -m pip install --user -r requirements.txt
```

（需要预装匹配 CUDA 的 PyTorch 发行版）

### 一键运行

```bash
python main.py
```

可用环境变量或 CLI 覆盖默认值：

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `DATA_ROOT` / `--data-root` | `./data` | CIFAR-10 下载/缓存目录 |
| `OUTPUT_ROOT` / `--output-root` | `./outputs` | 指标、图表输出目录 |
| `TRAIN_SUBSET` / `--train-subset` | `10000` | 训练子集大小，`0` 表示用满 50k |
| `TEST_SUBSET` / `--test-subset` | `2000` | 测试子集大小，`0` 表示用满 10k |
| `--pca-targets` | `10 30 50 70 100` | PCA 方差保留百分比 |
| `--cv-splits` | `5` | 交叉验证折数 |
| `--skip-task2`, `--skip-task3` | `False` | 跳过 MLP 或 CNN 实验 |

其它 MLP/CNN 超参可通过 `python -m mlcw.run_pipeline -h` 查看。

### 输出结构

```
outputs/
  task1/feature_metadata.json
  task2/
    mlp_feature_sweep/        # CSV/JSON
    mlp_hparam_sweep/         # CSV/JSON
    plots/                    # MLP 相关图表
  task3/
    cnn_feature_sweep/        # CSV/JSON
    cnn_hparam_sweep/         # CSV/JSON
    plots/                    # CNN 相关图表
```

每个子目录包含 5 折结果、汇总表、测试集分类报告（含各类 F1/精确率/召回率）以及相应的可视化。运行时需确保 GPU 与 CUDA 驱动就绪。***
