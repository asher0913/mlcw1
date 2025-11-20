## 实验流程说明（GPU 版）

本文件更细致描述一次完整实验的执行方式与结果产出格式，便于复现与写报告。

### 环境要求
- NVIDIA GPU + 可用 CUDA 驱动（`nvidia-smi` 正常）。
- Python 3.10+，已安装 CUDA 版 PyTorch。
- 依赖安装：`python3 -m pip install --user -r requirements.txt`

### 一键运行
```bash
python main.py
```

可通过环境变量覆盖：
- `TRAIN_SUBSET` / `TEST_SUBSET`：默认 10000 / 2000，设为 `0` 则使用全量。
- `DATA_ROOT` / `OUTPUT_ROOT`：数据缓存与结果输出路径。
- 其余超参可在命令行传入 `python -m mlcw.run_pipeline -h` 中的参数。

### 实验阶段
1. **Task 1（PCA）**  
   - 标准化 + PCA（10/30/50/70/100%），写入 `outputs/task1/feature_metadata.json`。
2. **Task 2（GPU MLP）**  
   - 特征维度 sweep：在各 PCA 特征上 5 折交叉验证 + 测试评估，结果位于 `outputs/task2/mlp_feature_sweep/`。  
   - 超参 sweep：在原始特征上变换网络深度 / lr / wd / dropout / epoch，结果位于 `outputs/task2/mlp_hparam_sweep/`。  
  - 输出 CSV/JSON（含 per-class F1），以及精度曲线。
3. **Task 3（CNN）**  
   - 增广强度 sweep：`standard` vs `strong_aug`，5 折 + 测试集，输出至 `outputs/task3/cnn_feature_sweep/`。  
   - 超参 sweep：fast / baseline / regularized 三套配置，输出至 `outputs/task3/cnn_hparam_sweep/`。  
  - 输出 CSV/JSON（含 per-class F1），以及精度曲线。

### 产出物
- `outputs/task*/.../summary.csv`：每组配置的 CV 均值与测试集准确率/宏 F1。
- `outputs/task*/.../fold_metrics.csv`：每折指标。
- `outputs/task*/.../reports/*.json`：完整分类报告（含 per-class precision/recall/F1）。
- `outputs/task*/plots/*.png`：对应的精度对比图。

### 填写报告建议
1. 将 `summary.csv` 中的主指标填入 `report/REPORT.md` 对应表格。
2. 引用 `plots/*.png` 作为结果对比图。
3. 在 Task 4 比较中，描述 MLP 与 CNN 的准确率、宏 F1、训练耗时（可用 `time` 或日志）及过拟合情况（CV vs Test 差距）。
4. 如使用全量数据或调整超参，请在报告中注明。***
