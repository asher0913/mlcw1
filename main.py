"""Cross-platform entry point for running all coursework experiments."""

from __future__ import annotations

import sys
from pathlib import Path

# 确保可以导入 src/mlcw 包
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlcw import run_pipeline  # noqa: E402
from mlcw.utils import get_torch_device  # noqa: E402


def _precheck_device() -> None:
    try:
        device, backend = get_torch_device(require_gpu=True)
        print(f"[main] 使用设备: {device} (backend={backend})")
    except RuntimeError as e:
        print(f"[main] 设备检查失败: {e}")
        sys.exit(1)


def main() -> None:
    _precheck_device()
    # 将命令行参数原样传递给 run_pipeline 主函数
    run_pipeline.main()


if __name__ == "__main__":
    main()
