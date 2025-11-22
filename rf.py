"""Entry point for the Random Forest coursework pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rf import run_pipeline  # noqa: E402


def main() -> None:
    run_pipeline.main()


if __name__ == "__main__":
    main()
