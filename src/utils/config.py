

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logging.getLogger(__name__).info("Конфиг загружен: %s", path)
    return cfg


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )