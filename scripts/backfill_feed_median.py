"""Дописывает feed_col / feed_median в существующие feature_spec.json без переобучения.

Нужно моделям, обученным ДО патча train.py: add_engineered считает load_rel как
feed / median(feed по ВСЕЙ витрине), а онлайн-буфер короткий. Признаки для модели
остаются теми же, что при обучении — просто фиксируем ту же медиану.

    python -m scripts.backfill_feed_median [--config src/agents/quality/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
import pyarrow.parquet as pq

from src.data_pipeline.feature_config import load_feature_config
from src.data_pipeline.features import _hdt_col
from src.data_pipeline.pipeline_models import data_pipeline_settings
from src.utils.config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/agents/quality/config.yaml")
    args = ap.parse_args()

    cfg = load_feature_config(args.config)
    parquet = data_pipeline_settings.converted_data_path / "telemetry_pac_lims.parquet"
    names = pd.Index(pq.read_schema(parquet).names)
    feed_col = _hdt_col(names, "T11") or _hdt_col(names, "F26")
    if not feed_col:
        raise SystemExit("Не нашёл T11/F26 в витрине — load_rel не строится, backfill не нужен")

    median = float(pd.read_parquet(parquet, columns=[feed_col])[feed_col].median())
    specs = sorted(cfg.models_dir.glob("h*_*/feature_spec.json"))
    if not specs:
        raise SystemExit(f"Нет feature_spec.json в {cfg.models_dir}")
    for path in specs:
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["feed_col"], spec["feed_median"] = feed_col, median
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("%s: feed_col=%s feed_median=%.6f", path.parent.name, feed_col, median)


if __name__ == "__main__":
    main()