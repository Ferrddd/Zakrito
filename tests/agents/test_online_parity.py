"""Паритет train/serve: признаки, собранные онлайн из буфера, должны совпасть с обучающими.

Самый важный тест инференса — без него агент может молча выдавать прогнозы на «другой» модели.
Требует data/converted/telemetry_pac_lims.parquet и обученных моделей (пропускается, если их нет).
Запуск: make test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.agents.quality.features_online import OnlineFeatureBuilder, to_regular_grid
from src.agents.quality.train import build_dataset
from src.data_pipeline.feature_config import load_feature_config, resolve_cfg_columns
from src.data_pipeline.pipeline_models import data_pipeline_settings

PARQUET = data_pipeline_settings.converted_data_path / "telemetry_pac_lims.parquet"
CONFIG = "src/agents/quality/config.yaml"


@pytest.mark.skipif(not PARQUET.exists(), reason="нет витрины telemetry_pac_lims.parquet")
@pytest.mark.parametrize("blind", [True, False])
def test_online_features_match_training(blind: bool) -> None:
    cfg = resolve_cfg_columns(load_feature_config(CONFIG), pq.read_schema(PARQUET).names)
    h = cfg.horizons_points[1] if len(cfg.horizons_points) > 1 else cfg.horizons_points[0]
    mode = "blind" if blind else "with_analyzer"
    spec = cfg.models_dir / f"h{h}_{mode}" / "feature_spec.json"
    if not spec.exists():
        pytest.skip(f"нет {spec} — сначала make train_quality")

    train_df, _cols, _raw_cols = build_dataset(cfg, h, blind)          # то, что видела модель
    raw = pd.read_parquet(PARQUET)                                    # то, что придёт онлайн
    builder = OnlineFeatureBuilder.from_spec_file(spec)

    rng = np.random.default_rng(0)
    for pos in rng.choice(np.arange(400, len(train_df) - 1), size=15, replace=False):
        ts = train_df.index[pos]
        window = to_regular_grid(raw.loc[:ts].tail(288), cfg.grid_freq)
        online = builder.build(window).iloc[0]
        offline = train_df.loc[ts, builder.cols].astype(float)
        both = online.notna() & offline.notna()
        np.testing.assert_allclose(online[both].to_numpy(), offline[both].to_numpy(), rtol=1e-5, atol=1e-4,   # atol: rolling.std на константном окне даёт шум ~1e-5
                                   
                                   err_msg=f"расхождение признаков в {ts} ({mode})")
        assert (online.isna() == offline.isna()).mean() > 0.99, f"разные NaN-паттерны в {ts}"