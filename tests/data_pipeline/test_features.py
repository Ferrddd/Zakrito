from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

# Нормальные импорты прямо из вашего пайплайна
from src.data_pipeline.features import (
    SULFUR_TAG,
    _parse_lag,
    add_engineered,
    add_lag_features,
    add_target,
    feature_columns,
)

# Импортируем реальные списки тегов, чтобы тесты не расходились с реальностью
from src.data_pipeline.loader import SULFUR_ANALYZER_TAGS, VIRTUAL_ANALYZER_TAGS


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Базовый датафрейм для тестов физики."""
    return pd.DataFrame({
        "T5_hdt": [300.0, 310.0, 0.0],  # 0.0 для проверки деления
        "P8_hdt": [320.0, 315.0, 300.0],
        "F15_hdt": [10.0, 15.0, 20.0],
        "T11_hdt": [100.0, 200.0, 0.0], # 0.0 вызовет деление на ноль без eps
        "F2_hdt": [50.0, 60.0, 70.0],
        "P24_hdt": [5.0, 10.0, 15.0],
    })


@pytest.fixture
def blocks_df() -> pd.DataFrame:
    """Датафрейм с несколькими block_id для проверки границ."""
    return pd.DataFrame({
        "block_id": [1, 1, 1, 2, 2, 2],
        "sensor_A": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        f"{SULFUR_TAG}__bad": [False, False, True, False, False, False]
    })


class TestEngineeredFeatures:
    
    def test_add_engineered_all_columns_present(self, sample_df: pd.DataFrame) -> None:
        res = add_engineered(sample_df)
        
        assert_series_equal(res["wabt"], pd.Series([310.0, 312.5, 150.0], name="wabt"))
        assert_series_equal(res["dt_reactors"], pd.Series([20.0, 5.0, 300.0], name="dt_reactors"))
        assert np.isfinite(res["gas_to_feed"].iloc[2])


    def test_add_engineered_missing_columns(self) -> None:
        df = pd.DataFrame({"T5_hdt": [1, 2, 3]})
        res = add_engineered(df)
        assert "wabt" not in res.columns
        assert "dt_reactors" not in res.columns
        assert "T5_hdt" in res.columns


class TestLagFeatures:

    def test_add_lag_features_block_isolation(self, blocks_df: pd.DataFrame) -> None:
        """ИНВАРИАНТ 1: Лаги и окна не должны пересекать границы block_id."""
        res = add_lag_features(blocks_df, columns=["sensor_A"], lags=(1, 2), windows=(2,))
        
        expected_lag1 = pd.Series([np.nan, 10.0, 20.0, np.nan, 100.0, 200.0], name="sensor_A__lag1")
        assert_series_equal(res["sensor_A__lag1"], expected_lag1)
        
        assert pd.isna(res["sensor_A__mean2"].iloc[3])
        assert res["sensor_A__mean2"].iloc[4] == 150.0

    def test_add_lag_features_missing_column(self, blocks_df: pd.DataFrame, caplog: Any) -> None:
        res = add_lag_features(blocks_df, columns=["missing_sensor"], lags=(1,))
        assert "Нет колонки missing_sensor, пропускаю" in caplog.text
        assert_frame_equal(res, blocks_df)


class TestTargetFeatures:

    def test_add_target_block_isolation(self, blocks_df: pd.DataFrame) -> None:
        """ИНВАРИАНТ 1: Таргет сдвигается только внутри block_id."""
        blocks_df[SULFUR_TAG] = [1.0, 2.0, 3.0, 11.0, 12.0, 13.0]
        res = add_target(blocks_df, horizon_points=1, tag=SULFUR_TAG)
        
        expected_target = pd.Series([2.0, np.nan, np.nan, 12.0, 13.0, np.nan], name="target_1")
        assert_series_equal(res["target_1"], expected_target)

    def test_add_target_violations(self, blocks_df: pd.DataFrame) -> None:
        blocks_df[SULFUR_TAG] = [5.0, 15.0, 5.0, 8.0, 12.0, 8.0]
        blocks_df[f"{SULFUR_TAG}__bad"] = False 
        
        res = add_target(blocks_df, horizon_points=1, tag=SULFUR_TAG)
        expected_violations = pd.Series([1.0, 0.0, np.nan, 1.0, 0.0, np.nan], name="target_violation_1")
        assert_series_equal(res["target_violation_1"], expected_violations)


class TestFeatureColumnsLogic:

    @pytest.mark.parametrize("name, expected", [
        ("sensor__lag6", 6),
        ("sensor__mean18", 18),
        ("sensor__delta72", 72),
        ("sensor__lag", None),
        ("sensor_without_lag", None),
    ])
    def test_parse_lag(self, name: str, expected: int | None) -> None:
        assert _parse_lag(name) == expected

    def test_feature_columns_leakage_protection(self) -> None:
        """ИНВАРИАНТ 2: Жёсткая защита от утечки (Data Leakage)."""
        
        # Берём реальные названия тегов из проекта
        analyzer_tag = SULFUR_ANALYZER_TAGS[0] if SULFUR_ANALYZER_TAGS else "24-2000:Mg.Sulfur"
        virtual_tag = VIRTUAL_ANALYZER_TAGS[0] if VIRTUAL_ANALYZER_TAGS else "Virtual:Sulfur"

        df = pd.DataFrame(columns=[
            "block_id", "is_valid", "target_6", "mask_test",
            "T5_hdt", "P8_hdt", 
            SULFUR_TAG,
            f"{analyzer_tag}__lag3",
            f"{analyzer_tag}__lag6",
            f"{analyzer_tag}__mean18",
            f"{virtual_tag}__delta2",
        ])
        
        df.loc[0] = [1, True, 5.0, True, 300.0, 310.0, 5.0, 5.0, 5.0, 5.0, 1.0]
        
        # Режим 1: blind=True (ПАК умер)
        cols_blind = feature_columns(df, horizon_points=6, blind=True)
        assert f"{analyzer_tag}__lag6" not in cols_blind
        assert "T5_hdt" in cols_blind
        assert len(cols_blind) == 2 # Только T5_hdt и P8_hdt

        # Режим 2: blind=False, H=6
        cols_seeing = feature_columns(df, horizon_points=6, blind=False)
        assert "T5_hdt" in cols_seeing
        
        # Проверка защиты от утечки:
        assert f"{analyzer_tag}__lag3" not in cols_seeing  # ЛИКИДЖ! Слишком свежие данные
        assert f"{virtual_tag}__delta2" not in cols_seeing # ЛИКИДЖ!
        
        assert f"{analyzer_tag}__lag6" in cols_seeing      # Можно (H=6)
        assert f"{analyzer_tag}__mean18" in cols_seeing    # Можно