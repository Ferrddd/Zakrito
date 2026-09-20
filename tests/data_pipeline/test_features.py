from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.data_pipeline.features import (
    _parse_lag,
    add_engineered,
    add_lag_features,
    add_target,
    feature_columns,
)

SULFUR_TAG = "24-2000:Mg.Sulfur"
SULFUR_ANALYZER_TAGS = ["T6", "W7", "P13"]   # config.yaml -> sulfur_analyzer_tags
VIRTUAL_ANALYZER_TAGS = ["F25"]              # config_density.yaml -> virtual_analyzer_tags

# Минимальный FeatureConfig: только поля, которые читают add_target / feature_columns.
CFG: Any = SimpleNamespace(target_tag=SULFUR_TAG, target_limit=10.0,
                      sulfur_analyzer_columns=SULFUR_ANALYZER_TAGS)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Базовый датафрейм для тестов физики."""
    return pd.DataFrame({
        "T5_hdt": [300.0, 310.0, 0.0],
        "P8_hdt": [320.0, 315.0, 300.0],
        "F15_hdt": [10.0, 15.0, 20.0],
        "T11_hdt": [100.0, 200.0, 0.0],   # 0.0 -> деление на ноль без eps
        "F2_hdt": [50.0, 60.0, 70.0],
        "P24_hdt": [5.0, 10.0, 15.0],
    })


@pytest.fixture
def blocks_df() -> pd.DataFrame:
    """Датафрейм с несколькими block_id для проверки границ."""
    return pd.DataFrame({
        "block_id": [1, 1, 1, 2, 2, 2],
        "sensor_A": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        f"{SULFUR_TAG}__bad": [False, False, True, False, False, False],
    })


class TestEngineeredFeatures:

    def test_add_engineered_all_columns_present(self, sample_df: pd.DataFrame) -> None:
        res = add_engineered(sample_df)

        # wabt = T5 (единственная температура слоя), не среднее нескольких
        assert_series_equal(res["wabt"], sample_df["T5_hdt"].rename("wabt"))
        # h2_to_feed = P24 / (T11 + eps)
        assert res["h2_to_feed"].iloc[0] == pytest.approx(0.05)
        assert res["h2_to_feed"].iloc[1] == pytest.approx(0.05)
        assert np.isfinite(res["h2_to_feed"].iloc[2])            # T11 = 0 -> спасает eps
        # load_rel = T11 / median(T11) = T11 / 100
        np.testing.assert_allclose(res["load_rel"].to_numpy(), [1.0, 2.0, 0.0], atol=1e-6)

    def test_add_engineered_missing_columns(self) -> None:
        df = pd.DataFrame({"T5_hdt": [1.0, 2.0, 3.0]})
        res = add_engineered(df)
        assert "wabt" in res.columns                # есть T5 -> wabt строится
        assert "h2_to_feed" not in res.columns      # нет P24 и T11/F26
        assert "load_rel" not in res.columns        # нет T11/F26
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
        assert "Нет колонок ['missing_sensor'], пропускаю" in caplog.text
        assert_frame_equal(res, blocks_df)


class TestTargetFeatures:

    def test_add_target_block_isolation(self, blocks_df: pd.DataFrame) -> None:
        """ИНВАРИАНТ 1: Таргет сдвигается только внутри block_id (+ __bad маскируется в NaN)."""
        blocks_df[SULFUR_TAG] = [1.0, 2.0, 3.0, 11.0, 12.0, 13.0]
        res = add_target(blocks_df, horizon_points=1, cfg=CFG)

        expected_target = pd.Series([2.0, np.nan, np.nan, 12.0, 13.0, np.nan], name="target_1")
        assert_series_equal(res["target_1"], expected_target)

    def test_add_target_violations(self, blocks_df: pd.DataFrame) -> None:
        blocks_df[SULFUR_TAG] = [5.0, 15.0, 5.0, 8.0, 12.0, 8.0]
        blocks_df[f"{SULFUR_TAG}__bad"] = False

        res = add_target(blocks_df, horizon_points=1, cfg=CFG)
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
        """ИНВАРИАНТ 2: защита от утечки таргета через анализаторы серы.

        Чёрный список — только анализаторы серы (T6/W7/P13) и сам таргет. Виртуальный
        анализатор F25 не измеряет серу и идёт в признаки на общих основаниях.
        """
        analyzer = SULFUR_ANALYZER_TAGS[0]
        virtual = VIRTUAL_ANALYZER_TAGS[0]

        # Конструктор из строк, а не .loc на пустом фрейме: иначе dtype=object, а
        # feature_columns отбрасывает всё, что не int/float.
        df = pd.DataFrame([[1, True, 5.0, True, 300.0, 310.0, 5.0, 5.0, 5.0, 5.0, 1.0]], columns=[
            "block_id", "is_valid", "target_6", "mask_test",
            "T5_hdt", "P8_hdt",
            SULFUR_TAG,
            f"{analyzer}__lag3",
            f"{analyzer}__lag6",
            f"{analyzer}__mean18",
            f"{virtual}__delta2",
        ])

        # blind=True: нет ни анализаторов серы, ни таргета, ни ПАК-колонок (с ':')
        cols_blind = feature_columns(df, horizon_points=6, cfg=CFG, blind=True)
        assert set(cols_blind) == {"T5_hdt", "P8_hdt", f"{virtual}__delta2"}
        assert SULFUR_TAG not in cols_blind
        assert not any(c.startswith(analyzer) for c in cols_blind)

        # blind=False, H=6: лаги анализатора разрешены только >= горизонта
        cols_seeing = feature_columns(df, horizon_points=6, cfg=CFG, blind=False)
        assert "T5_hdt" in cols_seeing
        assert f"{analyzer}__lag3" not in cols_seeing     # ЛИКИДЖ: слишком свежие данные
        assert f"{analyzer}__lag6" in cols_seeing         # можно (лаг == горизонт)
        assert f"{analyzer}__mean18" in cols_seeing       # можно
        assert SULFUR_TAG not in cols_seeing              # сам таргет — никогда без лага
        assert f"{virtual}__delta2" in cols_seeing        # F25 — обычный признак