from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.splitting import (
    Fold,
    assert_no_leakage,
    embargo_delta,
    holdout_split,
    rolling_origin_folds,
    train_matrix,
)


@pytest.fixture
def time_index() -> pd.DatetimeIndex:
    """Индекс на 100 дней с шагом 1 день для удобных тестов."""
    return pd.date_range("2026-01-01", periods=100, freq="D")


@pytest.fixture
def df_large() -> pd.DataFrame:
    """Большой датафрейм (частота 10 минут) для тестирования утечек."""
    idx = pd.date_range("2026-01-01", periods=2000, freq="10min")
    df = pd.DataFrame(index=idx)
    
    df["feature_1"] = np.random.randn(2000)
    df["target"] = df["feature_1"] * 0.5 + np.random.randn(2000) * 0.1
    df["leakage_feature"] = df["target"]
    df["is_valid"] = True
    
    return df


class TestEmbargoLogic:
    def test_embargo_delta_calculation(self) -> None:
        delta = embargo_delta(max_window_points=72, horizon_points=6, freq="10min", safety=1.5)
        assert delta == pd.Timedelta(minutes=1170)

    def test_embargo_delta_different_freq(self) -> None:
        delta = embargo_delta(max_window_points=10, horizon_points=2, freq="1H", safety=2.0)
        assert delta == pd.Timedelta(days=1)


class TestFoldGeneration:
    def test_fold_masks(self, time_index: pd.DatetimeIndex) -> None:
        fold = Fold(
            name="test_fold",
            train_start=time_index[0],
            train_end=time_index[2],
            test_start=time_index[5],
            test_end=time_index[7]
        )
        train_mask, test_mask = fold.masks(time_index)
        
        assert train_mask.sum() == 3
        assert test_mask.sum() == 3
        assert train_mask[0] and train_mask[2] and not train_mask[3]
        assert test_mask[5] and test_mask[7] and not test_mask[4]

    def test_holdout_split(self, time_index: pd.DatetimeIndex) -> None:
        fold = holdout_split(time_index, test_size="20D", embargo=pd.Timedelta("5D"))
        
        assert fold.test_end == time_index[-1]
        assert fold.test_start == time_index[-1] - pd.Timedelta("20D")
        assert fold.test_start - fold.train_end == pd.Timedelta("5D")
        assert fold.train_start == time_index[0]

    def test_rolling_origin_folds_expanding(self, time_index: pd.DatetimeIndex) -> None:
        folds = rolling_origin_folds(
            time_index, n_folds=3, test_size="20D", embargo=pd.Timedelta("5D"), 
            expanding=True, freq="1D"
        )
        
        assert len(folds) == 3
        last_fold = folds[-1]
        assert last_fold.test_end == time_index[-1]
        assert last_fold.train_start == time_index[0]
        assert last_fold.test_start - last_fold.train_end == pd.Timedelta("5D")

    def test_rolling_origin_folds_sliding_window(self, time_index: pd.DatetimeIndex) -> None:
        folds = rolling_origin_folds(
            time_index, n_folds=3, test_size="5D", embargo=pd.Timedelta("2D"), 
            expanding=False, freq="1D"
        )
        
        last_fold = folds[-1]
        span = pd.Timedelta("5D")
        expected_start = max(time_index[0], last_fold.train_end - span * 6)
        assert last_fold.train_start == expected_start
        assert last_fold.train_start > time_index[0]

    def test_rolling_origin_folds_with_holdout(self, time_index: pd.DatetimeIndex) -> None:
        holdout = holdout_split(time_index, test_size="20D", embargo=pd.Timedelta("5D"))
        folds = rolling_origin_folds(
            time_index, n_folds=2, test_size="10D", embargo=pd.Timedelta("1D"), 
            holdout=holdout, freq="1D"
        )
        
        last_fold = folds[-1]
        assert last_fold.test_end == holdout.train_end

    def test_rolling_origin_folds_history_too_short(self, time_index: pd.DatetimeIndex) -> None:
        folds = rolling_origin_folds(
            time_index, n_folds=10, test_size="20D", embargo=pd.Timedelta("2D"), freq="1D"
        )
        assert len(folds) < 10
        assert len(folds) > 0


class TestLeakageProtections:
    def test_assert_no_leakage_passes_clean(self, df_large: pd.DataFrame) -> None:
        fold = Fold(
            name="cv",
            train_start=df_large.index[0],
            train_end=df_large.index[500],
            test_start=df_large.index[600],
            test_end=df_large.index[1900]
        )
        assert_no_leakage(
            df_large, fold, feature_cols=["feature_1"], target_col="target", horizon_points=6
        )

    def test_assert_no_leakage_overlap_fails(self, df_large: pd.DataFrame) -> None:
        fold = Fold(
            name="cv",
            train_start=df_large.index[0],
            train_end=df_large.index[600],
            test_start=df_large.index[500],
            test_end=df_large.index[1900]
        )
        with pytest.raises(AssertionError, match="пересечение train и test"):
            assert_no_leakage(df_large, fold, ["feature_1"], "target", horizon_points=6)

    def test_assert_no_leakage_horizon_reach_fails(self, df_large: pd.DataFrame) -> None:
        fold = Fold(
            name="cv",
            train_start=df_large.index[0],
            train_end=df_large.index[500],
            test_start=df_large.index[505],
            test_end=df_large.index[1900]
        )
        with pytest.raises(AssertionError, match="таргет train дотягивается до test"):
            assert_no_leakage(df_large, fold, ["feature_1"], "target", horizon_points=6)

    def test_assert_no_leakage_high_correlation_fails(self, df_large: pd.DataFrame) -> None:
        fold = Fold(
            name="cv",
            train_start=df_large.index[0],
            train_end=df_large.index[500],
            test_start=df_large.index[600],
            test_end=df_large.index[1900]
        )
        with pytest.raises(AssertionError, match="подозрение на утечку"):
            assert_no_leakage(
                df_large, fold, ["feature_1", "leakage_feature"], "target", horizon_points=6
            )


class TestTrainMatrix:
    def test_train_matrix_filtering(self, df_large: pd.DataFrame) -> None:
        fold = Fold(
            name="cv",
            train_start=df_large.index[0],
            train_end=df_large.index[500],
            test_start=df_large.index[600],
            test_end=df_large.index[1000]
        )
        
        df_large.loc[df_large.index[10:20], "is_valid"] = False
        df_large.loc[df_large.index[650:660], "target"] = np.nan
        
        X_tr, y_tr, X_te, y_te = train_matrix(
            df_large, fold, ["feature_1"], "target", require_valid=True
        )
        
        assert len(X_tr) == 491
        assert len(X_tr) == len(y_tr)
        assert len(X_te) == 391
        assert len(X_te) == len(y_te)
        assert list(X_tr.columns) == ["feature_1"]
        assert y_tr.name == "target"