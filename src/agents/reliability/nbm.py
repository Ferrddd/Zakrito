from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 0.1   # нижняя граница ΔP при логарифмировании (в данных есть значения около нуля)


def resolve_column(tag_ids: str | list[str], columns: Any, suffixes: list[str]) -> str | None:
    """Тег(и) -> реальное имя колонки витрины (пробуем tag+suffix по порядку)."""
    ids = [tag_ids] if isinstance(tag_ids, str) else list(tag_ids)
    for t in ids:
        for suf in suffixes:
            if f"{t}{suf}" in columns:
                return f"{t}{suf}"
    return None


def mask_sentinels(df: pd.DataFrame, values: list[float] | tuple[float, ...], cols: list[str] | None = None) -> pd.DataFrame:
    if not values:
        return df
    out = df.copy()
    for c in (cols if cols is not None else list(out.select_dtypes("number").columns)):
        if c not in out.columns:
            continue
        s = out[c]
        for v in values:
            s = s.mask((s - v).abs() < 1e-6)
        out[c] = s
    return out


def smooth_target(df: pd.DataFrame, dp_col: str, roll: int) -> pd.Series:
    """ΔP, сглаженный медианой назад за roll точек (гасит выбросы КИП)."""
    return df[dp_col].astype(float).rolling(roll, min_periods=max(1, roll // 2)).median()


def build_features(df: pd.DataFrame, base_cols: list[str], dp_col: str, roll: int, lag: int) -> pd.DataFrame:
    dp_lag = smooth_target(df, dp_col, roll).shift(lag)
    out: dict[str, pd.Series] = {"log_dp_lag": np.log(dp_lag.clip(lower=EPS))}
    for c in base_cols:
        m = df[c].astype(float).rolling(roll, min_periods=max(1, roll // 2)).mean()
        out[c] = m
        out[f"{c}__d{lag}"] = m - m.shift(lag)
    return pd.DataFrame(out, index=df.index)


class StackedRegressor:
    def __init__(self, linear: Any, gbt: Any):
        self.linear = linear
        self.gbt = gbt

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.linear.predict(X), dtype=float) + np.asarray(self.gbt.predict(X), dtype=float)


def make_linear(alpha: float) -> Any:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def make_regressor(params: dict[str, Any], n_estimators: int | None = None) -> Any:
    p = dict(params)
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="regression_l1", **p)   # L1: устойчив к выбросам КИП
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        logger.warning("lightgbm не найден — используется HistGradientBoostingRegressor")
        return HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=p.get("n_estimators", 300),
            learning_rate=p.get("learning_rate", 0.05), max_leaf_nodes=p.get("num_leaves", 31),
            min_samples_leaf=p.get("min_child_samples", 200), l2_regularization=p.get("reg_lambda", 3.0),
            early_stopping=False, random_state=0)


def fit_with_early_stopping(model: Any, Xtr: pd.DataFrame, ytr: np.ndarray,
                            Xval: pd.DataFrame, yval: np.ndarray, rounds: int) -> int:
    try:
        import warnings

        import lightgbm
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        if isinstance(model, lightgbm.LGBMRegressor):
            model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                      callbacks=[lightgbm.early_stopping(rounds, verbose=False)])
            return int(model.best_iteration_ or model.n_estimators)
    except ImportError:
        pass
    model.fit(Xtr, ytr)
    return int(getattr(model, "max_iter", 300))


@dataclass
class NormalBehaviorModel:
    model: StackedRegressor
    base_cols: list[str]
    target_col: str
    feed_col: str | None
    roll: int
    lag: int
    resid_scale: float             # робастный σ ОТНОСИТЕЛЬНОГО остатка (факт/модель - 1) на out-of-sample
    columns: list[str] = field(default_factory=list)   # порядок признаков при обучении
    metrics: dict[str, Any] = field(default_factory=dict)

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = build_features(df, self.base_cols, self.target_col, self.roll, self.lag)
        return X[self.columns] if self.columns else X

    def predict_actual(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(модель, факт) для каждой строки df; NaN там, где не хватает истории/данных."""
        X = self.frame(df)
        ok = X.notna().all(axis=1).to_numpy()
        pred = np.full(len(df), np.nan)
        if ok.any():
            Xo = X[ok]
            pred[ok] = np.exp(Xo["log_dp_lag"].to_numpy()) * np.exp(self.model.predict(Xo))
        actual = smooth_target(df, self.target_col, self.roll).to_numpy(dtype=float)
        return pred, actual

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "base_cols": self.base_cols, "target_col": self.target_col,
                     "feed_col": self.feed_col, "roll": self.roll, "lag": self.lag,
                     "resid_scale": self.resid_scale, "columns": self.columns, "metrics": self.metrics}, p)

    @classmethod
    def load(cls, path: str | Path) -> NormalBehaviorModel | None:
        p = Path(path)
        if not p.exists():
            return None
        return cls(**joblib.load(p))
