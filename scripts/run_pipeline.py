"""Точка входа: подготовка данных, признаки, таргет, сплиты, сохранение.

Запуск из корня репозитория:
    python -m scripts.run_pipeline
или:
    python scripts/run_pipeline.py

Результат в data/converted/:
    telemetry_pac_lims.parquet          — полный датафрейм (из DataPreparer)
    train_h{H}_{fold}.parquet           — train по каждому фолду
    test_h{H}_{fold}.parquet            — test по каждому фолду
    folds_h{H}.json                     — границы фолдов (для отчёта/защиты)
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.data_pipeline import features as F
from src.data_pipeline import splitting as S
from src.data_pipeline.loader import DataPreparer
from src.data_pipeline.pipeline_models import data_pipeline_settings as s

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_pipeline")

# Показатели ЛИМС, которые хотим приклеить к телеметрии.
# Ключ — имя колонки в итоговом df, значение — точное имя показателя в ЛИМС.
# Оставьте {} если пока не нужно / не знаете точных названий.
LIMS_INDICATORS: dict[str, str] = {
    # "sulfur_lab": "Массовая доля серы",
}

# Горизонты прогноза (в точках сетки 10 мин): 6 = 1 час
HORIZONS: tuple[int, ...] = (6,)

# Считаем ли модель "слепой" к поточным анализаторам серы (режим "ПАК умер")
BLIND = True


def build_dataset() -> pd.DataFrame:
    """Собирает полный датафрейм: телеметрия + ПАК + ЛИМС + маски + сегменты."""
    dp = DataPreparer(s)
    df = dp.prepare_data(lims_indicators=LIMS_INDICATORS or None, save=True)
    return df


def add_features(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """Инженерные признаки + лаги/окна + таргет + список безопасных колонок-фич."""
    df = F.add_engineered(df)
    lag_source_cols = list(F.CONTROLS) + list(F.CONDITION) + [
        c for c in ("wabt", "gas_to_feed", "h2_makeup_to_feed", "quench_to_feed", "load_rel")
        if c in df.columns
    ]
    df = F.add_lag_features(df, lag_source_cols)
    df = F.add_target(df, horizon_points=horizon)
    cols = F.feature_columns(df, horizon_points=horizon, blind=BLIND)
    return df, cols


def make_splits(df: pd.DataFrame, horizon: int) -> tuple[S.Fold, list[S.Fold], pd.Timedelta]:
    """Holdout + rolling-origin CV с эмбарго, без утечки."""

    dt_index = pd.DatetimeIndex(df.index)
    embargo = S.embargo_delta(max_window_points=72, horizon_points=horizon)
    holdout = S.holdout_split(dt_index, test_size="120D", embargo=embargo)
    cv_folds = S.rolling_origin_folds(
        dt_index, n_folds=4, test_size="60D", embargo=embargo,
        expanding=True, holdout=holdout,
    )
    return holdout, cv_folds, embargo


def save_fold(df: pd.DataFrame, fold: S.Fold, cols: list[str],
             target_col: str, horizon: int) -> dict:
    """Проверяет фолд на утечку, сохраняет train/test в parquet, возвращает метаданные."""
    S.assert_no_leakage(df, fold, cols, target_col, horizon)
    Xtr, ytr, Xte, yte = S.train_matrix(df, fold, cols, target_col)

    train_path = s.converted_data_path / f"train_h{horizon}_{fold.name}.parquet"
    test_path = s.converted_data_path / f"test_h{horizon}_{fold.name}.parquet"

    Xtr.assign(**{target_col: ytr}).to_parquet(train_path, compression="brotli")
    Xte.assign(**{target_col: yte}).to_parquet(test_path, compression="brotli")

    logger.info("%s: train %s (%d строк) | test %s (%d строк)",
                fold.name, train_path.name, len(Xtr), test_path.name, len(Xte))

    return {
        "fold": fold.name,
        "train_start": str(fold.train_start), "train_end": str(fold.train_end),
        "test_start": str(fold.test_start), "test_end": str(fold.test_end),
        "n_train": len(Xtr), "n_test": len(Xte),
        "violation_rate_train": float((ytr > F.SULFUR_LIMIT_PPM).mean()) if len(ytr) else None,
        "violation_rate_test": float((yte > F.SULFUR_LIMIT_PPM).mean()) if len(yte) else None,
        "train_file": train_path.name, "test_file": test_path.name,
    }


def main() -> None:
    df = build_dataset()

    for horizon in HORIZONS:
        logger.info("=== Горизонт %d точек (%.1f ч) ===", horizon, horizon / 6)
        dfh, cols = add_features(df, horizon)
        target_col = f"target_{horizon}"

        holdout, cv_folds, embargo = make_splits(dfh, horizon)
        logger.info("Эмбарго: %s", embargo)

        meta = []
        for fold in cv_folds:
            meta.append(save_fold(dfh, fold, cols, target_col, horizon))
        meta.append(save_fold(dfh, holdout, cols, target_col, horizon))

        meta_path = s.converted_data_path / f"folds_h{horizon}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"horizon": horizon, "blind": BLIND, "feature_cols": cols,
                      "folds": meta}, f, ensure_ascii=False, indent=2)
        logger.info("Метаданные фолдов: %s", meta_path)


if __name__ == "__main__":
    main()