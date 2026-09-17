"""Разбиение временных рядов без утечки из будущего.

Правило из ТЗ (п. 6): случайное перемешивание строк недопустимо. Но простого
`train_test_split(shuffle=False)` тоже мало — между train и test нужен зазор
(embargo), иначе последние строки train содержат оконные признаки, построенные
на тех же точках, из которых берётся таргет первых строк test.

Размер зазора: max_window + horizon. Окно 72 точки (12 ч) плюс горизонт 6 точек
(1 ч) = 78 точек = 13 часов. Берём с запасом — сутки.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def masks(self, index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
        train = (index >= self.train_start) & (index <= self.train_end)
        test = (index >= self.test_start) & (index <= self.test_end)
        return train, test

    def __str__(self) -> str:
        return (f"{self.name}: train {self.train_start:%Y-%m-%d} → {self.train_end:%Y-%m-%d} | "
                f"test {self.test_start:%Y-%m-%d} → {self.test_end:%Y-%m-%d}")


def embargo_delta(max_window_points: int, horizon_points: int,
                  freq: str = "10min", safety: float = 1.5) -> pd.Timedelta:
    step = pd.Timedelta(freq)
    return step * int((max_window_points + horizon_points) * safety)


def holdout_split(index: pd.DatetimeIndex, test_size: str = "120D",
                  embargo: pd.Timedelta | None = None) -> Fold:
    """Финальный holdout: последние test_size по времени.

    По умолчанию последние ~4 месяца — апрель-август 2026. Туда попадают и
    период с повышенной серой (апрель 2026), и заморозка ПАК (июнь-июль 2026),
    то есть оба сложных демо-сценария оцениваются честно, вне обучения.
    """
    if embargo is None:
        embargo = pd.Timedelta("1D")

    end = index.max()
    test_start = end - pd.Timedelta(test_size)
    train_end = test_start - embargo
    fold = Fold("holdout", index.min(), train_end, test_start, end)
    _assert_no_overlap(fold, embargo)
    return fold


def rolling_origin_folds(index: pd.DatetimeIndex, n_folds: int = 4,
                         test_size: str = "60D",
                         embargo: pd.Timedelta | None = None,
                         expanding: bool = True,
                         holdout: Fold | None = None,
                         freq: str = "10min") -> list[Fold]:
    """Walk-forward кросс-валидация с зазором.

    expanding=True — train растёт от начала истории (обычно лучше для медленных
    эффектов вроде деградации катализатора).
    expanding=False — скользящее окно фиксированной длины (проверь оба: если
    метрики сильно расходятся, у процесса есть дрейф режима, и это стоит
    сказать на защите).

    Если передан holdout, валидация строится строго левее него.
    """
    if embargo is None:
        embargo = pd.Timedelta("1D")

    limit = holdout.train_end if holdout else index.max()
    span = pd.Timedelta(test_size)
    folds: list[Fold] = []

    for i in range(n_folds):
        step = pd.Timedelta(freq)
        test_end = limit - span * i - (step if i else pd.Timedelta(0))
        test_start = test_end - span + step
        train_end = test_start - embargo
        if expanding:
            train_start = index.min()
        else:
            train_start = max(index.min(), train_end - span * 6)
        if train_end <= train_start:
            logger.warning("Фолд %d не помещается в историю, пропускаю", i)
            continue
        fold = Fold(f"cv{n_folds - i}", train_start, train_end, test_start, test_end)
        _assert_no_overlap(fold, embargo)
        folds.append(fold)

    return list(reversed(folds))


def _assert_no_overlap(fold: Fold, embargo: pd.Timedelta) -> None:
    assert fold.train_end < fold.test_start, f"{fold.name}: train заходит в test"
    gap = fold.test_start - fold.train_end
    assert gap >= embargo, f"{fold.name}: зазор {gap} меньше требуемого {embargo}"


def assert_no_leakage(df: pd.DataFrame, fold: Fold, feature_cols: list[str],
                      target_col: str, horizon_points: int,
                      freq: str = "10min") -> None:
    """Проверки, которые должны падать громко.

    Их стоит вынести в тесты и показать на защите: «утечки нет» звучит слабее,
    чем зелёный прогон pytest.
    """
    train_mask, test_mask = fold.masks(df.index)

    assert not (train_mask & test_mask).any(), "пересечение train и test"

    train_idx = df.index[train_mask]
    test_idx = df.index[test_mask]
    if len(train_idx) and len(test_idx):
        assert train_idx.max() < test_idx.min(), "train содержит точки позже test"

    # горизонт таргета не должен доставать из train в test
    reach = pd.Timedelta(freq) * horizon_points
    assert train_idx.max() + reach <= test_idx.min(), (
        f"таргет train дотягивается до test: нужен зазор > {reach}")

    # ни один признак не должен коррелировать с таргетом как копия
    sample = df.loc[train_mask, feature_cols + [target_col]].dropna()
    if len(sample) > 1:
        corr = sample[feature_cols].corrwith(sample[target_col]).abs()
        suspicious = corr[corr > 0.98]
        assert suspicious.empty, f"подозрение на утечку: {suspicious.to_dict()}"


def train_matrix(df: pd.DataFrame, fold: Fold, feature_cols: list[str],
                 target_col: str, require_valid: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Возвращает X_train, y_train, X_test, y_test.

    Именно здесь, и только здесь, происходит фильтрация по is_valid: признаки уже
    посчитаны по полной сетке, поэтому выбрасывание строк больше ничего не рвёт.
    """
    train_mask, test_mask = fold.masks(df.index)
    usable = df[target_col].notna()
    if require_valid and "is_valid" in df.columns:
        usable &= df["is_valid"].fillna(False)

    tr = df[train_mask & usable]
    te = df[test_mask & usable]
    logger.info("%s: train %d строк, test %d строк", fold.name, len(tr), len(te))
    return (tr[feature_cols], tr[target_col], te[feature_cols], te[target_col])