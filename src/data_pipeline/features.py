"""Построение признаков и таргета для агента качества.

Два инварианта, которые здесь соблюдаются жёстко:

1. Все лаги и окна считаются ВНУТРИ block_id. Окно не может пересечь останов.
2. В признаки для прогноза не попадают поточные анализаторы серы с лагом
   меньше горизонта прогноза — иначе модель просто читает таргет.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_pipeline.loader import SULFUR_ANALYZER_TAGS, VIRTUAL_ANALYZER_TAGS

logger = logging.getLogger(__name__)

SULFUR_TAG = "24-2000:Mg.Sulfur"
D15_TAG = "24-2000:D15"
SULFUR_LIMIT_PPM = 10.0

# Управляющие кандидаты по гидроочистке (справочник КИП, лист 24-2000).
# Буквы в именах не соответствуют физическому смыслу — проверено по справочнику.
CONTROLS = {
    "T5_hdt": "Р-201, температура ГСС на выходе",
    "P8_hdt": "Р-202, температура ГСС на входе",
    "F15_hdt": "Расход квенча в Р-202",
    "T11_hdt": "Расход сырья на установку (массовый)",
    "F2_hdt": "Циркуляционный ВСГ от ЦК-201",
    "P24_hdt": "Расход свежего ВСГ с КЦА",
}

# Признаки состояния оборудования (нужны агенту надёжности, полезны и качеству)
CONDITION = {
    "W10_hdt": "Перепад давления Р-202 — прокси закоксовывания",
    "F19_hdt": "Давление на входе Р-202",
}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Физические отношения. Считаются построчно, разрывов не пересекают."""
    out = df.copy()
    eps = 1e-6

    if {"T5_hdt", "P8_hdt"} <= set(out.columns):
        # WABT-прокси: среднее по доступным температурам слоёв
        out["wabt"] = out[["T5_hdt", "P8_hdt"]].mean(axis=1)
        out["dt_reactors"] = out["P8_hdt"] - out["T5_hdt"]

    if {"F2_hdt", "T11_hdt"} <= set(out.columns):
        out["gas_to_feed"] = out["F2_hdt"] / (out["T11_hdt"] + eps)

    if {"P24_hdt", "T11_hdt"} <= set(out.columns):
        out["h2_makeup_to_feed"] = out["P24_hdt"] / (out["T11_hdt"] + eps)

    if {"F15_hdt", "T11_hdt"} <= set(out.columns):
        out["quench_to_feed"] = out["F15_hdt"] / (out["T11_hdt"] + eps)

    if "T11_hdt" in out.columns:
        # LHSV-прокси: загрузка относительно медианной
        out["load_rel"] = out["T11_hdt"] / (out["T11_hdt"].median() + eps)

    return out


def add_lag_features(df: pd.DataFrame, columns: list[str],
                     lags: tuple[int, ...] = (3, 6, 12, 18),
                     windows: tuple[int, ...] = (6, 18, 72)) -> pd.DataFrame:
    """Лаги и окна по block_id. Шаг сетки 10 мин: lag=6 -> час назад.

    Окна 6 / 18 / 72 точки = 1 / 3 / 12 часов. Диапазон подбирается под мёртвое
    время: оцени его кросс-корреляцией d(T5) против d(серы) и сдвинь сетку лагов
    так, чтобы пик попал внутрь.
    """
    out = df.copy()
    g = out.groupby("block_id", dropna=False)
    new = {}

    for col in columns:
        if col not in out.columns:
            logger.warning("Нет колонки %s, пропускаю", col)
            continue
        s = out[col]
        for lag in lags:
            new[f"{col}__lag{lag}"] = g[col].shift(lag)
        for win in windows:
            rolled = g[col].rolling(win, min_periods=max(2, win // 2))
            new[f"{col}__mean{win}"] = rolled.mean().reset_index(level=0, drop=True)
            new[f"{col}__std{win}"] = rolled.std().reset_index(level=0, drop=True)
            # скорость изменения: текущее минус значение win точек назад
            new[f"{col}__delta{win}"] = s - g[col].shift(win)

    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def add_target(df: pd.DataFrame, horizon_points: int,
               tag: str = SULFUR_TAG) -> pd.DataFrame:
    """Таргет: значение анализатора через horizon_points шагов вперёд.

    Сдвиг делается внутри block_id, поэтому последняя точка перед остановом не
    получит таргет из точки после запуска. Недостоверные измерения ПАК
    (заморозка / выход за диапазон) в таргет не попадают — становятся NaN.
    """
    out = df.copy()
    bad_col = f"{tag}__bad"
    clean = out[tag].where(~out[bad_col]) if bad_col in out.columns else out[tag]
    out["_clean_target_src"] = clean

    g = out.groupby("block_id", dropna=False)["_clean_target_src"]
    out[f"target_{horizon_points}"] = g.shift(-horizon_points)
    values = out[f"target_{horizon_points}"]
    out[f"target_violation_{horizon_points}"] = np.where(
        values.isna(), np.nan, (values > SULFUR_LIMIT_PPM).astype(float))

    out = out.drop(columns=["_clean_target_src"])
    logger.info("Таргет H=%d: %d размеченных строк, доля нарушений %.3f",
                horizon_points, int(out[f"target_{horizon_points}"].notna().sum()),
                float(out[f"target_violation_{horizon_points}"].mean(skipna=True)))
    return out


def feature_columns(df: pd.DataFrame, horizon_points: int,
                    blind: bool = True) -> list[str]:
    """Список признаков с защитой от утечки.

    blind=True — модель вообще не видит анализаторы серы (режим «ПАК умер»).
    blind=False — разрешены лаги анализаторов, но только >= горизонта прогноза:
    при H=6 лаг 3 означал бы знание значения на 30 минут вперёд относительно
    момента принятия решения.
    """
    banned_prefixes = tuple(SULFUR_ANALYZER_TAGS) + (SULFUR_TAG,)
    service = {"block_id", "is_valid", "on_grid", "anomaly_share"}

    cols = []
    for c in df.columns:
        if c in service or c.startswith(("target_", "mask_")):
            continue
        if df[c].dtype.kind not in "if":
            continue

        is_analyzer = c.startswith(banned_prefixes + tuple(VIRTUAL_ANALYZER_TAGS))
        if is_analyzer:
            if blind:
                continue
            lag = _parse_lag(c)
            if lag is None or lag < horizon_points:
                continue
        cols.append(c)

    logger.info("Признаков: %d (blind=%s)", len(cols), blind)
    return cols


def _parse_lag(name: str) -> int | None:
    for marker in ("__lag", "__mean", "__delta"):
        if marker in name:
            try:
                return int(name.split(marker)[1])
            except ValueError:
                return None
    return None