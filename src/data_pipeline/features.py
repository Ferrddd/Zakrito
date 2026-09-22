
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_pipeline.feature_config import FeatureConfig, resolve_column_name

logger = logging.getLogger(__name__)

# Служебные колонки пайплайна — никогда не признаки и не лагируются.
SERVICE_COLUMNS = {"block_id", "is_valid", "on_grid", "anomaly_share", ""}
SERVICE_PREFIXES = ("mask_", "target_")

# Монотонная рампа времени внутри блока: её лаги и std — это просто другое
# представление calendar time, а не физика. Оставляем сырой признак, но
# запрещаем лагировать — см. raw_feature_candidates.
LAG_BANNED_COLS = {"hours_since_block_start", "is_startup"}


def _hdt_col(columns: pd.Index, tag_id: str) -> str | None:
    """Реальное имя колонки тега 24-2000 ('T5' -> 'T5' или 'T5_hdt')."""
    try:
        return resolve_column_name(f"{tag_id}_hdt", columns)
    except KeyError:
        return None


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Физические отношения по фиксированным тегам гидроочистки (24-2000).

    Имена тегов захардкожены осознанно: это конкретные формулы для конкретного
    контура (Р-201/Р-202). Реальные имена колонок определяются по df (у тегов,
    которые есть только в 24-2000, суффикса _hdt нет).
    """
    out = df.copy()
    eps = 1e-6
    cols = out.columns

    t5 = _hdt_col(cols, "T5")
    p24 = _hdt_col(cols, "P24")
    feed_col = _hdt_col(cols, "T11") or _hdt_col(cols, "F26")

    if t5:
        out["wabt"] = out[t5]  # единственная доступная температура слоя пока одна
    if feed_col and p24:
        out["h2_to_feed"] = out[p24] / (out[feed_col] + eps)
    if feed_col:
        out["load_rel"] = out[feed_col] / (out[feed_col].median() + eps)

    missing = [n for n, c in (("T5", t5), ("P24", p24), ("T11/F26", feed_col)) if not c]
    if missing:
        logger.warning("add_engineered: нет колонок для %s — часть инженерных "
                       "признаков не создана", missing)
    logger.info("add_engineered: T5=%s P24=%s feed=%s", t5, p24, feed_col)

    if "hours_since_block_start" in out.columns:
        out["is_startup"] = (out["hours_since_block_start"] < 15).astype(float)

    return out


def raw_feature_candidates(df: pd.DataFrame, cfg: FeatureConfig) -> list[str]:
    """Все числовые колонки телеметрии, которые можно лагировать.

    Исключены только служебные колонки пайплайна и сам таргет с его
    метаданными (__bad/__frozen/__age_min) — таргет и его свежесть это не
    признак, а то, что мы предсказываем и по чему фильтруем валидность.
    Анализаторы серы (T6/W7/P13) сюда ВХОДЯТ — они лагируются на общих
    основаниях, а фильтрует их уже feature_columns() по флагу blind.
    """
    target_related = {cfg.target_tag} | {c for c in df.columns
                                         if c.startswith(f"{cfg.target_tag}__")}
    cols = []
    for c in df.columns:
        if c in SERVICE_COLUMNS or c in target_related:
            continue
        if c.startswith(SERVICE_PREFIXES):
            continue
        if df[c].dtype.kind not in "if":
            continue
        if c in LAG_BANNED_COLS:
            # Разрешаем использовать сырой признак напрямую (он попадёт в
            # feature_columns), но не лагируем: лаг монотонной рампы — это
            # просто другое имя для того же calendar time.
            continue
        cols.append(c)
    return cols


def _lag_one_column(block_id: pd.Series, series: pd.Series, col: str,
                    lags: tuple[int, ...], windows: tuple[int, ...]) -> dict[str, pd.Series]:
    g = series.groupby(block_id, dropna=False)
    out = {}
    for lag in lags:
        out[f"{col}__lag{lag}"] = g.shift(lag)
    for win in windows:
        rolled = g.rolling(win, min_periods=max(2, win // 2))
        out[f"{col}__mean{win}"] = rolled.mean().reset_index(level=0, drop=True)
        out[f"{col}__std{win}"] = rolled.std().reset_index(level=0, drop=True)
        out[f"{col}__delta{win}"] = series - g.shift(win)
    return out


def add_lag_features(df: pd.DataFrame, columns: list[str],
                     lags: tuple[int, ...] = (3, 6, 12, 18),
                     windows: tuple[int, ...] = (6, 18, 72),
                     n_jobs: int = -1) -> pd.DataFrame:
    """Лаги и окна по block_id. Шаг сетки 10 мин: lag=6 -> час назад.

    Окна 6 / 18 / 72 точки = 1 / 3 / 12 часов. Диапазон подбирается под мёртвое
    время: оцени его кросс-корреляцией d(тег) против d(серы) и сдвинь сетку
    лагов так, чтобы пик попал внутрь.

    Считается по колонкам параллельно (joblib, процессы): groupby/rolling в
    pandas однопоточны сами по себе.
    """
    from joblib import Parallel, delayed

    columns = list(dict.fromkeys(columns))  # без дублей, порядок сохраняется
    existing = [c for c in columns if c in df.columns]
    missing = set(columns) - set(existing)
    if missing:
        logger.warning("Нет колонок %s, пропускаю", sorted(missing))

    block_id = df["block_id"]
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_lag_one_column)(block_id, df[col], col, lags, windows)
        for col in existing
    )
    new: dict[str, pd.Series] = {}
    for r in results:
        new.update(r)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_target(df: pd.DataFrame, horizon_points: int, cfg: FeatureConfig) -> pd.DataFrame:
    """Таргет: значение анализатора через horizon_points шагов вперёд.

    Сдвиг делается внутри block_id, поэтому последняя точка перед остановом не
    получит таргет из точки после запуска. Недостоверные измерения ПАК
    (заморозка / выход за диапазон) в таргет не попадают — становятся NaN.

    Односторонний порог (value > limit): подходит для серы. Для двустороннего
    допуска (напр. плотность, limit_min/limit_max) эта функция пока не годится.
    """
    out = df.copy()
    tag = cfg.target_tag
    bad_col = f"{tag}__bad"
    clean = out[tag].where(~out[bad_col]) if bad_col in out.columns else out[tag]
    out["_clean_target_src"] = clean

    g = out.groupby("block_id", dropna=False)["_clean_target_src"]
    out[f"target_{horizon_points}"] = g.shift(-horizon_points)
    values = out[f"target_{horizon_points}"]
    out[f"target_violation_{horizon_points}"] = np.where(
        values.isna(), np.nan, (values > cfg.target_limit).astype(float))

    out = out.drop(columns=["_clean_target_src"])
    logger.info("Таргет H=%d: %d размеченных строк, доля нарушений %.3f",
                horizon_points, int(out[f"target_{horizon_points}"].notna().sum()),
                float(out[f"target_violation_{horizon_points}"].mean(skipna=True)))
    return out


def _base_name(col: str) -> str:
    """'T5__lag6' -> 'T5'; '24-2000:Mg.Sulfur__bad' -> '24-2000:Mg.Sulfur'."""
    return col.split("__")[0]


def _is_pak_or_lims(col: str) -> bool:
    return ":" in _base_name(col) or col.startswith("lims_")


def feature_columns(df: pd.DataFrame, horizon_points: int, cfg: FeatureConfig,
                    blind: bool = True) -> list[str]:
    """Список признаков с защитой от утечки.

    blind=True — модель не видит анализаторы серы, сам таргет и вообще
    ничего из выгрузки ПАК/ЛИМС (режим «ПАК и КИП-анализаторы серы
    недоступны»). blind=False — лаги анализаторов и таргета разрешены, но
    только >= горизонта прогноза.
    """
    banned_bases = set(cfg.sulfur_analyzer_columns) | {cfg.target_tag}

    cols = []
    for c in df.columns:
        if c in SERVICE_COLUMNS or c.startswith(SERVICE_PREFIXES):
            continue
        if df[c].dtype.kind not in "if":
            continue
        if blind and _is_pak_or_lims(c):
            continue

        if _base_name(c) in banned_bases:
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