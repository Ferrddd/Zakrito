"""Обучение агента качества.

Запуск:
    python -m src.agents.quality.train --horizon 6 --skip-cv --force-rebuild

Всё, что можно вынести из кода, вынесено в config.yaml (роли тегов, горизонты,
лаги/окна, размеры сплитов, гиперпараметры LightGBM, пути артефактов) и в .env
через DataPipelineSettings. Повторить обучение — значит передать тот же конфиг.

Кэш датасета: build_dataset кэширует результат фиче-инжиниринга в
data/converted/dataset_cache/h{H}_{mode}.parquet. Кэш не отслеживает изменения
конфига и кода признаков — после любой правки передай --force-rebuild.

Схема оценки честная в двух местах:
- early stopping и порог тревоги подбираются на валидации (последние VAL_DAYS
  train с эмбарго), а НЕ на holdout и не на train;
- рядом с метриками модели считаются базлайны (медиана train и persistence
  «сера сейчас»): без них MAE/PR-AUC ничего не значат.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from src.agents.quality.model import (
    LGBMTrainConfig,
    QuantileQualityModel,
    ViolationClassifier,
    build_monotone_constraints,
)
from src.data_pipeline.feature_config import (
    FeatureConfig,
    load_feature_config,
    resolve_cfg_columns,
)
from src.data_pipeline.features import (
    _hdt_col,
    add_engineered,
    add_lag_features,
    add_target,
    feature_columns,
    raw_feature_candidates,
)
from src.data_pipeline.pipeline_models import data_pipeline_settings
from src.data_pipeline.splitting import (
    assert_no_leakage,
    embargo_delta,
    holdout_split,
    rolling_origin_folds,
    train_matrix,
)
from src.utils.config import setup_logging
from src.utils.metrics import (
    best_threshold,
    evaluate_quantiles,
    evaluate_violation,
    lead_time_minutes,
)

logger = logging.getLogger(__name__)

# Хвост train, который отдаётся под early stopping и подбор порога тревоги.
VAL_DAYS = "30D"


# --------------------------------------------------------------------------
# Имена признаков
# --------------------------------------------------------------------------
def clean_feature_names(cols: list[str]) -> list[str]:
    """Чистит имена по белому списку (буквы/цифры/_/./-): LightGBM не переваривает
    пробелы и прочее в feature_names.

    Подчёркивания НЕ схлопываются: '__' — разделитель базового имени и суффикса
    (T5__lag6), по нему build_monotone_constraints и feature_columns находят
    базовый тег. Схлопывание '__' -> '_' раньше молча обнуляло все монотонные
    ограничения на лаговых признаках.
    """
    cleaned = [re.sub(r'[^0-9A-Za-zА-Яа-яЁё_.\-]+', '_', col) for col in cols]
    return [c.strip('_') for c in cleaned]


def _safe_rename_map(raw_cols: list[str]) -> dict[str, str]:
    """Коллизии после очистки разруливаются суффиксом __dupN; пустые имена
    получают unnamed_<i> (с предупреждением — скорее всего это мусорная колонка)."""
    cleaned = clean_feature_names(raw_cols)

    empties = [raw for raw, c in zip(raw_cols, cleaned) if not c]
    if empties:
        logger.warning("Колонки с пустым именем после очистки (%d): %r — "
                       "проверь, нужны ли они", len(empties), empties[:5])
        cleaned = [c if c else f"unnamed_{i}" for i, c in enumerate(cleaned)]

    counts = Counter(cleaned)
    dup_bases = {name for name, cnt in counts.items() if cnt > 1}
    if not dup_bases:
        return dict(zip(raw_cols, cleaned))

    logger.warning("Коллизии после очистки имён (%d групп): %s",
                   len(dup_bases), sorted(dup_bases))
    seen: dict[str, int] = {}
    rename_map = {}
    for orig, base in zip(raw_cols, cleaned):
        if base in dup_bases:
            seen[base] = seen.get(base, 0) + 1
            rename_map[orig] = f"{base}__dup{seen[base]}"
        else:
            rename_map[orig] = base
    return rename_map


# --------------------------------------------------------------------------
# Датасет
# --------------------------------------------------------------------------
def build_dataset(cfg: FeatureConfig, horizon_points: int, blind: bool,
                  use_cache: bool = True, force_rebuild: bool = False
                  ) -> tuple[pd.DataFrame, list[str], list[str] | None]:
    """Возвращает (df, cols, raw_cols). cols — имена после очистки,
    raw_cols — исходные имена в том же порядке (нужны для инференса)."""
    settings = data_pipeline_settings
    mode = "blind" if blind else "with_analyzer"

    cache_dir = settings.converted_data_path / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"h{horizon_points}_{mode}.parquet"
    cols_path = cache_dir / f"h{horizon_points}_{mode}.cols.json"

    if use_cache and not force_rebuild and cache_path.exists() and cols_path.exists():
        logger.info("Датасет из кэша: %s", cache_path)
        df = pd.read_parquet(cache_path)
        meta = json.loads(cols_path.read_text(encoding="utf-8"))
        if isinstance(meta, list):          # старый формат кэша
            return df, meta, None
        return df, meta["cols"], meta["raw_cols"]

    parquet_path = settings.converted_data_path / "telemetry_pac_lims.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} не найден — сначала прогони "
            "DataPreparer.prepare_data() (src/data_pipeline/loader.py)")

    df = pd.read_parquet(parquet_path)
    df = add_engineered(df)
    lag_source_cols = raw_feature_candidates(df, cfg)
    if not blind:
        # без этого with_analyzer ничем не отличался бы от blind: разрешённых
        # признаков анализатора (лаг >= горизонта) просто не существовало бы
        lag_source_cols = lag_source_cols + list(cfg.sulfur_analyzer_columns) + [cfg.target_tag]
    df = add_lag_features(df, lag_source_cols,
                          lags=cfg.lags_points, windows=cfg.window_points)
    df = add_target(df, horizon_points, cfg)

    raw_cols = feature_columns(df, horizon_points, cfg, blind=blind)

    rename_map = _safe_rename_map(raw_cols)
    df = df.rename(columns=rename_map)
    cols = [rename_map[col] for col in raw_cols]

    assert df.columns.is_unique, "df содержит дублирующиеся имена колонок после rename"
    assert len(cols) == len(set(cols)), "cols содержит дубликаты после rename"
    bad = [c for c in cols if re.search(r'\s', c) or not c]
    assert not bad, f"в именах фич остались пробелы/пустые имена: {bad[:5]}"

    if use_cache:
        df.to_parquet(cache_path, compression="zstd")
        cols_path.write_text(json.dumps({"cols": cols, "raw_cols": raw_cols},
                                        ensure_ascii=False), encoding="utf-8")
        logger.info("Датасет закэширован: %s (%d строк, %d колонок)", cache_path, len(df), len(cols))

    return df, cols, raw_cols


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------
def _clean_current(df: pd.DataFrame, idx: pd.Index, cfg: FeatureConfig) -> pd.Series:
    """Достоверное ТЕКУЩЕЕ значение таргета в момент t (для persistence и lead_time)."""
    cur = df.loc[idx, cfg.target_tag]
    bad = f"{cfg.target_tag}__bad"
    if bad in df:
        cur = cur.where(~df.loc[idx, bad].astype(bool))
    return cur


def _baselines(df: pd.DataFrame, idx: pd.Index, ytr: pd.Series, yte: pd.Series,
               cfg: FeatureConfig) -> dict:
    cur = _clean_current(df, idx, cfg)
    result = {
        "baseline_mae_median": float((yte - ytr.median()).abs().mean()),
        "baseline_mae_persistence": float((yte - cur).abs().mean()),
        "persistence_coverage": float(cur.notna().mean()),
    }
    # Persistence как классификатор: «сера сейчас > limit» = тревога.
    # Если PR-AUC persistence >= модели — добавленная ценность модели нулевая.
    mask = cur.notna() & yte.notna()
    if mask.sum() > 0:
        y_bin = (yte[mask].to_numpy() > cfg.target_limit).astype(int)
        if y_bin.sum() > 0:
            score = cur[mask].clip(lower=0).to_numpy()
            result["baseline_prauc_persistence"] = float(average_precision_score(y_bin, score))
    return result


# --------------------------------------------------------------------------
# Conformal quantile calibration
# --------------------------------------------------------------------------
def _conformal_calibrate(
    preds_val: dict[float, np.ndarray],
    y_val: np.ndarray,
    preds_test: dict[float, np.ndarray],
) -> tuple[dict[float, np.ndarray], float]:
    """Возвращает (калиброванные прогнозы, delta). delta сохраняется в calibration.json
    и применяется агентом на инференсе.

    Сдвигает квантильные прогнозы на константу, подобранную по val,
    чтобы эмпирическое покрытие интервала [p10, p90] стало близко к 0.80.

    Split-conformal в версии «один сдвиг на оба края»: вычисляем нехватку
    покрытия на val и сдвигаем p10 вниз / p90 вверх симметрично на delta,
    пока покрытие не достигнет целевого уровня (или мы не исчерпаем сетку).

    ~5 строк математики, без переобучения — именно то, что надёжно работает
    на коротком val-окне.
    """
    if 0.1 not in preds_val or 0.9 not in preds_val:
        return preds_test, 0.0  # нечего калибровать

    target_coverage = 0.80
    p10_v, p90_v = preds_val[0.1], preds_val[0.9]
    p50_v = preds_val.get(0.5, (p10_v + p90_v) / 2)

    # Бинарный поиск delta: минимальный симметричный сдвиг, дающий нужное покрытие.
    lo, hi = 0.0, float(np.max(np.abs(y_val - p50_v)))
    for _ in range(30):
        mid = (lo + hi) / 2
        coverage = float(np.mean((y_val >= p10_v - mid) & (y_val <= p90_v + mid)))
        if coverage >= target_coverage:
            hi = mid
        else:
            lo = mid
    delta = hi

    calibrated = dict(preds_test)
    calibrated[0.1] = preds_test[0.1] - delta
    calibrated[0.9] = preds_test[0.9] + delta
    logger.info("Conformal calibration: delta=%.4f (val coverage было %.3f -> цель %.2f)",
                delta, float(np.mean((y_val >= p10_v) & (y_val <= p90_v))), target_coverage)
    return calibrated, float(delta)


def _split_train_val(index: pd.Index, embargo: pd.Timedelta) -> tuple[np.ndarray, np.ndarray]:
    """Маски fit/val внутри train: val = последние VAL_DAYS, перед ним эмбарго."""
    idx = pd.DatetimeIndex(index)
    val_start = idx.max() - pd.Timedelta(VAL_DAYS)
    fit = np.asarray(idx <= (val_start - embargo))
    val = np.asarray(idx >= val_start)
    return fit, val


def _write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------
def train_one_horizon(df: pd.DataFrame, cols: list[str], raw_cols: list[str] | None,
                      cfg: FeatureConfig, horizon_points: int, blind: bool,
                      skip_cv: bool = False) -> dict:
    horizon_min = horizon_points * int(pd.Timedelta(cfg.grid_freq).total_seconds() // 60)
    target_col = f"target_{horizon_points}"
    violation_col = f"target_violation_{horizon_points}"

    embargo = embargo_delta(max(cfg.window_points), horizon_points, cfg.grid_freq, cfg.embargo_safety)
    index = pd.DatetimeIndex(df.index)
    holdout = holdout_split(index, cfg.holdout_size, embargo)
    folds = [] if skip_cv else rolling_origin_folds(
        index, cfg.cv_n_folds, cfg.cv_fold_size, embargo,
        expanding=cfg.cv_expanding, holdout=holdout, freq=cfg.grid_freq)

    assert_no_leakage(df, holdout, cols, target_col, horizon_points, cfg.grid_freq)

    train_cfg = LGBMTrainConfig(quantiles=cfg.quantiles, params=cfg.lgbm_params,
                                early_stopping_rounds=cfg.early_stopping_rounds)
    monotone = build_monotone_constraints(cols, cfg.monotone_directions)
    n_mono = sum(1 for m in monotone if m)
    if cfg.monotone_directions and n_mono < 2 * len(cfg.monotone_directions):
        logger.warning("Монотонных ограничений подозрительно мало: %d (по тегам %s). "
                       "Ожидается порядка 8 на тег (сырой + лаги + средние). Проверь "
                       "имена колонок.", n_mono, sorted(cfg.monotone_directions))

    # --- walk-forward CV: без early stopping (фиксированное число деревьев) ---
    cv_metrics = []
    for fold in folds:
        assert_no_leakage(df, fold, cols, target_col, horizon_points, cfg.grid_freq)
        Xtr, ytr, Xte, yte = train_matrix(df, fold, cols, target_col)
        _, ytr_v, _, _ = train_matrix(df, fold, cols, violation_col)
        if len(Xtr) < 500 or len(Xte) < 50:
            logger.warning("%s: слишком мало данных, пропускаю", fold.name)
            continue

        qm = QuantileQualityModel(cols, cfg.target_tag, horizon_min, train_cfg, monotone)
        qm.fit(Xtr, ytr)
        preds = qm.predict(Xte)

        clf = ViolationClassifier(cols, train_cfg, monotone)
        clf.fit(Xtr, ytr_v)
        p_violation = clf.predict_proba(Xte)

        m = {"fold": fold.name, "bias_q50": float((preds[0.5] - yte.to_numpy()).mean())}
        m.update(evaluate_quantiles(yte, preds))
        m.update(_baselines(df, Xte.index, ytr, yte, cfg))
        m.update({f"violation_{k}": v for k, v in
                  evaluate_violation(yte, p_violation, cfg.target_limit).items()})
        cv_metrics.append(m)
        logger.info("%s: MAE(q50)=%.3f (persistence %.3f, median %.3f) PR-AUC=%.3f events=%s",
                    fold.name, m["mae_q50"], m["baseline_mae_persistence"],
                    m["baseline_mae_median"], m["violation_pr_auc"], m.get("violation_n_events"))

    # --- финальная модель: fit | val (early stopping, порог) | holdout (отчёт) ---
    Xtr, ytr, Xte, yte = train_matrix(df, holdout, cols, target_col)
    _, ytr_v, _, _ = train_matrix(df, holdout, cols, violation_col)

    fit_m, val_m = _split_train_val(Xtr.index, embargo)
    Xfit, yfit, yfit_v = Xtr.loc[fit_m], ytr.loc[fit_m], ytr_v.loc[fit_m]
    Xval, yval, yval_v = Xtr.loc[val_m], ytr.loc[val_m], ytr_v.loc[val_m]
    logger.info("fit %d строк | val %d строк | holdout %d строк", len(Xfit), len(Xval), len(Xte))

    quantile_model = QuantileQualityModel(cols, cfg.target_tag, horizon_min, train_cfg, monotone)
    quantile_model.fit(Xfit, yfit, Xval, yval)
    preds_raw = quantile_model.predict(Xte)
    preds_val_raw = quantile_model.predict(Xval)

    # Conformal calibration: сдвигаем квантили так, чтобы покрытие [p10,p90]
    # на holdout стало близко к 0.80 (сейчас 0.63–0.68 без калибровки).
    preds, conformal_delta = _conformal_calibrate(preds_val_raw, yval.to_numpy(), preds_raw)

    violation_model = ViolationClassifier(cols, train_cfg, monotone)
    violation_model.fit(Xfit, yfit_v, Xval, yval_v)
    p_violation_test = violation_model.predict_proba(Xte)
    p_violation_val = violation_model.predict_proba(Xval)

    # порог — по валидации (не по train: там модель почти идеальна)
    threshold = best_threshold(yval, p_violation_val, cfg.target_limit,
                               cfg.violation_min_precision)

    holdout_metrics = {"bias_q50": float((preds[0.5] - yte.to_numpy()).mean())}
    holdout_metrics.update(evaluate_quantiles(yte, preds))
    holdout_metrics.update(_baselines(df, Xte.index, ytr, yte, cfg))
    holdout_metrics.update({f"violation_{k}": v for k, v in
                            evaluate_violation(yte, p_violation_test, cfg.target_limit,
                                               threshold).items()})

    clean_actual = _clean_current(df, Xte.index, cfg)
    holdout_metrics["lead_time"] = lead_time_minutes(
        pd.DatetimeIndex(Xte.index), clean_actual, pd.Series(p_violation_test, index=Xte.index),
        cfg.target_limit, horizon_points, threshold, cfg.grid_freq)

    logger.info("HOLDOUT H=%d %s: MAE(q50)=%.3f (persistence %.3f, median %.3f) "
                "coverage80=%.3f PR-AUC=%.3f (persistence_prauc=%.3f) "
                "recall=%.3f precision=%.3f threshold=%.3f",
                horizon_points, "blind" if blind else "with_analyzer",
                holdout_metrics["mae_q50"], holdout_metrics["baseline_mae_persistence"],
                holdout_metrics["baseline_mae_median"], holdout_metrics.get("coverage_80", float("nan")),
                holdout_metrics["violation_pr_auc"],
                holdout_metrics.get("baseline_prauc_persistence", float("nan")),
                holdout_metrics["violation_recall"],
                holdout_metrics["violation_precision"], threshold)

    # --- сохранение артефактов ---------------------------------------------
    mode = "with_analyzer" if not blind else "blind"
    model_dir = cfg.models_dir / f"h{horizon_points}_{mode}"
    quantile_model.save(model_dir / "quantile")
    violation_model.save(model_dir / "violation")
    _write_json(model_dir / "threshold.json", {"alert_threshold": threshold})
    _write_json(model_dir / "calibration.json", {"conformal_delta": conformal_delta})
    # всё, что нужно инференсу, чтобы пересобрать признаки один в один
    # add_engineered делит load_rel на медиану всего датафрейма; онлайн буфер короткий,
    # поэтому фиксируем медиану обучения (см. features_online.OnlineFeatureBuilder)
    feed_col = _hdt_col(df.columns, "T11") or _hdt_col(df.columns, "F26")
    feed_median = float(df[feed_col].median()) if feed_col and feed_col in df.columns else None
    _write_json(model_dir / "feature_spec.json", {
        "feed_col": feed_col, "feed_median": feed_median,
        "horizon_points": horizon_points, "grid_freq": cfg.grid_freq, "mode": mode,
        "target_tag": cfg.target_tag, "lags_points": list(cfg.lags_points),
        "window_points": list(cfg.window_points),
        "raw_cols": raw_cols, "cols": cols,
        "monotone_directions": cfg.monotone_directions,
        "sulfur_analyzer_columns": list(cfg.sulfur_analyzer_columns),
    })

    top_features = quantile_model.feature_importance(15)
    logger.info("Топ-признаки H=%d (%s):\n%s", horizon_points, mode, top_features)

    return {
        "horizon_points": horizon_points, "horizon_min": horizon_min, "mode": mode,
        "n_fit": len(Xfit), "n_val": len(Xval), "n_test": len(Xte),
        "alert_threshold": threshold, "n_monotone": n_mono,
        "cv": cv_metrics, "holdout": holdout_metrics,
        "top_features": top_features.to_dict(),
        "model_dir": str(model_dir),
    }


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/agents/quality/config.yaml")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Обучить только один горизонт (число точек, напр. 6). "
                             "По умолчанию — все из horizons_points конфига.")
    parser.add_argument("--blind-only", action="store_true",
                        help="Обучить только blind-режим")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Не гонять walk-forward CV (в ~4 раза быстрее)")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Игнорировать кэш датасета и пересобрать признаки заново — "
                             "обязательно после правки кода признаков/конфига")
    parser.add_argument("--no-cache", action="store_true",
                        help="Не читать и не писать кэш датасета вообще")
    args = parser.parse_args()

    cfg = load_feature_config(args.config)

    parquet_path = data_pipeline_settings.converted_data_path / "telemetry_pac_lims.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} не найден — сначала DataPreparer.prepare_data()")
    # привязываем monotone и чёрный список анализаторов к РЕАЛЬНЫМ колонкам;
    # если колонки нет — падаем здесь, а не молча обучаемся с утечкой
    cfg = resolve_cfg_columns(cfg, pq.read_schema(parquet_path).names)
    logger.info("monotone: %s | анализаторы (чёрный список): %s",
                cfg.monotone_directions, cfg.sulfur_analyzer_columns)

    horizons = [args.horizon] if args.horizon else list(cfg.horizons_points)
    modes = [True] if args.blind_only else [True, False]

    report: dict = {"config": str(args.config), "runs": []}
    for horizon_points in horizons:
        for blind in modes:
            mode = "blind" if blind else "with_analyzer"
            logger.info("=== Горизонт %d точек, режим %s ===", horizon_points, mode)
            df, cols, raw_cols = build_dataset(cfg, horizon_points, blind,
                                               use_cache=not args.no_cache,
                                               force_rebuild=args.force_rebuild)
            result = train_one_horizon(df, cols, raw_cols, cfg, horizon_points, blind,
                                       skip_cv=args.skip_cv)
            report["runs"].append(result)
            _write_json(cfg.metrics_export, report)   # пишем после каждого прогона

    logger.info("Отчёт сохранён: %s", cfg.metrics_export)


if __name__ == "__main__":
    main()