from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.reliability.nbm import (
    EPS,
    NormalBehaviorModel,
    StackedRegressor,
    build_features,
    fit_with_early_stopping,
    make_linear,
    make_regressor,
    mask_sentinels,
    resolve_column,
    smooth_target,
)
from src.utils.config import BASE_DIR, load_config

logger = logging.getLogger(__name__)

MAD = 1.4826


def _split_masks(index: pd.DatetimeIndex, holdout_size: str, embargo: str) -> tuple[np.ndarray, np.ndarray]:
    end = index.max()
    test_start = end - pd.Timedelta(holdout_size)
    train_end = test_start - pd.Timedelta(embargo)
    return np.asarray(index <= train_end), np.asarray(index >= test_start)


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x))) * MAD) if len(x) else float("nan")


def describe_target(y: pd.Series, usable: pd.Series, name: str) -> None:
    """Короткая сводка по целевому тегу: помогает увидеть сдвиги уровня и мусор в данных."""
    yy = y[usable.to_numpy()]
    if yy.empty:
        return
    q = yy.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    logger.info("%s: n=%d p1=%.3g p5=%.3g p50=%.3g p95=%.3g p99=%.3g, доля <=0: %.1f%%", name, len(yy),
                q.iloc[0], q.iloc[1], q.iloc[2], q.iloc[3], q.iloc[4], 100 * float((yy <= 0).mean()))


def train(df: pd.DataFrame, cfg: dict[str, Any]) -> NormalBehaviorModel:
    """Чистая функция (без ввода-вывода): витрина + конфиг -> обученная модель."""
    nb = cfg["nbm"]
    suffixes = cfg["suffixes"]
    roll, lag = int(nb["roll_points"]), int(nb["lag_points"])

    target_col = resolve_column(nb["target"], df.columns, suffixes)
    if target_col is None:
        raise KeyError(f"Целевой тег ΔP '{nb['target']}' не найден в витрине")
    feed_col = resolve_column(cfg["tags"]["feed"], df.columns, suffixes)
    base_cols: list[str] = []
    for tag in nb["features"]:
        c = resolve_column(tag, df.columns, suffixes)
        if c is None:
            logger.warning("NBM: признак %s не найден в витрине — пропускаю", tag)
        elif c != target_col and c not in base_cols:
            base_cols.append(c)
    if len(base_cols) < 3:
        raise ValueError(f"Слишком мало признаков для NBM: {base_cols}")
    logger.info("NBM: таргет=%s, признаки=%s, лаг=%d точек", target_col, base_cols, lag)
    df = mask_sentinels(df, cfg.get("sentinel_values", []), base_cols + [target_col])

    X = build_features(df, base_cols, target_col, roll, lag)
    dp_s = smooth_target(df, target_col, roll)
    dp_lag = dp_s.shift(lag)
    y_log = np.log(dp_s.clip(lower=EPS)) - np.log(dp_lag.clip(lower=EPS))     # во сколько раз изменился ΔP

    on = pd.Series(True, index=df.index)
    if "is_valid" in df.columns:
        on &= df["is_valid"].fillna(False).astype(bool)
    if feed_col is not None:
        on &= df[feed_col] >= cfg["baseline_min_load_ratio"] * float(df[feed_col].median())
    if "hours_since_block_start" in df.columns:
        on &= df["hours_since_block_start"] >= float(nb["startup_exclude_h"])
    if "anomaly_share" in df.columns:
        on &= ~(df["anomaly_share"].fillna(0) > cfg["anomaly_share_hard"])
    usable = X.notna().all(axis=1) & dp_s.notna() & on & on.shift(lag, fill_value=False)
    logger.info("NBM: пригодно %d из %d строк", int(usable.sum()), len(df))
    describe_target(dp_s, usable, target_col)
    train_m, hold_m = _split_masks(pd.DatetimeIndex(df.index), nb["holdout_size"], nb["embargo"])
    tr_all = usable.to_numpy() & train_m
    ho = usable.to_numpy() & hold_m
    if tr_all.sum() < 2000 or ho.sum() < 200:
        raise ValueError(f"Мало данных: train={int(tr_all.sum())}, holdout={int(ho.sum())}")

    tr_idx = np.flatnonzero(tr_all)
    cut = int(len(tr_idx) * (1 - float(nb["val_frac"])))
    gap = int(pd.Timedelta(nb["embargo"]) / pd.Timedelta(cfg.get("grid_freq", "10min")))
    fit_idx, val_idx = tr_idx[:cut], tr_idx[min(cut + gap, len(tr_idx) - 1):]

    yl = y_log.to_numpy(dtype=float)
    params = nb["params"]
    lin = make_linear(float(nb["ridge_alpha"])).fit(X.iloc[fit_idx], yl[fit_idx])
    lin_res_fit = yl[fit_idx] - lin.predict(X.iloc[fit_idx])
    lin_res_val = yl[val_idx] - lin.predict(X.iloc[val_idx])
    gbt = make_regressor(params)
    n_trees = fit_with_early_stopping(gbt, X.iloc[fit_idx], lin_res_fit, X.iloc[val_idx], lin_res_val,
                                      int(nb["early_stopping_rounds"]))
    stacked = StackedRegressor(lin, gbt)
    logger.info("NBM: бустинг по остатку линейной части, деревьев: %d", n_trees)

    # ---- честная оценка на holdout (модель его не видела)
    h = np.flatnonzero(ho)
    lw = np.exp(X["log_dp_lag"].to_numpy()[h])
    pred = lw * np.exp(stacked.predict(X.iloc[h]))
    actual = dp_s.to_numpy(dtype=float)[h]
    persist = lw                                            # наивный прогноз: ΔP не изменился за лаг
    mae_model, mae_pers = float(np.mean(np.abs(actual - pred))), float(np.mean(np.abs(actual - persist)))
    ss_res, ss_tot = float(np.sum((actual - pred) ** 2)), float(np.sum((actual - actual.mean()) ** 2))
    rel = (actual - pred) / pred
    sigma = max(_robust_sigma(rel), float(nb["min_rel_sigma"]))
    smooth = pd.Series(rel, index=df.index[h]).rolling(int(nb["smooth_points"]), min_periods=3).median() / sigma
    med_dp = float(np.median(dp_s[tr_all]))
    big = np.abs(actual) > 0.1 * abs(med_dp)
    metrics: dict[str, Any] = {
        "target": target_col, "features": base_cols, "lag_points": lag,
        "n_train": len(fit_idx), "n_val": len(val_idx), "n_holdout": len(h), "n_trees": n_trees,
        "holdout_mae": mae_model, "holdout_mae_persistence": mae_pers,
        "holdout_skill_vs_persistence": 1 - mae_model / mae_pers if mae_pers > 0 else float("nan"),
        "holdout_r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "holdout_mape_pct": float(np.mean(np.abs(actual - pred)[big] / np.abs(actual)[big]) * 100) if big.any() else None,
        "resid_sigma_rel": sigma,
        # сколько «лишних тревог» дал бы остаток на holdout при текущих порогах агента
        "holdout_share_z_gt_soft": float(np.nanmean(smooth > cfg["z_soft"])),
        "holdout_share_z_gt_hard": float(np.nanmean(smooth > cfg["z_hard"])),
        "holdout_start": str(df.index[h].min()), "holdout_end": str(df.index[h].max()),
    }
    logger.info("NBM holdout: MAE модели %.4g vs наивный «ΔП не менялся» %.4g -> skill=%.3f | R2=%.3f | "
                "σ относительного остатка=%.3f | z>soft %.1f%%, z>hard %.1f%%",
                mae_model, mae_pers, metrics["holdout_skill_vs_persistence"], metrics["holdout_r2"], sigma,
                100 * metrics["holdout_share_z_gt_soft"], 100 * metrics["holdout_share_z_gt_hard"])
    skill = metrics["holdout_skill_vs_persistence"]
    if not np.isfinite(skill) or skill < float(nb["min_skill"]):
        logger.warning("NBM: skill %.3f < %.2f — модель не лучше наивного прогноза, агент её использовать не "
                       "будет (останется правило-основанный ΔP)", skill, nb["min_skill"])

    # ---- финальная модель: вся пригодная история, число деревьев как в early stopping
    all_idx = np.flatnonzero(usable.to_numpy() & (train_m | hold_m))
    lin_f = make_linear(float(nb["ridge_alpha"])).fit(X.iloc[all_idx], yl[all_idx])
    gbt_f = make_regressor(params, n_estimators=max(n_trees, 50)).fit(
        X.iloc[all_idx], yl[all_idx] - lin_f.predict(X.iloc[all_idx]))
    return NormalBehaviorModel(model=StackedRegressor(lin_f, gbt_f), base_cols=base_cols, target_col=target_col,
                               feed_col=feed_col, roll=roll, lag=lag, resid_scale=sigma,
                               columns=list(X.columns), metrics=metrics)


def main() -> None:
    from src.data_pipeline.pipeline_models import data_pipeline_settings
    from src.utils.config import setup_logging

    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/agents/reliability/config.yaml")
    ap.add_argument("--data", default=None, help="parquet витрины (по умолчанию telemetry_pac_lims.parquet)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = Path(args.data) if args.data else data_pipeline_settings.converted_data_path / "telemetry_pac_lims.parquet"
    df = pd.read_parquet(path)
    nbm = train(df, cfg)

    out = Path(cfg["nbm"]["model_path"])
    out = out if out.is_absolute() else BASE_DIR / out
    nbm.save(out)
    (out.with_suffix(".metrics.json")).write_text(
        json.dumps(nbm.metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Сохранено: %s", out)


if __name__ == "__main__":
    main()
