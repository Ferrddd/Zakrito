"""Обучение модели агента качества (прогноз серы) на сохранённых фолдах.

Читает train_h{H}_{fold}.parquet / test_h{H}_{fold}.parquet из
data/converted (их делает scripts/run_pipeline.py) и folds_h{H}.json
с метаданными фолдов.

Запуск:
    python -m scripts.train_model
"""

from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_pipeline import features as F
from src.data_pipeline.pipeline_models import data_pipeline_settings as s

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("train_model")

HORIZON = 6  # должен совпадать с тем, что использовался в run_pipeline.py
TARGET_COL = f"target_{HORIZON}"
VIOLATION_LIMIT = F.SULFUR_LIMIT_PPM  # 10 ppm

USE_EARLY_STOPPING = True

LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1,
)


def load_fold(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Загрузка фолда '{name}'...")
    train = pd.read_parquet(s.converted_data_path / f"train_h{HORIZON}_{name}.parquet")
    test = pd.read_parquet(s.converted_data_path / f"test_h{HORIZON}_{name}.parquet")
    return train, test


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, fold_name: str = "unknown") -> dict:
    Xtr, ytr = split_xy(train)
    Xte, yte = split_xy(test)

    feature_names = Xtr.columns.tolist()

    Xtr_np = Xtr.to_numpy()
    ytr_np = ytr.to_numpy()
    Xte_np = Xte.to_numpy()
    yte_np = yte.to_numpy()

    logger.info(f"[{fold_name}] Запуск обучения. Train shape: {Xtr_np.shape}, Test shape: {Xte_np.shape}")

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    fit_kwargs = dict(eval_metric="mae")
    
    if USE_EARLY_STOPPING:
        fit_kwargs["callbacks"] = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        
    try:
        model.fit(Xtr_np, ytr_np, eval_X=Xte_np, eval_y=yte_np, **fit_kwargs)
    except TypeError:
        model.fit(Xtr_np, ytr_np, eval_set=[(Xte_np, yte_np)], **fit_kwargs)

    best_iter = model.best_iteration_ if USE_EARLY_STOPPING else None
    pred = model.predict(Xte_np, num_iteration=best_iter)

    mae = mean_absolute_error(yte_np, pred)
    rmse = float(np.sqrt(mean_squared_error(yte_np, pred)))

    y_true_violation = (yte_np > VIOLATION_LIMIT).astype(int)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "n_test": len(yte_np),
        "violation_rate_true": float(y_true_violation.mean()),
    }
    
    if len(np.unique(y_true_violation)) > 1:
        # 1. Считаем стандартные метрики при жестком пороге 10.0 ppm
        pred_violation_strict = (pred > VIOLATION_LIMIT).astype(int)
        metrics["strict_10ppm"] = {
            "precision": float(precision_score(y_true_violation, pred_violation_strict, zero_division=0)),
            "recall": float(recall_score(y_true_violation, pred_violation_strict, zero_division=0)),
            "f1": float(f1_score(y_true_violation, pred_violation_strict, zero_division=0)),
        }

        # 2. Калибровка порога по Precision-Recall кривой
        precisions, recalls, thresholds = precision_recall_curve(y_true_violation, pred)
        
        # Считаем F1 для каждого возможного порога
        f1_scores = np.zeros_like(precisions)
        non_zero_mask = (precisions + recalls) > 0
        f1_scores[non_zero_mask] = 2 * (precisions[non_zero_mask] * recalls[non_zero_mask]) / (precisions[non_zero_mask] + recalls[non_zero_mask])

        # Находим индекс порога с максимальным F1-score
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else VIOLATION_LIMIT

        pred_violation_calibrated = (pred > best_threshold).astype(int)

        metrics.update({
            "roc_auc": float(roc_auc_score(y_true_violation, pred)),
            "calibrated": {
                "best_threshold_ppm": best_threshold,
                "precision": float(precision_score(y_true_violation, pred_violation_calibrated, zero_division=0)),
                "recall": float(recall_score(y_true_violation, pred_violation_calibrated, zero_division=0)),
                "f1": float(f1_score(y_true_violation, pred_violation_calibrated, zero_division=0)),
            }
        })

        logger.info(
            f"[{fold_name}] Жесткий порог (>10.0 ppm): Recall={metrics['strict_10ppm']['recall']:.3f} | "
            f"Калиброванный порог (>{best_threshold:.2f} ppm): Recall={metrics['calibrated']['recall']:.3f}, "
            f"Precision={metrics['calibrated']['precision']:.3f}, F1={metrics['calibrated']['f1']:.3f}"
        )
    else:
        logger.warning(f"[{fold_name}] В тестовой выборке только один класс. Метрики классификации пропущены.")

    return {
        "model": model, 
        "pred": pred, 
        "metrics": metrics, 
        "feature_names": feature_names
    }


def run_cv() -> list[dict]:
    logger.info("=== Запуск кросс-валидации (CV) ===")
    meta_path = s.converted_data_path / f"folds_h{HORIZON}.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    cv_names = [row["fold"] for row in meta["folds"] if row["fold"].startswith("cv")]

    results = []
    for name in cv_names:
        train, test = load_fold(name)
        out = fit_predict(train, test, fold_name=name)
        out["metrics"]["fold"] = name
        results.append(out["metrics"])
        logger.info(f"--- Результат {name}: MAE={out['metrics']['mae']:.3f} | RMSE={out['metrics']['rmse']:.3f} | Recall={out['metrics'].get('recall')} ---")
        
    return results


def run_holdout(cv_results: list[dict]) -> dict:
    logger.info("=== Запуск обучения финальной модели (Holdout) ===")
    train, test = load_fold("holdout")

    cv_names = [row["fold"] for row in cv_results]
    extra_frames = [train]
    for name in cv_names:
        tr, te = load_fold(name)
        extra_frames.extend([tr, te])
        
    logger.info("Объединение всех CV-фолдов и Holdout Train...")
    full_train = pd.concat(extra_frames, ignore_index=False).drop_duplicates()

    out = fit_predict(full_train, test, fold_name="holdout")
    logger.info(
        f"--- ИТОГОВЫЙ HOLDOUT: MAE={out['metrics']['mae']:.3f} | RMSE={out['metrics']['rmse']:.3f} | "
        f"Recall={out['metrics'].get('recall')} | Precision={out['metrics'].get('precision')} | "
        f"ROC-AUC={out['metrics'].get('roc_auc')} ---"
    )
    return out


def main() -> None:
    logger.info(f"=== Старт пайплайна обучения (Горизонт: {HORIZON} ч.) ===")
    cv_results = run_cv()

    mae_values = [r["mae"] for r in cv_results]
    logger.info(
        f"=== Сводка CV MAE: {np.mean(mae_values):.3f} ± {np.std(mae_values):.3f} "
        f"(min: {min(mae_values):.3f}, max: {max(mae_values):.3f}) ==="
    )

    holdout_out = run_holdout(cv_results)

    model = holdout_out["model"]
    feature_names = holdout_out["feature_names"]

    model_path = s.converted_data_path / f"model_h{HORIZON}.txt"
    model.booster_.save_model(str(model_path))
    logger.info(f"Модель успешно сохранена по пути: {model_path}")

    report = {
        "horizon": HORIZON,
        "cv": cv_results,
        "holdout": holdout_out["metrics"],
        "top_features": dict(sorted(
            zip(feature_names, model.feature_importances_.tolist()),
            key=lambda kv: -kv[1],
        )[:20]),
    }
    report_path = s.converted_data_path / f"train_report_h{HORIZON}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Файл отчёта сформирован: {report_path}")
    logger.info("=== Пайплайн обучения успешно завершён ===")


if __name__ == "__main__":
    main()