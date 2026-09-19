"""Метрики агента качества.

Кроме стандартных MAE/RMSE считаем три вещи, которые важнее для этой задачи:

- pinball loss и coverage квантильных голов — иначе легко получить красивый
  MAE и разъехавшиеся, бесполезные для решения о риске квантили;
- PR-AUC/recall по событию «нарушение через H минут» — целевой класс
  небалансный (~11-16% по годам), обычный accuracy бессмысленен;
- lead time — за сколько минут ДО реального превышения система подала
  первый сигнал. Эта метрика продаёт решение жюри лучше R².
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def evaluate_quantiles(y_true: pd.Series, preds: dict[float, np.ndarray]) -> dict:
    """preds: {квантиль: массив прогнозов}. Ожидает как минимум 0.1/0.5/0.9."""
    y = y_true.to_numpy()
    out = {"mae_q50": float(mean_absolute_error(y, preds[0.5])),
           "rmse_q50": float(np.sqrt(np.mean((y - preds[0.5]) ** 2)))}
    for q, p in preds.items():
        out[f"pinball_q{q}"] = pinball_loss(y, p, q)

    if 0.1 in preds and 0.9 in preds:
        inside = (y >= preds[0.1]) & (y <= preds[0.9])
        out["coverage_80"] = float(inside.mean())  # цель ~0.80, если квантили откалиброваны
        out["mean_interval_width"] = float(np.mean(preds[0.9] - preds[0.1]))
    return out


def evaluate_violation(y_true_value: pd.Series, p_violation: np.ndarray,
                       limit: float, threshold: float = 0.5) -> dict:
    """PR-AUC и recall/precision по факту y_true_value > limit."""
    y_bin = (y_true_value.to_numpy() > limit).astype(int)
    if y_bin.sum() == 0:
        logger.warning("В выборке нет положительного класса — PR-AUC не определён")
        return {"pr_auc": float("nan"), "recall": float("nan"),
                "precision": float("nan"), "n_events": 0}

    pr_auc = float(average_precision_score(y_bin, p_violation))
    pred_bin = (p_violation >= threshold).astype(int)
    tp = int(((pred_bin == 1) & (y_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (y_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (y_bin == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return {"pr_auc": pr_auc, "recall": recall, "precision": precision,
            "n_events": int(y_bin.sum()), "threshold": threshold}


def best_threshold(y_true_value: pd.Series, p_violation: np.ndarray,
                   limit: float, min_precision: float = 0.5) -> float:
    """Порог с максимальным recall при precision >= min_precision.

    Решение о нарушении — цена пропуска (нарушить спецификацию) намного выше
    цены ложной тревоги, поэтому оптимизируем recall, а не F1.
    """
    y_bin = (y_true_value.to_numpy() > limit).astype(int)
    if y_bin.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_bin, p_violation)
    precision, recall = precision[:-1], recall[:-1]
    ok = precision >= min_precision
    if not ok.any():
        logger.warning("Ни один порог не даёт precision >= %.2f, беру max recall", min_precision)
        return float(thresholds[int(np.argmax(recall))])
    idx = np.argmax(np.where(ok, recall, -1))
    return float(thresholds[idx])


def lead_time_minutes(index: pd.DatetimeIndex, clean_actual: pd.Series,
                      p_violation: pd.Series, limit: float, horizon_points: int,
                      alert_threshold: float, freq: str = "10min") -> dict:
    """Средний/медианный запас времени до реального превышения.

    p_violation(t) — вероятность, что значение в момент t + horizon превысит
    limit (прогноз, сделанный в момент t). Событие — непрерывный отрезок
    реального сигнала clean_actual выше limit. Для события ищем самый ранний
    момент t, чей прогнозный горизонт (t + horizon) попадает внутрь события
    и который система уже флагнула тревогой; lead = начало события - t.

    Если система не подала сигнал заранее ни разу — событие пропущено
    (не входит в среднее, считается отдельно как miss_rate).
    """
    step = pd.Timedelta(freq)
    horizon_td = step * horizon_points

    is_violation = clean_actual > limit
    event_id = (is_violation != is_violation.shift(fill_value=False)).cumsum()
    events = (pd.DataFrame({"violation": is_violation, "event_id": event_id}, index=index)
              .query("violation")
              .groupby("event_id").apply(lambda g: (g.index.min(), g.index.max())))

    leads = []
    missed = 0
    for start, _ in events:
        # прогнозные точки, чей "прицел" (t+horizon) попадает в начало события
        window_start = start - horizon_td * 3  # ищем сигнал не глубже 3 горизонтов до события
        candidates = p_violation.loc[window_start:start]
        candidates = candidates[candidates.index + horizon_td <= start + step]
        fired = candidates[candidates >= alert_threshold]
        if fired.empty:
            missed += 1
            continue
        first_alert = fired.index.min()
        leads.append((start - first_alert).total_seconds() / 60)

    n_events = len(events)
    return {
        "n_events": n_events,
        "n_detected_early": len(leads),
        "miss_rate": missed / n_events if n_events else float("nan"),
        "lead_time_mean_min": float(np.mean(leads)) if leads else float("nan"),
        "lead_time_median_min": float(np.median(leads)) if leads else float("nan"),
    }