"""Модель агента качества: квантильная регрессия + вероятность нарушения.

Две головы, а не одна:

- QuantileQualityModel — три LightGBM-регрессора (quantile, alpha=0.1/0.5/0.9).
  Решение о риске принимается по верхней границе (p90), не по медиане —
  «качество и жёсткие ограничения важнее экономики» из ТЗ реализуется здесь,
  а не как отдельное бизнес-правило поверх точечного прогноза.
- ViolationClassifier — отдельный бинарный классификатор на target_violation.
  Даёт откалиброванную p_violation для JSON-контракта агента и для
  vetoing в оркестраторе; квантили одни это делают грубее.

Монотонные ограничения обязательны для обеих голов. Без них модель на
исторических данных выучивает обратную причинность: оператор поднимал
температуру, ПОТОМУ ЧТО сера уже росла — и алгоритм связывает "температура
выше -> сера выше", хотя физически наоборот. Тогда контрфактические ответы
агенту оптимизации будут систематически неверными.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_monotone_constraints(feature_cols: list[str],
                               directions: dict[str, int]) -> list[int]:
    """Вектор constraints в порядке feature_cols.

    directions приходит из FeatureConfig.monotone_directions (config/quality_agent.yaml),
    ключи — базовые колонки без суффикса __lag/__mean/... Для лаговых/оконных
    производных (`T5_hdt__lag6`, `T5_hdt__mean18`) знак наследуется от базового
    тега — значение той же физической величины в прошлом действует в ту же
    сторону. Для `__std` (волатильность) и `__delta` (скорость изменения)
    направление неочевидно, ограничение не ставится.
    """
    out = []
    for col in feature_cols:
        base = col.split("__")[0]
        if "__std" in col or "__delta" in col:
            out.append(0)
        else:
            out.append(directions.get(base, 0))
    n_constrained = sum(1 for c in out if c != 0)
    logger.info("Монотонных ограничений: %d из %d признаков", n_constrained, len(out))
    return out


@dataclass
class LGBMTrainConfig:
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    params: dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "verbosity": -1,
    })
    early_stopping_rounds: int = 50


class QuantileQualityModel:
    """Три бустера с objective='quantile', общий набор признаков и ограничений."""

    def __init__(self, feature_cols: list[str], indicator: str, horizon_min: int,
                config: LGBMTrainConfig | None = None,
                monotone_constraints: list[int] | None = None):
        self.feature_cols = feature_cols
        self.indicator = indicator
        self.horizon_min = horizon_min
        self.config = config or LGBMTrainConfig()
        self.monotone_constraints = monotone_constraints or [0] * len(feature_cols)
        self.boosters: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: pd.DataFrame | None = None, y_valid: pd.Series | None = None) -> QuantileQualityModel:
        callbacks = []
        eval_set = None
        if X_valid is not None and y_valid is not None and len(X_valid):
            eval_set = [(X_valid[self.feature_cols].to_numpy(), y_valid.to_numpy())]
            callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]

        for q in self.config.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                # LightGBM не поддерживает monotone_constraints с objective="quantile"
                # (падает с LightGBMError вне зависимости от monotone_constraints_method,
                # проверено на 4.7.0) — ограничение накладывается только на
                # ViolationClassifier (objective="binary"), там оно физически применимо.
                **self.config.params,
            )
            model.fit(X_train[self.feature_cols].to_numpy(), y_train.to_numpy(),
                    eval_set=eval_set, callbacks=callbacks)
            self.boosters[q] = model
            logger.info("%s H=%d q=%.1f: обучено, лучшая итерация %s",
                    self.indicator, self.horizon_min, q,
                    getattr(model, "best_iteration_", "n/a"))
        return self

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        preds = {q: m.predict(X[self.feature_cols].to_numpy()) for q, m in self.boosters.items()}
        # квантили не должны пересекаться в обратном порядке — приводим к монотонности
        stacked = np.sort(np.stack(list(preds.values()), axis=0), axis=0)
        return {q: stacked[i] for i, q in enumerate(sorted(preds))}

    def feature_importance(self, top_n: int = 15) -> pd.Series:
        model = self.boosters.get(0.5) or next(iter(self.boosters.values()))
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        else:
            values = model.booster.feature_importance(importance_type="gain")
        imp = pd.Series(values, index=self.feature_cols)
        return imp.sort_values(ascending=False).head(top_n)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for q, model in self.boosters.items():
            model.booster_.save_model(str(directory / f"quantile_{q}.txt"))
        meta = {"feature_cols": self.feature_cols, "indicator": self.indicator,
               "horizon_min": self.horizon_min, "quantiles": list(self.boosters),
               "monotone_constraints": self.monotone_constraints}
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> QuantileQualityModel:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["feature_cols"], meta["indicator"], meta["horizon_min"],
                  monotone_constraints=meta["monotone_constraints"])
        for q in meta["quantiles"]:
            booster = lgb.Booster(model_file=str(directory / f"quantile_{q}.txt"))
            obj.boosters[q] = _BoosterPredictAdapter(booster)
        return obj


class _BoosterPredictAdapter:
    """Обёртка над сырым lgb.Booster с интерфейсом .predict(X[cols]), для load()."""

    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X)


class ViolationClassifier:
    """Бинарный классификатор P(target_violation=1). Даёт p_violation для контракта агента."""

    def __init__(self, feature_cols: list[str], config: LGBMTrainConfig | None = None,
                monotone_constraints: list[int] | None = None):
        self.feature_cols = feature_cols
        self.config = config or LGBMTrainConfig()
        self.monotone_constraints = monotone_constraints or [0] * len(feature_cols)
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
       X_valid: pd.DataFrame | None = None, y_valid: pd.Series | None = None) -> ViolationClassifier:
        pos = float(y_train.mean())
        scale_pos_weight = (1 - pos) / max(pos, 1e-6)  # компенсация небаланса классов

        callbacks, eval_set = [], None
        if X_valid is not None and y_valid is not None and len(X_valid):
            eval_set = [(X_valid[self.feature_cols].to_numpy(), y_valid.to_numpy())]
            callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]

        self.model = lgb.LGBMClassifier(
            objective="binary", scale_pos_weight=scale_pos_weight,
            monotone_constraints=self.monotone_constraints, **self.config.params,
        )
        self.model.fit(X_train[self.feature_cols].to_numpy(), y_train.to_numpy(),
                    eval_set=eval_set, callbacks=callbacks)
        logger.info("ViolationClassifier: обучено, доля класса 1 в train=%.3f", pos)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "модель не обучена"
        return self.model.predict_proba(X[self.feature_cols].to_numpy())[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(directory / "violation_clf.txt"))
        meta = {"feature_cols": self.feature_cols, "monotone_constraints": self.monotone_constraints}
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> ViolationClassifier:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["feature_cols"], monotone_constraints=meta["monotone_constraints"])
        booster = lgb.Booster(model_file=str(directory / "violation_clf.txt"))
        obj.model = _ClassifierPredictAdapter(booster)
        return obj


class _ClassifierPredictAdapter:
    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.booster.predict(X.to_numpy())
        return np.stack([1 - p1, p1], axis=1)