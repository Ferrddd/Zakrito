"""QualityAgent — реализация протокола Agent для конкретной модели серы.

Инкапсулирует и quantile-модель, и violation-классификатор для одного
горизонта/режима, но снаружи виден только evaluate(state) -> AgentReport.
Оркестратор не знает про LightGBM — только про этот контракт.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.quality.model import QuantileQualityModel, ViolationClassifier
from src.data_pipeline.feature_config import FeatureConfig, load_feature_config
from src.schemas.process_state import (
    AgentReport,
    Capabilities,
    DataQualityInfo,
    Driver,
    ProcessState,
    QualityPrediction,
)

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


class QualityAgent:
    name = "quality"
    version = VERSION

    def __init__(self, cfg: FeatureConfig, horizon_points: int,
                model_with_analyzer: QuantileQualityModel, clf_with_analyzer: ViolationClassifier,
                model_blind: QuantileQualityModel, clf_blind: ViolationClassifier,
                threshold_with_analyzer: float, threshold_blind: float,
                grid_freq: str = "10min"):
        self.cfg = cfg
        self.horizon_points = horizon_points
        self.horizon_min = horizon_points * int(pd.Timedelta(grid_freq).total_seconds() // 60)
        self.models = {"with_analyzer": (model_with_analyzer, clf_with_analyzer, threshold_with_analyzer),
                      "blind": (model_blind, clf_blind, threshold_blind)}

    @classmethod
    def load(cls, config_path: str | Path = "src/agents/quality/config.yaml",
             horizon_points: int | None = None) -> QualityAgent:
        cfg = load_feature_config(config_path)
        horizon_points = horizon_points or cfg.horizons_points[0]

        def load_mode(mode: str):
            model_dir = cfg.models_dir / f"h{horizon_points}_{mode}"
            qm = QuantileQualityModel.load(model_dir / "quantile")
            clf = ViolationClassifier.load(model_dir / "violation")
            import json
            threshold = json.loads((model_dir / "threshold.json").read_text())["alert_threshold"]
            return qm, clf, threshold

        qm_a, clf_a, thr_a = load_mode("with_analyzer")
        qm_b, clf_b, thr_b = load_mode("blind")
        return cls(cfg, horizon_points, qm_a, clf_a, qm_b, clf_b, thr_a, thr_b, cfg.grid_freq)

    def capabilities(self) -> Capabilities:
        required = sorted(set(self.models["with_analyzer"][0].feature_cols)
                          | set(self.models["blind"][0].feature_cols))
        return Capabilities(
            name=self.name, version=self.version,
            indicators=[self.cfg.target_tag],
            horizons_min=[self.horizon_min],
            required_tags=required,
        )

    def _pak_status(self, tags: dict[str, float]) -> tuple[str, float | None]:
        bad = tags.get(f"{self.cfg.target_tag}__bad")
        frozen = tags.get(f"{self.cfg.target_tag}__frozen")
        age = tags.get(f"{self.cfg.target_tag}__age_min")
        if bad is None:
            return "missing", age
        if frozen:
            return "frozen", age
        if bad:
            return "out_of_range", age
        return "ok", age

    def evaluate(self, state: ProcessState) -> AgentReport:
        tags = state.tags
        pak_status, pak_age = self._pak_status(tags)
        mode = "blind" if pak_status != "ok" else "with_analyzer"
        model, clf, threshold = self.models[mode]

        missing = [c for c in model.feature_cols if c not in tags]
        if missing:
            logger.warning("Не хватает тегов для %s: %s", mode, missing)
            return self._abstain(state, pak_status, pak_age, missing,
                                 reason=f"Не хватает {len(missing)} тегов для расчёта прогноза")

        row = pd.DataFrame([{c: tags[c] for c in model.feature_cols}])
        preds = model.predict(row)
        p_violation = float(clf.predict_proba(row)[0])

        prediction = QualityPrediction(
            indicator=self.cfg.target_tag, unit=self.cfg.target_unit,
            horizon_min=self.horizon_min,
            p10=float(preds[min(preds)][0]), p50=float(preds[0.5][0]), p90=float(preds[max(preds)][0]),
            limit=self.cfg.target_limit, p_violation=p_violation, source_model=mode,
        )

        confidence = self._confidence(pak_age, mode)
        drivers = self._drivers(model, row)

        abstain, reason = False, None
        if pak_status != "ok" and pak_age is not None and pak_age > 24 * 60:
            abstain = True
            reason = (f"Последнее достоверное измерение {self.cfg.target_tag} устарело "
                     f"на {pak_age:.0f} мин — надёжной рекомендации нет")

        return AgentReport(
            agent=self.name, version=self.version, trace_id=state.trace_id,
            timestamp=state.timestamp, predictions=[prediction],
            data_quality=DataQualityInfo(pak_status=pak_status, pak_age_min=pak_age,
                                        missing_tags=[]),
            confidence=confidence, drivers=drivers, abstain=abstain, reason=reason,
        )

    def _abstain(self, state: ProcessState, pak_status: str, pak_age: float | None,
                missing: list[str], reason: str) -> AgentReport:
        return AgentReport(
            agent=self.name, version=self.version, trace_id=state.trace_id,
            timestamp=state.timestamp, predictions=[],
            data_quality=DataQualityInfo(pak_status=pak_status, pak_age_min=pak_age,
                                        missing_tags=missing),
            confidence=0.0, drivers=[], abstain=True, reason=reason,
        )

    @staticmethod
    def _confidence(pak_age: float | None, mode: str) -> float:
        base = 0.9 if mode == "with_analyzer" else 0.6
        if pak_age is None:
            return base
        decay = np.clip(1 - pak_age / (24 * 60), 0.0, 1.0)
        return round(base * (0.5 + 0.5 * decay), 3)

    @staticmethod
    def _drivers(model: QuantileQualityModel, row: pd.DataFrame, top_n: int = 5) -> list[Driver]:
        try:
            imp = model.feature_importance(top_n)
        except Exception:  # noqa: BLE001 — важна не причина, а то, что drivers опциональны
            return []
        return [Driver(tag=tag, shap=float(value)) for tag, value in imp.items()]