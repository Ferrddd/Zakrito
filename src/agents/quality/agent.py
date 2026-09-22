

from __future__ import annotations

import json
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.quality.features_online import OnlineFeatureBuilder, to_regular_grid
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

VERSION = "1.1.0"
MODES = ("with_analyzer", "blind")


@dataclass
class ModeBundle:
    """Всё, что нужно для прогноза на одном горизонте в одном режиме."""
    horizon_points: int
    mode: str
    model: QuantileQualityModel
    clf: ViolationClassifier
    threshold: float
    builder: OnlineFeatureBuilder
    conformal_delta: float = 0.0


def _num(x: object) -> float | None:
    """None/NaN/нечисло -> None, иначе float."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


class QualityAgent:
    name = "quality"
    version = VERSION

    def __init__(self, cfg: FeatureConfig, bundles: list[ModeBundle], grid_freq: str = "10min",
                 max_history_points: int = 288, max_pak_age_min: float = 24 * 60,
                 max_missing_share: float = 0.3):
        if not any(b.mode == "blind" for b in bundles):
            raise ValueError("Нужна хотя бы одна blind-модель: это режим-фолбэк при отказе ПАК")
        self.cfg = cfg
        self.grid_freq = grid_freq
        self.step_min = int(pd.Timedelta(grid_freq).total_seconds() // 60)
        self.bundles = {(b.horizon_points, b.mode): b for b in bundles}
        self.horizons = sorted({b.horizon_points for b in bundles})
        self.max_history_points = max_history_points
        self.max_pak_age_min = max_pak_age_min
        self.max_missing_share = max_missing_share
        self._rows: OrderedDict[pd.Timestamp, dict[str, float]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ загрузка
    @classmethod
    def load(cls, config_path: str | Path = "src/agents/quality/config.yaml",
             horizons_points: list[int] | None = None, **kwargs: Any) -> QualityAgent:
        """Загружает все горизонты, для которых есть артефакты (h{H}_{mode}/)."""
        cfg = load_feature_config(config_path)
        wanted = horizons_points or list(cfg.horizons_points)
        bundles: list[ModeBundle] = []
        for h in wanted:
            for mode in MODES:
                model_dir = cfg.models_dir / f"h{h}_{mode}"
                if not (model_dir / "threshold.json").exists():
                    logger.info("Нет артефактов %s — пропускаю", model_dir)
                    continue
                if not (model_dir / "feature_spec.json").exists():
                    logger.warning("%s: нет feature_spec.json (обучено старой версией train.py) — "
                                   "пропускаю, переобучите: python -m src.agents.quality.train "
                                   "--horizon %d --skip-cv --force-rebuild", model_dir, h)
                    continue
                cal = model_dir / "calibration.json"
                if not cal.exists():
                    logger.warning("%s: нет calibration.json (модель обучена до патча) — "
                                   "квантили без conformal-сдвига", model_dir)
                bundles.append(ModeBundle(
                    horizon_points=h, mode=mode,
                    model=QuantileQualityModel.load(model_dir / "quantile"),
                    clf=ViolationClassifier.load(model_dir / "violation"),
                    threshold=json.loads((model_dir / "threshold.json").read_text())["alert_threshold"],
                    builder=OnlineFeatureBuilder.from_spec_file(model_dir / "feature_spec.json"),
                    conformal_delta=(json.loads(cal.read_text())["conformal_delta"] if cal.exists() else 0.0),
                ))
        if not bundles:
            raise FileNotFoundError(f"В {cfg.models_dir} нет обученных моделей для горизонтов {wanted}. "
                                    "Запусти: make train_quality")
        logger.info("QualityAgent: горизонты %s, режимы %s", sorted({b.horizon_points for b in bundles}),
                    sorted({b.mode for b in bundles}))
        return cls(cfg, bundles, cfg.grid_freq, **kwargs)

    # ------------------------------------------------------------------ контракт
    def capabilities(self) -> Capabilities:
        required = sorted({t for b in self.bundles.values() for t in b.builder.base_tags})
        return Capabilities(
            name=self.name, version=self.version, indicators=[self.cfg.target_tag],
            horizons_min=[h * self.step_min for h in self.horizons], required_tags=required)

    # ------------------------------------------------------------------ история
    def _to_grid(self, ts: Any) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_localize(None)   # витрина безтайм-зонная; берём «настенное» время
        return t.floor(self.grid_freq)

    def observe(self, state: ProcessState) -> None:
        """Добавить точку в буфер (идемпотентно по времени: тот же timestamp перезаписывается)."""
        with self._lock:
            self._rows[self._to_grid(state.timestamp)] = {
                k: v for k, v in ((k, _num(v)) for k, v in state.tags.items()) if v is not None}
            while len(self._rows) > self.max_history_points:
                self._rows.popitem(last=False)

    def warm_up(self, frame: pd.DataFrame) -> int:
        """Загрузить историю из витрины (индекс — время). Возвращает число загруженных точек."""
        tail = frame.tail(self.max_history_points)
        with self._lock:
            for ts, row in tail.iterrows():
                self._rows[self._to_grid(ts)] = {str(k): f for k, v in row.items() if (f := _num(v)) is not None}
            self._rows = OrderedDict(sorted(self._rows.items())[-self.max_history_points:])
        return len(tail)

    def _history(self, now: pd.Timestamp) -> pd.DataFrame:
        with self._lock:
            df = pd.DataFrame.from_dict(dict(self._rows), orient="index").sort_index()
        df = df[df.index <= now]
        # регулярная сетка: пропущенные такты = NaN, чтобы лаги в точках означали лаги во времени
        return to_regular_grid(df, self.grid_freq)

    # ------------------------------------------------------------------ статус ПАК
    def _pak_status(self, tags: dict[str, float | None]) -> tuple[str, float | None]:
        t = self.cfg.target_tag
        bad, frozen, age = _num(tags.get(f"{t}__bad")), _num(tags.get(f"{t}__frozen")), _num(tags.get(f"{t}__age_min"))
        if bad is None:
            return "missing", age
        if frozen:
            return "frozen", age
        if bad:
            return "out_of_range", age
        return "ok", age

    # ------------------------------------------------------------------ основной метод
    def evaluate(self, state: ProcessState, overrides: dict[str, float] | None = None) -> AgentReport:
        self.observe(state)
        now = self._to_grid(state.timestamp)
        pak_status, pak_age = self._pak_status(state.tags)
        mode = "with_analyzer" if pak_status == "ok" else "blind"
        notes: list[str] = []

        if overrides:
            notes.append("what-if: подменены " + ", ".join(f"{k}={v}" for k, v in overrides.items()))
        if mode == "with_analyzer" and not any(m == "with_analyzer" for _, m in self.bundles):
            mode = "blind"
            notes.append("with_analyzer-модели не загружены — использую blind")

        bundles = [self.bundles[(h, mode)] for h in self.horizons if (h, mode) in self.bundles]
        if not bundles:  # для части горизонтов нет with_analyzer
            mode = "blind"
            bundles = [self.bundles[(h, "blind")] for h in self.horizons if (h, "blind") in self.bundles]

        hist = self._history(now)
        n_hist = int(hist.notna().any(axis=1).sum())
        if overrides:
            unknown = [k for k in overrides if k not in hist.columns]
            if unknown:
                return self._abstain(state, pak_status, pak_age, [], n_hist,
                                     f"Сценарий содержит неизвестные теги: {unknown}")
            hist = hist.copy()
            for k, v in overrides.items():
                hist.loc[hist.index[-1], k] = v

        need = max(b.builder.min_history_points for b in bundles)
        if n_hist < need:
            return self._abstain(state, pak_status, pak_age, [], n_hist,
                                 f"Недостаточно истории: {n_hist} из {need} точек "
                                 f"(прогрев буфера — вызови warm_up или подожди)")

        base_tags = sorted({t for b in bundles for t in b.builder.base_tags})
        cur = hist.iloc[-1]
        missing = [t for t in base_tags if t not in hist.columns or pd.isna(cur[t])]
        if len(missing) > self.max_missing_share * len(base_tags):
            return self._abstain(state, pak_status, pak_age, missing, n_hist,
                                 f"Нет текущих значений у {len(missing)} из {len(base_tags)} тегов")

        preds: list[QualityPrediction] = []
        rows: dict[int, pd.DataFrame] = {}
        for b in bundles:
            row = b.builder.build(hist)
            rows[b.horizon_points] = row
            q = b.model.predict(row)
            p10, p50, p90 = float(q[min(q)][0]), float(q[0.5][0]), float(q[max(q)][0])
            p10, p90 = max(p10 - b.conformal_delta, 0.0), p90 + b.conformal_delta
            p_viol = float(b.clf.predict_proba(row)[0])
            preds.append(QualityPrediction(
                indicator=self.cfg.target_tag, unit=self.cfg.target_unit,
                horizon_min=b.horizon_points * self.step_min,
                p10=p10, p50=max(p50, 0.0), p90=p90, limit=self.cfg.target_limit,
                p_violation=p_viol, alert_threshold=b.threshold, alert=p_viol >= b.threshold,
                p90_over_limit=p90 > self.cfg.target_limit, source_model=mode))  # type: ignore[arg-type]

        # драйверы — для горизонта с максимальным риском
        worst = max(range(len(preds)), key=lambda i: preds[i].p_violation)
        drivers = self._drivers(bundles[worst], rows[bundles[worst].horizon_points])

        abstain, reason = False, None
        if pak_status != "ok" and pak_age is not None and pak_age > self.max_pak_age_min:
            abstain = True
            reason = (f"Последнее достоверное измерение {self.cfg.target_tag} устарело на "
                      f"{pak_age:.0f} мин — надёжной рекомендации нет")

        return AgentReport(
            agent=self.name, version=self.version, trace_id=state.trace_id, timestamp=state.timestamp,
            predictions=preds,
            data_quality=DataQualityInfo(pak_status=pak_status, pak_age_min=pak_age,  # type: ignore[arg-type]
                                         missing_tags=missing, history_points=n_hist),
            confidence=self._confidence(pak_age, mode, len(missing), len(base_tags)),
            drivers=drivers, abstain=abstain, reason=reason, notes=notes)

    # ------------------------------------------------------------------ helpers
    def _abstain(self, state: ProcessState, pak_status: str, pak_age: float | None,
                 missing: list[str], n_hist: int, reason: str) -> AgentReport:
        logger.warning("abstain: %s", reason)
        return AgentReport(
            agent=self.name, version=self.version, trace_id=state.trace_id, timestamp=state.timestamp,
            predictions=[],
            data_quality=DataQualityInfo(pak_status=pak_status, pak_age_min=pak_age,  # type: ignore[arg-type]
                                         missing_tags=missing, history_points=n_hist),
            confidence=0.0, drivers=[], abstain=True, reason=reason)

    def _confidence(self, pak_age: float | None, mode: str, n_missing: int, n_base: int) -> float:
        """Эвристика доверия (не вероятность): база по режиму * затухание по возрасту ПАК * полнота тегов."""
        base = 0.9 if mode == "with_analyzer" else 0.6
        decay = 1.0 if pak_age is None else float(np.clip(1 - pak_age / self.max_pak_age_min, 0.0, 1.0))
        completeness = 1.0 - n_missing / max(n_base, 1)
        return round(base * (0.5 + 0.5 * decay) * completeness, 3)

    @staticmethod
    def _drivers(bundle: ModeBundle, row: pd.DataFrame, top_n: int = 5) -> list[Driver]:
        try:
            contrib = bundle.model.explain(row, top_n)
        except Exception as e:  # noqa: BLE001 — drivers опциональны, не валим прогноз
            logger.warning("SHAP не посчитан: %s", e)
            return []
        return [Driver(tag=tag, shap=float(v), value=_num(row[tag].iloc[0])) for tag, v in contrib.items()]