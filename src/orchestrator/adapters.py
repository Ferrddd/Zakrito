"""Переходники между контрактом агента качества (AgentReport) и общими схемами команды
(agents_schemas.py: ProcessSnapshot / QualityAssessment / QualitySignal)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.schemas.agents_schemas import (
    DataFreshness,
    DataSourceStatus,
    ProcessSnapshot,
    QualityAssessment,
    QualitySignal,
    QualitySource,
)
from src.schemas.process_state import AgentReport, ProcessState, QualityPrediction

SULFUR_KEY = "sulfur_ppm"


def state_from_row(ts: Any, row: pd.Series, trace_id: str | None = None) -> ProcessState:
    """Строка витрины -> ProcessState (нечисловое и NaN отбрасываются в None)."""
    tags: dict[str, float | None] = {}
    for k, v in row.items():
        if isinstance(v, (int, float, bool)) or hasattr(v, "dtype"):
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            tags[str(k)] = None if math.isnan(f) else f
    kwargs = {"trace_id": trace_id} if trace_id else {}
    return ProcessState(timestamp=pd.Timestamp(ts).to_pydatetime(), tags=tags, **kwargs)


def to_snapshot(state: ProcessState, report: AgentReport) -> ProcessSnapshot:
    dq = report.data_quality
    pac_status = {"ok": DataSourceStatus.ok, "missing": DataSourceStatus.down}.get(
        dq.pak_status, DataSourceStatus.stale)
    return ProcessSnapshot(
        timestamp=state.timestamp,
        telemetry={k: v for k, v in state.tags.items() if v is not None},
        data_freshness=DataFreshness(
            lims_age_hours=None,  # агент качества ЛИМС не использует (таргет — ПАК)
            pac_age_hours=None if dq.pak_age_min is None else dq.pak_age_min / 60,
            pac_status=pac_status))


def worst_prediction(report: AgentReport) -> QualityPrediction | None:
    return max(report.predictions, key=lambda p: p.p_violation) if report.predictions else None


def to_quality_assessment(report: AgentReport) -> QualityAssessment:
    forecast: dict[str, float] = {}
    risk: dict[str, float] = {}
    w = worst_prediction(report)
    if w is not None:
        forecast[SULFUR_KEY] = w.p50
        risk[SULFUR_KEY] = w.p_violation
    for p in report.predictions:
        forecast[f"{SULFUR_KEY}_p50_h{p.horizon_min}"] = p.p50
        forecast[f"{SULFUR_KEY}_p90_h{p.horizon_min}"] = p.p90
        risk[f"{SULFUR_KEY}_h{p.horizon_min}"] = p.p_violation
    dq = report.data_quality
    return QualityAssessment(
        timestamp=report.timestamp, quality_forecast=forecast, risk_exceed_spec=risk,
        confidence=report.confidence,
        target_source_used=QualitySource.pac if dq.pak_status == "ok" else QualitySource.none,
        source_age_hours=None if dq.pak_age_min is None else dq.pak_age_min / 60,
        stale_data_warning=dq.pak_status != "ok")


def to_quality_signal(a: QualityAssessment) -> QualitySignal:
    return QualitySignal(quality_forecast=a.quality_forecast, risk_exceed_spec=a.risk_exceed_spec,
                         confidence=a.confidence)