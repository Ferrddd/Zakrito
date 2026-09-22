

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

PakStatus = Literal["ok", "frozen", "out_of_range", "missing"]
ModelMode = Literal["with_analyzer", "blind"]


class ProcessState(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime
    tags: dict[str, float | None] = Field(
        ..., description="Текущие значения тегов (колонки витрины). None/NaN = нет данных")


class Capabilities(BaseModel):
    name: str
    version: str
    indicators: list[str]
    horizons_min: list[int]
    required_tags: list[str]


class Driver(BaseModel):
    tag: str
    shap: float = Field(..., description="Локальный вклад признака в прогноз p50 (SHAP, в единицах показателя)")
    value: float | None = Field(None, description="Значение признака в текущей точке")


class QualityPrediction(BaseModel):
    indicator: str
    unit: str
    horizon_min: int
    p10: float
    p50: float
    p90: float
    limit: float
    p_violation: float = Field(
        ..., description="Скор классификатора P(показатель > limit через horizon). "
                         "Не калиброван (scale_pos_weight) — сравнивать с alert_threshold")
    alert_threshold: float | None = None
    alert: bool = Field(False, description="p_violation >= alert_threshold")
    p90_over_limit: bool = Field(False, description="Верхний квантиль выше лимита (консервативный сигнал)")
    source_model: ModelMode


class DataQualityInfo(BaseModel):
    pak_status: PakStatus
    pak_age_min: float | None = None
    missing_tags: list[str] = Field(default_factory=list)
    history_points: int | None = Field(None, description="Сколько точек истории было в буфере агента")


class AgentReport(BaseModel):
    agent: str
    version: str
    trace_id: str
    timestamp: datetime
    predictions: list[QualityPrediction]
    data_quality: DataQualityInfo
    confidence: float
    drivers: list[Driver] = Field(default_factory=list)
    abstain: bool = False
    reason: str | None = None
    notes: list[str] = Field(default_factory=list)