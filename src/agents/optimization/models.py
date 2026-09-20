from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.schemas.agents_schemas import RiskClass


class BoundSource(str, Enum):
    historical = "historical"
    configured = "configured"
    none = "none"


class ControlVariable(BaseModel):
    tag: str
    description: str
    unit: str

    current_value: float

    lower_bound: float | None = None
    upper_bound: float | None = None
    bound_source: BoundSource = BoundSource.none


class ControlChange(BaseModel):
    tag: str
    description: str
    unit: str

    current_value: float
    proposed_value: float
    delta: float

    lower_bound: float | None = None
    upper_bound: float | None = None


class PredictedQuality(BaseModel):
    values: dict[str, float] = Field(default_factory=dict)
    risk_exceed_spec: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0


class ScenarioEvaluation(BaseModel):
    quality: PredictedQuality

    reliability_score: float
    reliability_class: RiskClass

    throughput_score: float = 0.0
    energy_score: float = 0.0

    objective_value: float

    feasible: bool
    violations: list[str] = Field(default_factory=list)


class OptimizationScenario(BaseModel):
    scenario_id: str

    changes: list[ControlChange] = Field(default_factory=list)

    evaluation: ScenarioEvaluation | None = None


class OptimizationResult(BaseModel):
    timestamp: datetime

    status: str

    selected_scenario: OptimizationScenario | None = None

    alternatives: list[OptimizationScenario] = Field(
        default_factory=list
    )

    reason: str | None = None

    warnings: list[str] = Field(default_factory=list)