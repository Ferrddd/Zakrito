"""Контракты оркестратора: сценарии оптимизатора, итоговая рекомендация, заглушки агентов."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.schemas.agents_schemas import (
    OptimizationInput,
    ProcessSnapshot,
    ReliabilityAssessment,
    RiskClass,
)
from src.schemas.process_state import AgentReport, ProcessState


class Scenario(BaseModel):
    """Один вариант изменения режима от агента оптимизации."""
    name: str
    changes: dict[str, float] = Field(..., description="колонка витрины -> НОВОЕ абсолютное значение")
    metrics: dict[str, float] = Field(default_factory=dict, description="выпуск, энергопрокси, ...")
    score: float = Field(0.0, description="чем больше, тем лучше (после прохождения жёстких ограничений)")
    blend_fractions: dict[str, float] | None = Field(None, description="доли компонентов блендинга, % (сумма 100)")
    assumptions: list[str] = Field(default_factory=list)


class ConstraintCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class CheckedScenario(BaseModel):
    scenario: Scenario
    checks: list[ConstraintCheck]
    passed: bool
    predicted: list[dict] = Field(default_factory=list, description="what-if прогноз качества по горизонтам")


class Status(str, Enum):
    stable = "stable"                        # риска нет, действий не требуется
    action = "action"                        # есть допустимый вариант
    risk_no_optimizer = "risk_no_optimizer"  # риск есть, оптимизатор не подключён/не дал вариантов
    no_recommendation = "no_recommendation"  # отказ: данные/ограничения
    escalate = "escalate"                    # тяжёлый режим — решает человек


class Recommendation(BaseModel):
    trace_id: str
    timestamp: datetime
    status: Status
    headline: str
    problem: str | None = None
    action: list[dict] = Field(default_factory=list, description="[{tag, current, recommended}]")
    expected_effect: list[dict] = Field(default_factory=list)
    constraints_checked: list[ConstraintCheck] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    data_freshness: dict = Field(default_factory=dict)
    alternatives: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    stubbed_agents: list[str] = Field(default_factory=list, description="Заглушки, участвовавшие в цикле")
    warnings: list[str] = Field(default_factory=list)


@dataclass
class CycleContext:
    """Всё, что произошло за один цикл — вход для публикации в UI и аудита."""
    state: ProcessState
    quality: AgentReport
    reliability: ReliabilityAssessment
    scenarios: list[CheckedScenario]
    recommendation: Recommendation
    optimizer_called: bool
    stubbed: list[str]
    agent_calls: dict[str, int]
    elapsed_s: float
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeResult:
    """Итог what-if прогноза агента качества для гипотетического режима (худший горизонт)."""
    p50: float
    p90: float
    p_violation: float
    ok: bool          # нет тревоги классификатора и p90 <= лимит на всех горизонтах
    limit: float


# what-if: {колонка: новое значение} -> прогноз качества (None — агент качества отказался)
Probe = Callable[[dict[str, float]], ProbeResult | None]


# ---- протоколы агентов + заглушки -------------------------------------------
class ReliabilityAgent(Protocol):
    name: str
    is_stub: bool

    def assess(self, snapshot: ProcessSnapshot) -> ReliabilityAssessment: ...


class OptimizationAgent(Protocol):
    name: str
    is_stub: bool

    def propose(self, inp: OptimizationInput) -> list[Scenario]: ...

    def bind_probe(self, probe: Probe | None) -> None:
        """Оркестратор даёт оптимизатору what-if через агента качества на время одного цикла."""
        ...

    def check_constraints(self, state_tags: Mapping[str, float | None], scenario: Scenario,
                          reliability: ReliabilityAssessment | None) -> list[ConstraintCheck]:
        """Жёсткие проверки сценария на стороне оптимизатора (диапазоны, шаг, ВАК, надёжность)."""
        ...


class NotConnectedReliability:
    """TODO: заменить реальным агентом надёжности (severity_index, dp_trend_slope, ...)."""
    name = "reliability"
    is_stub = True

    def assess(self, snapshot: ProcessSnapshot) -> ReliabilityAssessment:
        return ReliabilityAssessment(
            timestamp=snapshot.timestamp, severity_index=0.0, risk_class=RiskClass.normal,
            limiting_factors=["агент надёжности не подключён (заглушка) — оценка тяжести режима НЕ выполнялась"])


class NotConnectedOptimizer:
    """TODO: заменить реальным агентом оптимизации (генерация сценариев Δu)."""
    name = "optimizer"
    is_stub = True

    def propose(self, inp: OptimizationInput) -> list[Scenario]:
        return []

    def bind_probe(self, probe: Probe | None) -> None:
        return None

    def check_constraints(self, state_tags: Mapping[str, float | None], scenario: Scenario,
                          reliability: ReliabilityAssessment | None) -> list[ConstraintCheck]:
        return []