"""Оркестратор с фейковым агентом качества и тестовым оптимизатором."""

from __future__ import annotations

from datetime import UTC, datetime

from src.orchestrator.contracts import Scenario, Status
from src.orchestrator.orchestrator import Orchestrator
from src.schemas.agents_schemas import OptimizationInput
from src.schemas.process_state import (
    AgentReport,
    Capabilities,
    DataQualityInfo,
    ProcessState,
    QualityPrediction,
)


def pred(p90: float, alert: bool) -> QualityPrediction:
    return QualityPrediction(indicator="S", unit="ppm", horizon_min=60, p10=1, p50=p90 - 1, p90=p90, limit=10.0,
                             p_violation=0.9 if alert else 0.1, alert_threshold=0.5, alert=alert,
                             p90_over_limit=p90 > 10.0, source_model="with_analyzer")


class FakeQuality:
    name, version = "quality", "fake"

    def capabilities(self) -> Capabilities:
        return Capabilities(name="quality", version="fake", indicators=["S"], horizons_min=[60], required_tags=[])

    def evaluate(self, state: ProcessState, overrides: dict[str, float] | None = None) -> AgentReport:
        risky = state.tags.get("risk") == 1.0 and not (overrides and overrides.get("risk") == 0.0)
        stale = state.tags.get("stale") == 1.0
        return AgentReport(
            agent="quality", version="fake", trace_id=state.trace_id, timestamp=state.timestamp,
            predictions=[] if stale else [pred(12.0, True) if risky else pred(6.0, False)],
            data_quality=DataQualityInfo(pak_status="ok"), confidence=0.8,
            abstain=stale, reason="ПАК устарел" if stale else None)


class Opt:
    name, is_stub = "optimizer", False

    def __init__(self, scenarios: list[Scenario]):
        self.s = scenarios

    def propose(self, inp: OptimizationInput) -> list[Scenario]:
        return self.s


def st(**tags: float) -> ProcessState:
    return ProcessState(timestamp=datetime(2026, 6, 1, tzinfo=UTC), tags=tags)


def test_stable_no_action() -> None:
    rec = Orchestrator(FakeQuality(), audit_path=None).run_cycle(st(risk=0.0))
    assert rec.status == Status.stable and not rec.action
    assert set(rec.stubbed_agents) == {"reliability", "optimizer"}


def test_abstain_gives_no_recommendation() -> None:
    rec = Orchestrator(FakeQuality(), audit_path=None).run_cycle(st(stale=1.0))
    assert rec.status == Status.no_recommendation


def test_risk_without_optimizer() -> None:
    rec = Orchestrator(FakeQuality(), audit_path=None).run_cycle(st(risk=1.0))
    assert rec.status == Status.risk_no_optimizer


def test_only_safe_scenario_is_chosen_even_if_worse_score() -> None:
    bad = Scenario(name="дёшево, но опасно", changes={"risk": 1.0}, score=100)
    good = Scenario(name="безопасно", changes={"risk": 0.0}, score=1)
    rec = Orchestrator(FakeQuality(), optimizer=Opt([bad, good]), audit_path=None).run_cycle(st(risk=1.0))
    assert rec.status == Status.action and rec.action[0]["tag"] == "risk" and rec.rejected


def test_all_scenarios_unsafe_refuses() -> None:
    bad = Scenario(name="опасно", changes={"risk": 1.0}, score=5)
    rec = Orchestrator(FakeQuality(), optimizer=Opt([bad]), audit_path=None).run_cycle(st(risk=1.0))
    assert rec.status == Status.no_recommendation


def test_blend_must_sum_to_100() -> None:
    sc = Scenario(name="блендинг", changes={"risk": 0.0}, blend_fractions={"a": 60, "b": 30})
    rec = Orchestrator(FakeQuality(), optimizer=Opt([sc]), audit_path=None).run_cycle(st(risk=1.0))
    assert rec.status == Status.no_recommendation