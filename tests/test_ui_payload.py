"""Формат JSON для UI: структура совпадает с tests/fixtures/request_example.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.orchestrator.contracts import CycleContext, Scenario
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.ui_payload import TrendBuffer, build_payload
from tests.test_orchestrator import FakeQuality, Opt, st

EXAMPLE = json.loads((Path(__file__).parent / "fixtures" / "request_example.json").read_text(encoding="utf-8"))


def assert_superset(ours: Any, example: Any, path: str = "$") -> None:
    """У нас есть все ключи примера на каждом уровне (лишние ключи допустимы)."""
    if isinstance(example, dict):
        assert isinstance(ours, dict), f"{path}: ожидался объект"
        missing = set(example) - set(ours)
        assert not missing, f"{path}: нет ключей {sorted(missing)}"
        for k in example:
            assert_superset(ours[k], example[k], f"{path}.{k}")
    elif isinstance(example, list) and example and isinstance(example[0], (dict, list)):
        assert isinstance(ours, list) and ours, f"{path}: ожидался непустой список"
        assert_superset(ours[0], example[0], f"{path}[0]")


def run(state_tags: dict[str, float], scenarios: list[Scenario]) -> tuple[CycleContext, dict[str, Any]]:
    captured: list[CycleContext] = []
    orch = Orchestrator(FakeQuality(), optimizer=Opt(scenarios), audit_path=None, publisher=captured.append)
    orch.run_cycle(st(**state_tags))
    ctx = captured[0]
    return ctx, build_payload(ctx, TrendBuffer())


def test_payload_matches_ui_example_structure() -> None:
    good = Scenario(name="снизить расход", changes={"risk": 0.0}, score=1, metrics={"throughput": 145.0})
    bad = Scenario(name="опасный", changes={"risk": 1.0}, score=9, metrics={"throughput": 150.0})
    _, payload = run({"risk": 1.0}, [good, bad])

    assert_superset(payload, EXAMPLE)
    json.dumps(payload, allow_nan=False)                      # строгий JSON: никаких NaN
    assert payload["meta"]["system_status"] == "WARNING" and payload["meta"]["has_safe_solution"] is True
    rec = payload["orchestrator_decision"]["recommendation"][0]
    assert (rec["tag"], rec["current_value"], rec["target_value"], rec["direction"]) == ("risk", 1.0, 0.0, "down")
    assert payload["alternatives"][0]["rejected_reason"]
    # на Pareto попадают и отвергнутые варианты (как в примере: 150 т/ч -> 10.2, is_selected=false)
    assert [p["is_selected"] for p in payload["visualization_data"]["pareto_front"]] == [True, False]


def test_stable_cycle_payload() -> None:
    _, payload = run({"risk": 0.0}, [])
    assert payload["meta"]["system_status"] == "NORMAL" and payload["meta"]["has_safe_solution"] is True
    assert payload["orchestrator_decision"]["recommendation"] == []
    nodes = {n["id"]: n["status"] for n in payload["agents_trace"]["nodes"]}
    assert nodes["optimization_agent"] == "skipped" and nodes["reliability_agent"] == "stub"


def test_abstain_payload_is_not_safe() -> None:
    _, payload = run({"stale": 1.0}, [])
    assert payload["meta"]["has_safe_solution"] is False
    assert payload["agents_trace"]["nodes"][0]["status"] == "abstain"
    json.dumps(payload, allow_nan=False)


def test_statistics_accumulate() -> None:
    orch = Orchestrator(FakeQuality(), audit_path=None)
    for _ in range(3):
        orch.run_cycle(st(risk=0.0))
    s = orch.stats.snapshot()
    assert s["total_sessions"] == 3 and s["success_rate"] == 100.0
    assert {c["agent"]: c["calls"] for c in s["agent_calls"]}["orchestrator"] == 3


def test_ui_failure_does_not_break_cycle() -> None:
    def boom(ctx: CycleContext) -> None:
        raise RuntimeError("ui down")

    rec = Orchestrator(FakeQuality(), audit_path=None, publisher=boom).run_cycle(st(risk=0.0))
    assert rec.status.value == "stable"