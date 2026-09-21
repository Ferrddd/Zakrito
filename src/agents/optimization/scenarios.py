"""Сценарий, вход оптимизатора и генерация сетки сценариев (логика исходного optim_agent)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from src.agents.optimization.registry import CONTROL_REGISTRY, ControlVar
from src.agents.optimization.spec import Spec
from src.orchestrator.contracts import ProbeResult


@dataclass
class Scenario:
    deltas: dict[str, float]                # {ключ реестра: новое значение управляемого параметра}
    predicted_quality: dict[str, float] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)   # цели для Парето (минимизация)
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    probe: ProbeResult | None = None        # what-if агента качества (серa для ЭТОГО сценария)


@dataclass
class OptimizerInput:
    current_state: dict[str, float]                     # значения управляемых + контекстных тегов
    quality_forecast: dict[str, dict[str, Any]]         # {"sulfur_ppm": {"pred":.., "confidence":..}}
    reliability_assessment: dict[str, Any]              # {"risk_level":.., "risk_score":.., "hard_constraints": {...}}
    spec: Spec
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {"quality_risk": 0.4, "throughput": 0.2, "energy": 0.2, "reliability_risk": 0.2})


def changed_tags(scn: Scenario, state: Mapping[str, float]) -> list[str]:
    return [t for t, v in scn.deltas.items() if v != state.get(t)]


def generate_scenarios(opt_input: OptimizerInput, controls: list[str] | None = None,
                       registry: Mapping[str, ControlVar] = CONTROL_REGISTRY) -> list[Scenario]:
    """Сетка сценариев: для каждого тега 3 точки в пределах допустимого шага delta_pct
    (это ограничение СКОРОСТИ изменения режима за цикл, а не всего диапазона bound_pct).
    Базовый сценарий «ничего не менять» + одиночные + парные комбинации, где сдвинуты оба тега
    (в исходнике пары с одним неизменным тегом дублировали одиночные сценарии)."""
    controls = controls or list(registry.keys())
    state = opt_input.current_state

    per_tag_options: dict[str, list[float]] = {}
    for tag in controls:
        cv = registry[tag]
        current = state.get(tag)
        if current is None:
            continue
        step = current * cv.delta_pct
        options = [current - step, current, current + step]
        if cv.is_hard_bounded and cv.hard_bounds:
            lo, hi = cv.hard_bounds
            options = [max(lo, min(hi, v)) for v in options]
        per_tag_options[tag] = sorted(set(options))

    scenarios: list[Scenario] = [Scenario(deltas={tag: state[tag] for tag in per_tag_options})]  # базовый

    for tag, options in per_tag_options.items():                                                  # одиночные
        for val in options:
            if val == state[tag]:
                continue
            d = {t: state[t] for t in per_tag_options}
            d[tag] = val
            scenarios.append(Scenario(deltas=d))

    tags = list(per_tag_options)                                                                  # парные
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            t1, t2 = tags[i], tags[j]
            for v1, v2 in product(per_tag_options[t1], per_tag_options[t2]):
                if v1 == state[t1] or v2 == state[t2]:
                    continue   # пара с неизменным тегом = одиночный сценарий, он уже есть
                d = {t: state[t] for t in per_tag_options}
                d[t1], d[t2] = v1, v2
                scenarios.append(Scenario(deltas=d))
    return scenarios