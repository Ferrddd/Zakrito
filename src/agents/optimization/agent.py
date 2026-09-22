

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.agents.optimization.evaluation import constraint_report, evaluate_scenario
from src.agents.optimization.pareto import (
    pareto_front,
    pick_recommendation,
    weighted_score,
)
from src.agents.optimization.registry import (
    CONTROL_REGISTRY,
    ControlVar,
    resolve_columns,
)
from src.agents.optimization.scenarios import (
    OptimizerInput,
    Scenario,
    changed_tags,
    generate_scenarios,
)
from src.agents.optimization.spec import Spec
from src.agents.optimization.vak import vak_inputs
from src.orchestrator.contracts import ConstraintCheck, Probe
from src.orchestrator.contracts import Scenario as PlanScenario
from src.schemas.agents_schemas import OptimizationInput, ReliabilityAssessment

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    all_scenarios: list[Scenario]
    feasible_scenarios: list[Scenario]
    pareto_front: list[Scenario]
    recommendation: Scenario | None
    message: str
    probes_used: int = 0


def run_optimization_agent(opt_input: OptimizerInput, controls: list[str] | None = None,
                           registry: Mapping[str, ControlVar] = CONTROL_REGISTRY,
                           reference: Mapping[str, float] | None = None,
                           probe: Probe | None = None, columns: Mapping[str, str] | None = None,
                           max_probes: int = 300) -> OptimizationResult:
    state = opt_input.current_state
    raw = generate_scenarios(opt_input, controls=controls, registry=registry)
    base = raw[0]
    evaluated = [evaluate_scenario(s, opt_input, registry, reference, check_sulfur=probe is None) for s in raw]
    used = 0

    if probe is not None and columns is not None:
        base.feasible = False
        base.violations.append("режим без изменений: риск по качеству уже зафиксирован агентом качества")

        def run_probe(s: Scenario) -> bool:
            nonlocal used
            used += 1
            s.probe = probe({columns[t]: s.deltas[t] for t in changed_tags(s, state)})
            if s.probe is None:
                s.feasible = False
                s.violations.append("what-if агента качества недоступен (отказ от прогноза)")
                return False
            evaluate_scenario(s, opt_input, registry, reference, check_sulfur=True)
            return True

        cheap_ok = [s for s in evaluated if s is not base and s.feasible]
        singles = [s for s in cheap_ok if len(changed_tags(s, state)) == 1]
        pairs = [s for s in cheap_ok if len(changed_tags(s, state)) > 1]
        single_p90: dict[tuple[str, float], float] = {}
        for s in singles:
            if used >= max_probes:
                break
            if run_probe(s) and s.probe is not None:
                t = changed_tags(s, state)[0]
                single_p90[(t, s.deltas[t])] = s.probe.p90
        # пары: сначала те, чьи составляющие по отдельности снижают p90 сильнее всего
        def pair_key(s: Scenario) -> float:
            return sum(single_p90.get((t, s.deltas[t]), float("inf")) for t in changed_tags(s, state))
        pairs.sort(key=pair_key)
        for s in pairs:
            if used >= max_probes:
                s.feasible = False
                s.violations.append("не проверен what-if агента качества (исчерпан бюджет проверок)")
                continue
            run_probe(s)
        for s in singles:
            if s.probe is None and s.feasible:
                s.feasible = False
                s.violations.append("не проверен what-if агента качества (исчерпан бюджет проверок)")

    feasible = [s for s in evaluated if s.feasible]
    front = pareto_front(evaluated)
    rec = pick_recommendation(front, opt_input.objective_weights)
    if rec is None:
        message = ("Надёжной рекомендации нет: ни один из рассмотренных вариантов не удовлетворяет жёстким "
                   "ограничениям (качество/надёжность). Требуется расширить пространство сценариев или "
                   "снизить нагрузку установки.")
    else:
        message = "Сформирован Парето-фронт допустимых сценариев, рекомендация выбрана по взвешенному критерию."
    return OptimizationResult(evaluated, feasible, front, rec, message, used)


class OptimizationAgent:
    """Адаптер к протоколу оркестратора (propose / bind_probe / check_constraints)."""
    name = "optimizer"
    is_stub = False

    def __init__(self, spec: Spec | None = None, registry: Mapping[str, ControlVar] | None = None,
                 reference: Mapping[str, float] | None = None,
                 objective_weights: dict[str, float] | None = None,
                 hard_constraints: dict[str, Any] | None = None,
                 max_probes: int = 300, top_k: int = 10):
        self.spec = spec or Spec()
        self.registry = dict(registry or CONTROL_REGISTRY)
        self.reference = dict(reference or {})
        self.weights = objective_weights or {"quality_risk": 0.4, "throughput": 0.2, "energy": 0.2,
                                             "reliability_risk": 0.2}
        self.hard_constraints = hard_constraints or {}
        self.max_probes = max_probes
        self.top_k = top_k
        self._probe: Probe | None = None
        self.last_result: OptimizationResult | None = None

    def bind_probe(self, probe: Probe | None) -> None:
        self._probe = probe

    # ---- состояние: колонки витрины -> ключи реестра/имена тегов формул ВАК -------------------------
    def _state(self, tags: Mapping[str, float | None]) -> tuple[dict[str, float], dict[str, str]]:
        values = {k: float(v) for k, v in tags.items() if v is not None}
        columns = resolve_columns(values, self.registry)
        state = vak_inputs(values)                       # контекст формул ВАК (теги 24-2000)
        for key, col in columns.items():
            state[key] = values[col]                     # управляемые (F14_avt отдельно от F14 из 24-2000)
        return state, columns

    def _input(self, state: dict[str, float], forecast: dict[str, float], confidence: float,
               risk_class: str, severity: float) -> OptimizerInput:
        return OptimizerInput(
            current_state=state,
            quality_forecast={"sulfur_ppm": {"pred": forecast.get("sulfur_ppm"), "confidence": confidence}},
            reliability_assessment={"risk_level": risk_class, "risk_score": severity,
                                    "hard_constraints": self.hard_constraints},
            spec=self.spec, objective_weights=self.weights)

    # ---- propose --------------------------------------------------------------------------------------
    def propose(self, inp: OptimizationInput) -> list[PlanScenario]:
        state, columns = self._state(inp.snapshot.telemetry)
        if not columns:
            logger.warning("Ни один управляющий тег не найден в состоянии — сценариев нет")
            return []
        opt_in = self._input(state, inp.quality.quality_forecast, inp.quality.confidence,
                             inp.reliability.risk_class.value, inp.reliability.severity_index)
        res = run_optimization_agent(opt_in, controls=list(columns), registry=self.registry,
                                     reference=self.reference, probe=self._probe, columns=columns,
                                     max_probes=self.max_probes)
        self.last_result = res
        logger.info("Оптимизатор: сценариев %d, допустимых %d, на фронте %d, what-if %d",
                    len(res.all_scenarios), len(res.feasible_scenarios), len(res.pareto_front), res.probes_used)

        front_ids = {id(s) for s in res.pareto_front}
        feasible = sorted(res.feasible_scenarios,
                          key=lambda s: (id(s) not in front_ids, weighted_score(s, self.weights)))
        if feasible:
            chosen = feasible[:self.top_k]
        else:   # честный отказ: показать лучшие из проверенных, чтобы оркестратор объяснил причины
            probed = [s for s in res.all_scenarios if s.probe is not None]
            chosen = sorted(probed, key=lambda s: s.probe.p90 if s.probe else 0.0)[:3]
        return [self._to_plan(s, state, columns, id(s) in front_ids) for s in chosen]

    def _to_plan(self, s: Scenario, state: dict[str, float], columns: dict[str, str], on_front: bool) -> PlanScenario:
        tags = changed_tags(s, state)
        cv = self.registry
        name = "; ".join(f"{columns[t]} {state[t]:.4g}→{s.deltas[t]:.4g} {cv[t].unit}" for t in tags)
        metrics: dict[str, float] = {k: float(v) for k, v in s.objectives.items()}
        tp = "F65" if "F65" in columns else ("F26" if "F26" in columns else None)
        if tp is not None:
            metrics["throughput"] = float(s.deltas.get(tp, state[tp]))
            metrics["throughput_change_pct"] = 100.0 * (metrics["throughput"] - state[tp]) / (abs(state[tp]) or 1.0)
        metrics.update({f"vak_{k}": float(v) for k, v in s.predicted_quality.items()})
        if s.probe is not None:
            metrics["sulfur_p50_pred"], metrics["sulfur_p90_pred"] = s.probe.p50, s.probe.p90
        metrics["pareto"] = 1.0 if on_front else 0.0
        assumptions = ["границы управляющих тегов — модельные (± bound_pct от медианы истории), не регламент",
                       "энергия — прокси (относительный сдвиг уставок T33/T16/T18), фактических затрат нет",
                       "ВАК — line-fit модели; приоритет достоверности ЛИМС > ПАК > ВАК"]
        if on_front:
            assumptions.append("сценарий на Парето-фронте допустимых")
        return PlanScenario(name=name, changes={columns[t]: s.deltas[t] for t in tags}, metrics=metrics,
                            score=-weighted_score(s, self.weights), assumptions=assumptions)

    # ---- жёсткие проверки оптимизатора для оркестратора -------------------------------------------
    def check_constraints(self, state_tags: Mapping[str, float | None], scenario: PlanScenario,
                          reliability: ReliabilityAssessment | None) -> list[ConstraintCheck]:
        state, columns = self._state(state_tags)
        col_to_key = {c: k for k, c in columns.items()}
        deltas = {k: state[k] for k in columns if k in state}
        for col, val in scenario.changes.items():
            if col not in col_to_key:
                return [ConstraintCheck(name=f"{col}: управляемый тег", passed=False,
                                        detail="тег не входит в реестр управляющих воздействий")]
            deltas[col_to_key[col]] = val
        sev = reliability.severity_index if reliability is not None else 0.0
        risk = reliability.risk_class.value if reliability is not None else "normal"
        opt_in = self._input(state, {}, 1.0, risk, sev)
        scn = Scenario(deltas=deltas)
        from src.agents.optimization.vak import vak_predict_state
        pred = vak_predict_state({**state, **deltas})
        rows = constraint_report(scn, opt_in, pred, self.registry, self.reference)
        return [ConstraintCheck(name=n, passed=ok, detail=d) for n, ok, d in rows]