"""Оркестратор v1.

Один цикл (ТЗ п.1):
    состояние -> агент качества -> [агент надёжности] -> риск? -> [агент оптимизации]
    -> проверка жёстких ограничений на каждом сценарии (what-if через агент качества)
    -> рекомендация / отказ

Реально работает: агент качества, решение «стабильно / риск / отказ», проверка
сценариев, аудит-лог. Заглушки (помечены is_stub и попадают в Recommendation.stubbed_agents):
агент надёжности и агент оптимизации — подключаются через конструктор без правок кода.

Принцип ТЗ «качество важнее экономики» реализован тут структурно: оптимизатор может
только предлагать, а сценарий, не прошедший жёсткие проверки, физически не попадает
в выбор — score его не спасает.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agents.base import Agent
from src.orchestrator.adapters import (
    to_quality_assessment,
    to_quality_signal,
    to_snapshot,
    worst_prediction,
)
from src.orchestrator.contracts import (
    CheckedScenario,
    ConstraintCheck,
    NotConnectedOptimizer,
    NotConnectedReliability,
    OptimizationAgent,
    Recommendation,
    ReliabilityAgent,
    Scenario,
    Status,
)
from src.schemas.agents_schemas import (
    OptimizationInput,
    ReliabilityAssessment,
    ReliabilitySignal,
    RiskClass,
)
from src.schemas.process_state import AgentReport, ProcessState

logger = logging.getLogger(__name__)


def has_quality_risk(report: AgentReport) -> bool:
    """Консервативно: тревога классификатора ИЛИ верхний квантиль выше лимита."""
    return any(p.alert or p.p90_over_limit for p in report.predictions)


class Orchestrator:
    def __init__(self, quality: Agent, reliability: ReliabilityAgent | None = None,
                 optimizer: OptimizationAgent | None = None,
                 audit_path: str | Path | None = "data/converted/audit/cycles.jsonl",
                 max_alternatives: int = 3):
        self.quality = quality
        self.reliability = reliability or NotConnectedReliability()
        self.optimizer = optimizer or NotConnectedOptimizer()
        self.audit_path = Path(audit_path) if audit_path else None
        self.max_alternatives = max_alternatives

    # ------------------------------------------------------------------ цикл
    def run_cycle(self, state: ProcessState) -> Recommendation:
        # 1-3. качество
        q_report = self.quality.evaluate(state)
        q_assess = to_quality_assessment(q_report)
        snapshot = to_snapshot(state, q_report)

        # 4. надёжность
        rel = self.reliability.assess(snapshot)

        stubs = [a.name for a in (self.reliability, self.optimizer) if getattr(a, "is_stub", False)]
        scenarios: list[CheckedScenario] = []

        # 5-8. решение
        if q_report.abstain:
            rec = self._no_recommendation(state, q_report, q_assess,
                                          q_report.reason or "Агент качества отказался от прогноза")
        elif rel.risk_class == RiskClass.critical:
            rec = self._escalate(state, q_report, q_assess, rel)
        elif not has_quality_risk(q_report):
            rec = self._stable(state, q_report, q_assess, rel)
        else:
            rec, scenarios = self._handle_risk(state, q_report, q_assess, rel, snapshot)

        rec.stubbed_agents = stubs
        if stubs:
            rec.warnings.append(f"Заглушки (не реальные агенты): {', '.join(stubs)}")
        self._audit(state, q_report, rel, scenarios, rec)
        return rec

    # ------------------------------------------------------------------ ветки решения
    def _handle_risk(self, state: ProcessState, q_report: AgentReport, q_assess: Any,
                     rel: ReliabilityAssessment, snapshot: Any) -> tuple[Recommendation, list[CheckedScenario]]:
        w = worst_prediction(q_report)
        problem = (f"Риск превышения серы {w.limit:g} {w.unit} через {w.horizon_min} мин: "
                   f"p50={w.p50:.1f}, p90={w.p90:.1f} {w.unit}, скор нарушения {w.p_violation:.2f} "
                   f"(порог {w.alert_threshold:.2f})")
        inp = OptimizationInput(
            snapshot=snapshot, quality=to_quality_signal(q_assess),
            reliability=ReliabilitySignal(severity_index=rel.severity_index, risk_class=rel.risk_class))
        proposed = self.optimizer.propose(inp)
        checked = [self._check(state, s) for s in proposed]
        ok = sorted((c for c in checked if c.passed), key=lambda c: c.scenario.score, reverse=True)
        bad = [c for c in checked if not c.passed]
        rejected = [f"{c.scenario.name}: " + "; ".join(k.detail or k.name for k in c.checks if not k.passed)
                    for c in bad]

        base = dict(trace_id=state.trace_id, timestamp=state.timestamp, problem=problem,
                    confidence=q_report.confidence, data_freshness=self._freshness(q_report),
                    rejected=rejected, warnings=self._warnings(q_report, rel))

        if ok:
            best = ok[0]
            cur = state.tags
            return Recommendation(
                status=Status.action, headline="Рекомендуется изменить режим",
                action=[{"tag": t, "current": cur.get(t), "recommended": v}
                        for t, v in best.scenario.changes.items()],
                expected_effect=best.predicted,
                constraints_checked=best.checks,
                alternatives=[c.scenario.name for c in ok[1:1 + self.max_alternatives]],
                explanation=(f"Вариант «{best.scenario.name}» прошёл все жёсткие проверки и имеет "
                             f"лучший score={best.scenario.score:.3g} среди {len(ok)} допустимых. "
                             + " ".join(best.scenario.assumptions)),
                **base), checked

        if bad:  # предложения были, но все нарушают ограничения — честный отказ
            return Recommendation(
                status=Status.no_recommendation,
                headline="Надёжной рекомендации нет: все варианты нарушают ограничения",
                explanation="Оптимизатор предложил варианты, но ни один не прошёл жёсткие проверки.",
                **base), checked

        return Recommendation(
            status=Status.risk_no_optimizer, headline="Обнаружен риск качества — действий не предложено",
            explanation=("Агент оптимизации не подключён или не нашёл вариантов. Оператору передан только "
                         "сигнал риска и главные драйверы прогноза. Ниже — драйверы: "
                         + ", ".join(f"{d.tag} ({d.shap:+.2f})" for d in q_report.drivers)),
            **base), checked

    def _stable(self, state: ProcessState, q_report: AgentReport, q_assess: Any,
                rel: ReliabilityAssessment) -> Recommendation:
        w = worst_prediction(q_report)
        return Recommendation(
            trace_id=state.trace_id, timestamp=state.timestamp, status=Status.stable,
            headline="Режим стабилен, изменений не требуется",
            explanation=(f"Прогноз серы p90={w.p90:.1f} {w.unit} при лимите {w.limit:g}; "
                         f"тревога не сработала — лишних управляющих действий не создаём."),
            confidence=q_report.confidence, data_freshness=self._freshness(q_report),
            constraints_checked=[ConstraintCheck(
                name=f"сера <= {w.limit:g} {w.unit} (прогноз)", passed=True,
                detail=f"p90={w.p90:.1f}, скор {w.p_violation:.2f}")],
            warnings=self._warnings(q_report, rel))

    def _no_recommendation(self, state: ProcessState, q_report: AgentReport, q_assess: Any,
                           reason: str) -> Recommendation:
        return Recommendation(
            trace_id=state.trace_id, timestamp=state.timestamp, status=Status.no_recommendation,
            headline="Надёжной рекомендации нет", problem=reason, confidence=q_report.confidence,
            explanation="Система отказывается от рискованной рекомендации при недостатке/устаревании данных.",
            data_freshness=self._freshness(q_report),
            warnings=[f"Не хватает тегов: {q_report.data_quality.missing_tags[:10]}"]
            if q_report.data_quality.missing_tags else [])

    def _escalate(self, state: ProcessState, q_report: AgentReport, q_assess: Any,
                  rel: ReliabilityAssessment) -> Recommendation:
        return Recommendation(
            trace_id=state.trace_id, timestamp=state.timestamp, status=Status.escalate,
            headline="Тяжёлый режим оборудования — требуется решение оператора",
            problem="; ".join(rel.limiting_factors) or f"severity_index={rel.severity_index:.2f}",
            confidence=q_report.confidence, data_freshness=self._freshness(q_report),
            explanation="Агент надёжности классифицировал режим как critical; автоматических рекомендаций нет.")

    # ------------------------------------------------------------------ жёсткие проверки
    def _check(self, state: ProcessState, sc: Scenario) -> CheckedScenario:
        report = self.quality.evaluate(state, overrides=sc.changes)
        checks: list[ConstraintCheck] = []
        if report.abstain or not report.predictions:
            checks.append(ConstraintCheck(name="прогноз качества для сценария", passed=False,
                                          detail=report.reason or "нет прогноза"))
        for p in report.predictions:
            ok = (not p.alert) and (not p.p90_over_limit)
            checks.append(ConstraintCheck(
                name=f"сера <= {p.limit:g} {p.unit} @ {p.horizon_min} мин", passed=ok,
                detail=f"p90={p.p90:.1f}, скор {p.p_violation:.2f}/{p.alert_threshold:.2f}"))
        if sc.blend_fractions is not None:
            total = sum(sc.blend_fractions.values())
            checks.append(ConstraintCheck(name="сумма долей блендинга = 100%", passed=abs(total - 100) < 0.01,
                                          detail=f"сумма={total:g}"))
        # TODO: явные технологические/модельные диапазоны управляющих тегов (п.4 ТЗ «правило границ»)
        return CheckedScenario(
            scenario=sc, checks=checks, passed=all(c.passed for c in checks),
            predicted=[{"horizon_min": p.horizon_min, "p50": p.p50, "p90": p.p90} for p in report.predictions])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _freshness(r: AgentReport) -> dict:
        dq = r.data_quality
        return {"pak_status": dq.pak_status, "pak_age_min": dq.pak_age_min, "lims": "не используется агентом качества"}

    @staticmethod
    def _warnings(r: AgentReport, rel: ReliabilityAssessment) -> list[str]:
        w = list(r.notes)
        if r.data_quality.pak_status != "ok":
            w.append(f"ПАК серы: {r.data_quality.pak_status} — прогноз в режиме blind, уверенность ниже")
        if r.confidence < 0.5:
            w.append(f"Низкая уверенность прогноза: {r.confidence:.2f}")
        return w

    def _audit(self, state: ProcessState, q_report: AgentReport, rel: ReliabilityAssessment,
               scenarios: list[CheckedScenario], rec: Recommendation) -> None:
        if not self.audit_path:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"trace_id": state.trace_id, "state": state.model_dump(mode="json"),
                  "quality": q_report.model_dump(mode="json"), "reliability": rel.model_dump(mode="json"),
                  "scenarios": [s.model_dump(mode="json") for s in scenarios],
                  "recommendation": rec.model_dump(mode="json")}
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")