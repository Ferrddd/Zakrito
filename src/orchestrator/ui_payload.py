"""Сборка JSON для UI из результатов цикла (формат — tests/fixtures/request_example.json)."""

from __future__ import annotations

import math
from collections import OrderedDict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from src.orchestrator.adapters import worst_prediction
from src.orchestrator.contracts import CheckedScenario, CycleContext, Status
from src.schemas.agents_schemas import RiskClass

DEFAULT_TARGET = "24-2000:Mg.Sulfur"
UNIT_UI = {"ppm": "мг/кг"}          # ppm по массе ≈ мг/кг
OUTDATED_AFTER_MIN = 60.0           # ПАК опрашивается раз в 10 мин; старше часа — считаем устаревшим
MAX_TELEMETRY = 8

# ДОПУЩЕНИЕ: в примере есть только "WARNING" — остальные значения согласовать с автором UI.
SYSTEM_STATUS = {
    Status.stable: "NORMAL",
    Status.action: "WARNING",
    Status.risk_no_optimizer: "WARNING",
    Status.no_recommendation: "WARNING",
    Status.escalate: "CRITICAL",
}

TagDirectory = dict[str, tuple[str, str]]   # колонка -> (человекочитаемое имя, единица)


def clean(obj: Any) -> Any:
    """Приводит к JSON: NaN/inf -> None, datetime -> ISO, Enum -> value, модели -> dict."""
    if isinstance(obj, BaseModel):
        return clean(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class TrendBuffer:
    """Ряд «факт vs прогноз» для графика: прогноз, выданный в t, кладётся на момент t + horizon,
    чтобы его можно было сравнить с фактом, когда тот появится."""

    def __init__(self, horizon_min: int = 60, max_points: int = 144):
        self.horizon_min = horizon_min
        self.max_points = max_points
        self.actual: OrderedDict[datetime, float | None] = OrderedDict()
        self.forecast: dict[datetime, float] = {}

    def update(self, ctx: CycleContext) -> None:
        ts = ctx.state.timestamp
        dq = ctx.quality.data_quality
        target = ctx.quality.predictions[0].indicator if ctx.quality.predictions else DEFAULT_TARGET
        value = ctx.state.tags.get(target) if dq.pak_status == "ok" else None
        self.actual[ts] = value
        if ctx.quality.predictions:
            p = min(ctx.quality.predictions, key=lambda x: abs(x.horizon_min - self.horizon_min))
            self.forecast[ts + timedelta(minutes=p.horizon_min)] = p.p50
        while len(self.actual) > self.max_points:
            old, _ = self.actual.popitem(last=False)
            self.forecast.pop(old + timedelta(minutes=self.horizon_min), None)

    def snapshot(self, iso: Any) -> dict[str, list[Any]]:
        stamps = sorted(set(self.actual) | set(self.forecast))
        return {"timestamps": [iso(t) for t in stamps],
                "sulfur_actual": [self.actual.get(t) for t in stamps],
                "sulfur_forecast": [self.forecast.get(t) for t in stamps]}


def _unit(u: str) -> str:
    return UNIT_UI.get(u, u)


def _ru(text: str) -> str:
    """Единицы для оператора: ppm -> мг/кг (в UI и ТЗ используется мг/кг)."""
    return text.replace("ppm", "мг/кг")


def _base(col: str) -> str:
    return col.split("__")[0]


def _failed_reasons(c: CheckedScenario) -> str:
    return "; ".join(k.detail or k.name for k in c.checks if not k.passed) or "не прошёл проверки"


def build_payload(ctx: CycleContext, trend: TrendBuffer, directory: TagDirectory | None = None,
                  tz: ZoneInfo | None = None) -> dict[str, Any]:
    directory = directory or {}
    rec, q, rel, state = ctx.recommendation, ctx.quality, ctx.reliability, ctx.state
    dq = q.data_quality
    w = worst_prediction(q)

    def iso(ts: datetime) -> str:
        if ts.tzinfo is None and tz is not None:
            ts = ts.replace(tzinfo=tz)
        return ts.isoformat()

    def meta_of(tag: str) -> tuple[str, str]:
        return directory.get(tag, (tag, ""))

    passed = sorted((c for c in ctx.scenarios if c.passed), key=lambda c: c.scenario.score, reverse=True)
    best = passed[0] if rec.status == Status.action and passed else None

    # ---- input_state -------------------------------------------------------
    tags_shown: list[str] = []
    for t in [a["tag"] for a in rec.action] + [_base(d.tag) for d in q.drivers]:
        if t not in tags_shown and state.tags.get(t) is not None:
            tags_shown.append(t)
    telemetry = [{"tag": t, "name": meta_of(t)[0], "value": state.tags[t], "unit": meta_of(t)[1]}
                 for t in tags_shown[:MAX_TELEMETRY]]

    target = q.predictions[0].indicator if q.predictions else DEFAULT_TARGET
    unit = _unit(q.predictions[0].unit) if q.predictions else "мг/кг"
    age = dq.pak_age_min
    outdated = dq.pak_status != "ok" or (age is not None and age > OUTDATED_AFTER_MIN)
    quality_sources = [{
        "source_type": "PAC", "parameter": "Сера",
        "value": state.tags.get(target) if dq.pak_status == "ok" else None,
        "unit": unit,
        "measured_at": iso(state.timestamp - timedelta(minutes=age)) if age is not None else None,
        "age_minutes": age, "is_outdated": outdated,
    }]
    warnings: list[str] = []
    if dq.pak_status != "ok":
        warnings.append(f"ПАК серы недостоверен: {dq.pak_status}")
    if age is not None and age > OUTDATED_AFTER_MIN:
        warnings.append(f"Анализ серы устарел: {age:.0f} мин")
    if dq.missing_tags:
        warnings.append(f"Нет данных по тегам: {', '.join(dq.missing_tags[:5])}")
    warnings += [x for x in rec.warnings if not x.startswith("Заглушки") and x not in warnings]

    # ---- orchestrator_decision --------------------------------------------
    actions = []
    for a in rec.action:
        cur, tgt = a.get("current"), a.get("recommended")
        direction = "hold" if cur is None or tgt is None or tgt == cur else ("up" if tgt > cur else "down")
        actions.append({"tag": a["tag"], "name": meta_of(a["tag"])[0], "current_value": cur,
                        "target_value": tgt, "direction": direction, "unit": meta_of(a["tag"])[1]})

    if rec.expected_effect:
        e = max(rec.expected_effect, key=lambda x: x["p90"])
        quality_effect = (f"Прогноз серы после изменения: p50={e['p50']:.1f}, p90={e['p90']:.1f} {unit} "
                          f"через {e['horizon_min']} мин")
    elif w is not None:
        tail = "нарушений не ожидается" if rec.status == Status.stable else "без изменений режима риск сохраняется"
        quality_effect = f"Прогноз серы: p50={w.p50:.1f}, p90={w.p90:.1f} {unit} через {w.horizon_min} мин — {tail}"
    else:
        quality_effect = "Прогноз недоступен"
    if best is not None and best.scenario.metrics:
        production = ", ".join(f"{k}={v:g}" for k, v in best.scenario.metrics.items())
    else:
        production = "Не оценивалось (агент оптимизации не подключён)" if "optimizer" in ctx.stubbed \
            else "Без изменений"
    equipment = f"Класс риска: {rel.risk_class.value}, severity_index={rel.severity_index:.2f}"
    if "reliability" in ctx.stubbed:
        equipment += " (агент надёжности — заглушка, оценка не выполнялась)"

    decision = {
        "problem_detected": _ru(rec.problem or rec.headline),
        "recommendation": actions,
        "expected_effects": {"quality": quality_effect, "production": production, "equipment_risk": equipment},
        "constraints_checked": [{"constraint": _ru(c.name), "status": "pass" if c.passed else "fail",
                                 "priority": "high"} for c in rec.constraints_checked],
        "explanation": _ru(rec.explanation or rec.headline),
        "overall_confidence": rec.confidence,
    }

    # ---- alternatives ------------------------------------------------------
    alternatives = [{
        "scenario_id": f"alt_{i}", "description": c.scenario.name,
        "rejected_reason": (_ru(_failed_reasons(c)) if not c.passed else "Допустим, но score ниже выбранного"),
    } for i, c in enumerate([c for c in ctx.scenarios if c is not best], start=1)]

    # ---- agents_trace ------------------------------------------------------
    def status_of(name: str, called: bool = True) -> str:
        if not called:
            return "skipped"
        return "stub" if name in ctx.stubbed else "done"

    nodes = [
        {"id": "quality_agent", "status": "abstain" if q.abstain else "done", "confidence": q.confidence},
        {"id": "reliability_agent", "status": status_of("reliability"),
         "confidence": 0.0 if "reliability" in ctx.stubbed else None},
        {"id": "optimization_agent", "status": status_of("optimizer", ctx.optimizer_called),
         "confidence": 0.0 if "optimizer" in ctx.stubbed or not ctx.optimizer_called else None},
        {"id": "orchestrator", "status": "done", "confidence": rec.confidence},
    ]
    sink = "optimization_agent" if ctx.optimizer_called else "orchestrator"
    forecast_tag = f"forecast_sulfur_{w.p50:.1f}" if w is not None else "forecast_none"
    edges = [
        {"from": "quality_agent", "to": sink, "data_passed": [forecast_tag]},
        {"from": "reliability_agent", "to": sink, "data_passed": [f"risk_class_{rel.risk_class.value}"]},
    ]
    raw = {
        "quality_agent": {
            "forecast": {"sulfur": w.p50 if w else None},
            "risk_of_violation": bool(w and (w.alert or w.p90_over_limit)),
            "p90": w.p90 if w else None, "p_violation": w.p_violation if w else None,
            "horizon_min": w.horizon_min if w else None, "mode": w.source_model if w else None,
            "abstain": q.abstain, "reason": q.reason,
        },
        "reliability_agent": {"risk_index": rel.severity_index,
                              "mode_permissible": rel.risk_class != RiskClass.critical},
        "optimization_agent": {"generated_scenarios": len(ctx.scenarios), "valid_scenarios": len(passed)},
    }

    # ---- visualization_data ------------------------------------------------
    pareto = []
    for c in ctx.scenarios:
        x = c.scenario.metrics.get("throughput")
        if x is None or not c.predicted:
            continue   # без выпуска или прогноза точку на фронте не построить
        y = max(c.predicted, key=lambda p: p["p90"])
        pareto.append({"x_production": x, "y_sulfur": y["p50"], "is_selected": c is best})

    trend.update(ctx)
    payload = {
        "meta": {"timestamp": iso(state.timestamp), "cycle_id": state.trace_id,
                 "has_safe_solution": rec.status in (Status.stable, Status.action),
                 "system_status": SYSTEM_STATUS[rec.status]},
        "input_state": {"telemetry_summary": telemetry, "quality_sources": quality_sources,
                        "data_quality_warnings": warnings},
        "orchestrator_decision": decision,
        "alternatives": alternatives,
        "agents_trace": {"nodes": nodes, "edges": edges, "raw_evaluations": raw},
        "visualization_data": {"pareto_front": pareto, "trend_charts": trend.snapshot(iso)},
        "agent_statistics": ctx.stats,
    }
    return clean(payload)  # type: ignore[no-any-return]