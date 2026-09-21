"""Оценка сценария: прогноз качества (ВАК), жёсткие ограничения, целевые функции
(логика исходного optim_agent.evaluate_scenario)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agents.optimization.registry import CONTROL_REGISTRY, ControlVar
from src.agents.optimization.scenarios import OptimizerInput, Scenario
from src.agents.optimization.vak import vak_predict_state

ConstraintRow = tuple[str, bool, str]   # (имя, выполнено, детали)


def _in_band(new: float, cur: float, ref: float, bound_pct: float) -> bool:
    lo, hi = ref * (1 - bound_pct), ref * (1 + bound_pct)
    lo, hi = min(lo, hi), max(lo, hi)
    if lo <= new <= hi:
        return True
    # текущее значение уже вне диапазона — разрешаем только движение к диапазону
    clamp = lambda v: min(max(v, lo), hi)  # noqa: E731
    return abs(new - clamp(new)) < abs(cur - clamp(cur))


def constraint_report(scn: Scenario, inp: OptimizerInput, predicted: Mapping[str, float],
                      registry: Mapping[str, ControlVar] = CONTROL_REGISTRY,
                      reference: Mapping[str, float] | None = None) -> list[ConstraintRow]:
    """Жёсткие ограничения БЕЗ серы: модельные диапазоны, прочая спецификация, ограничения надёжности."""
    reference = reference or {}
    rows: list[ConstraintRow] = []
    state, spec = inp.current_state, inp.spec

    # модельный диапазон управляемого параметра (правило границ ТЗ п.4 — ДОПУЩЕНИЕ, не регламент)
    for tag, new in scn.deltas.items():
        cur = state.get(tag)
        cv = registry.get(tag)
        if cur is None or cv is None or new == cur:
            continue
        ref = reference.get(tag, cur)
        src = "медиана истории" if tag in reference else "текущее значение"
        ok = _in_band(new, cur, ref, cv.bound_pct)
        rows.append((f"{tag}: модельный диапазон ±{cv.bound_pct:.0%} от опорного", ok,
                     f"{new:.4g} при опоре {ref:.4g} ({src}; допущение, не регламент)"))

    # прочие показатели качества (только если спецификация задана)
    for name, limit in (("CFPP", spec.cfpp_max_c), ("CloudPoint", spec.cloud_point_max_c), ("T90", spec.t90_max_c)):
        if limit is not None and name in predicted:
            rows.append((f"ВАК {name} <= {limit:g}", predicted[name] <= limit, f"{predicted[name]:.1f}"))

    # жёсткие ограничения от агента надёжности
    hard: dict[str, Any] = inp.reliability_assessment.get("hard_constraints", {}) or {}
    for tag, limit in hard.items():
        val = scn.deltas.get(tag)
        if val is None:
            continue
        if isinstance(limit, tuple):
            lo, hi = limit
            rows.append((f"{tag}: диапазон надёжности {limit}", lo <= val <= hi, f"{val:.4g}"))
        else:
            rows.append((f"{tag}: предел надёжности {limit}", val <= limit, f"{val:.4g}"))
    return rows


def evaluate_scenario(scn: Scenario, inp: OptimizerInput,
                      registry: Mapping[str, ControlVar] = CONTROL_REGISTRY,
                      reference: Mapping[str, float] | None = None,
                      check_sulfur: bool = True) -> Scenario:
    state = dict(inp.current_state)
    state.update(scn.deltas)

    # прогноз качества по ВАК (где хватает тегов)
    scn.predicted_quality = vak_predict_state(state)

    violations = [f"{n}: {d}" for n, ok, d in constraint_report(scn, inp, scn.predicted_quality, registry, reference)
                  if not ok]

    # сера: жёсткое ограничение. Источник — what-if агента качества ДЛЯ ЭТОГО сценария (scn.probe);
    # без what-if (исходное поведение) — общий прогноз агента качества, одинаковый для всех сценариев.
    sulfur_pred: float | None
    if scn.probe is not None:
        sulfur_pred = scn.probe.p50
        if check_sulfur and not scn.probe.ok:
            violations.append(f"сера: p50={scn.probe.p50:.2f}, p90={scn.probe.p90:.2f} "
                              f"(лимит {scn.probe.limit:g}, скор нарушения {scn.probe.p_violation:.2f})")
    else:
        sulfur_pred = inp.quality_forecast.get("sulfur_ppm", {}).get("pred")
        if check_sulfur and sulfur_pred is not None and sulfur_pred > inp.spec.sulfur_max_ppm:
            violations.append(f"sulfur_ppm={sulfur_pred:.2f} > {inp.spec.sulfur_max_ppm}")

    scn.violations = violations
    scn.feasible = not violations

    # целевые функции (все — МИНИМИЗАЦИЯ)
    quality_risk = 0.0
    if sulfur_pred is not None:
        quality_risk += max(0.0, sulfur_pred / inp.spec.sulfur_max_ppm - 0.7)   # риск растёт после 70% от предела
    if inp.spec.t90_max_c and "T90" in scn.predicted_quality:
        quality_risk += max(0.0, scn.predicted_quality["T90"] / inp.spec.t90_max_c - 0.9)

    base_f65 = inp.current_state.get("F65", 0.0)
    throughput_loss = max(0.0, base_f65 - scn.deltas.get("F65", base_f65))     # срезали загрузку — хуже

    energy_proxy = 0.0                                                          # прокси: относит. сдвиг уставок
    for tag in ("T33", "T16", "T18"):
        if tag in scn.deltas and tag in inp.current_state:
            base = inp.current_state[tag]
            if base:
                energy_proxy += abs(scn.deltas[tag] - base) / abs(base)

    reliability_risk = float(inp.reliability_assessment.get("risk_score", 0.0))

    scn.objectives = {"quality_risk": quality_risk, "throughput_loss": throughput_loss,
                      "energy_proxy": energy_proxy, "reliability_risk": reliability_risk}
    return scn