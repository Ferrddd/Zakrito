
from __future__ import annotations

from src.agents.optimization.scenarios import Scenario


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """a доминирует b, если не хуже по всем целям и строго лучше хотя бы по одной (все цели — минимизация)."""
    not_worse = all(a[k] <= b[k] for k in a)
    strictly_better = any(a[k] < b[k] for k in a)
    return not_worse and strictly_better


def pareto_front(scenarios: list[Scenario]) -> list[Scenario]:
    """Недоминируемое множество среди ДОПУСТИМЫХ. Жёсткие ограничения отфильтрованы ДО фронта:
    качество/безопасность нельзя компенсировать другими метриками (ТЗ п.1)."""
    feasible = [s for s in scenarios if s.feasible]
    return [s for s in feasible
            if not any(dominates(o.objectives, s.objectives) for o in feasible if o is not s)]


def weighted_score(s: Scenario, weights: dict[str, float]) -> float:
    """Взвешенная сумма целей (меньше = лучше). Веса throughput/energy маппятся на throughput_loss/energy_proxy."""
    alias = {"throughput_loss": "throughput", "energy_proxy": "energy"}
    return sum(weights.get(k, weights.get(alias.get(k, ""), 0.0)) * v for k, v in s.objectives.items())


def pick_recommendation(front: list[Scenario], weights: dict[str, float]) -> Scenario | None:
    if not front:
        return None
    return min(front, key=lambda s: weighted_score(s, weights))