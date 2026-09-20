"""Накопительная статистика работы оркестратора (блок agent_statistics для UI)."""

from __future__ import annotations

from collections import Counter
from typing import Any


class CycleStats:
    def __init__(self) -> None:
        self.total = 0
        self.ok = 0
        self.elapsed_s = 0.0
        self.calls: Counter[str] = Counter()

    def record(self, ok: bool, elapsed_s: float, calls: dict[str, int]) -> None:
        self.total += 1
        self.ok += int(ok)
        self.elapsed_s += elapsed_s
        self.calls.update(calls)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_sessions": self.total,
            # успех = цикл дошёл до рекомендации/осознанного отказа без исключения
            "success_rate": round(100 * self.ok / self.total, 1) if self.total else 0.0,
            "avg_response_time": round(self.elapsed_s / self.total, 3) if self.total else 0.0,
            "agent_calls": [{"agent": k, "calls": v} for k, v in self.calls.items()],
        }