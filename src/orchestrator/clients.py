"""Клиент агента качества, поднятого как HTTP-сервис (src/agents/quality/api.py).
Реализует тот же протокол Agent — оркестратору всё равно, in-process агент или удалённый."""

from __future__ import annotations

import httpx

from src.schemas.process_state import AgentReport, Capabilities, ProcessState


class RemoteQualityAgent:
    name = "quality"
    version = "remote"

    def __init__(self, base_url: str = "http://localhost:8001", timeout_s: float = 30.0):
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def capabilities(self) -> Capabilities:
        r = self._http.get("/capabilities")
        r.raise_for_status()
        return Capabilities.model_validate(r.json())

    def evaluate(self, state: ProcessState, overrides: dict[str, float] | None = None) -> AgentReport:
        r = self._http.post("/evaluate", json={"state": state.model_dump(mode="json"), "overrides": overrides})
        r.raise_for_status()
        return AgentReport.model_validate(r.json())