
from __future__ import annotations

import httpx
import pandas as pd

from src.orchestrator.adapters import state_from_row
from src.schemas.process_state import AgentReport, Capabilities, ProcessState


class RemoteQualityAgent:
    name = "quality"
    version = "remote"

    def __init__(self, base_url: str = "http://localhost:8001", timeout_s: float = 30.0):
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def warm_up(self, frame: pd.DataFrame, batch: int = 96) -> int:
        """Прогрев буфера удалённого агента через POST /history (без 288 полных evaluate)."""
        rows = [state_from_row(ts, row).model_dump(mode="json") for ts, row in frame.iterrows()]
        for i in range(0, len(rows), batch):
            r = self._http.post("/history", json=rows[i:i + batch])
            r.raise_for_status()
        return len(rows)

    def capabilities(self) -> Capabilities:
        r = self._http.get("/capabilities")
        r.raise_for_status()
        return Capabilities.model_validate(r.json())

    def evaluate(self, state: ProcessState, overrides: dict[str, float] | None = None) -> AgentReport:
        r = self._http.post("/evaluate", json={"state": state.model_dump(mode="json"), "overrides": overrides})
        r.raise_for_status()
        return AgentReport.model_validate(r.json())