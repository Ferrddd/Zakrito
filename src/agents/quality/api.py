

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.quality.agent import QualityAgent
from src.schemas.process_state import AgentReport, Capabilities, ProcessState

app = FastAPI(title="Quality Agent", version="1.1.0")


class EvaluateRequest(BaseModel):
    state: ProcessState
    overrides: dict[str, float] | None = None


@lru_cache(maxsize=1)
def get_agent() -> QualityAgent:
    return QualityAgent.load(os.environ.get("QUALITY_CONFIG", "src/agents/quality/config.yaml"))


@app.get("/health")
def health() -> dict:
    agent = get_agent()
    return {"status": "ok", "version": agent.version, "horizons_min": [h * agent.step_min for h in agent.horizons]}


@app.get("/capabilities", response_model=Capabilities)
def capabilities() -> Capabilities:
    return get_agent().capabilities()


@app.post("/history")
def history(states: list[ProcessState]) -> dict:
    agent = get_agent()
    for s in states:
        agent.observe(s)
    return {"accepted": len(states)}


@app.post("/evaluate", response_model=AgentReport)
def evaluate(req: EvaluateRequest) -> AgentReport:
    return get_agent().evaluate(req.state, req.overrides)