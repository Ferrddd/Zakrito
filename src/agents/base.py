"""Общий протокол агента, который видит оркестратор."""

from __future__ import annotations

from typing import Protocol

from src.schemas.process_state import AgentReport, Capabilities, ProcessState


class Agent(Protocol):
    name: str
    version: str

    def capabilities(self) -> Capabilities: ...

    def evaluate(self, state: ProcessState,
                 overrides: dict[str, float] | None = None) -> AgentReport:
        """overrides — what-if: подменить текущие значения тегов (сценарий оптимизатора)."""
        ...