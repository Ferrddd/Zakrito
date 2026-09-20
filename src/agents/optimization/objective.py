from __future__ import annotations

from dataclasses import dataclass

from .models import ScenarioEvaluation


@dataclass(frozen=True)
class ObjectiveWeights:
    reliability: float = 1.0
    throughput: float = 1.0
    energy: float = 0.5


class ObjectiveEvaluator:
    def __init__(
        self,
        weights: ObjectiveWeights | None = None,
    ) -> None:
        self.weights = weights or ObjectiveWeights()

    def evaluate(
        self,
        quality_risk: float,
        reliability_score: float,
        throughput_score: float,
        energy_score: float,
    ) -> float:

        return (
            self.weights.reliability * reliability_score
            + self.weights.throughput * throughput_score
            + self.weights.energy * energy_score
            + 100.0 * quality_risk
        )