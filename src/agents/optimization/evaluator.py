from __future__ import annotations

from src.schemas.agents_schemas import (
    OptimizationInput,
    RiskClass,
)

from .models import (
    OptimizationScenario,
    ScenarioEvaluation,
)
from .objective import ObjectiveEvaluator
from .predictor import QualityPredictor


class ScenarioEvaluator:
    def __init__(
        self,
        predictor: QualityPredictor,
        objective: ObjectiveEvaluator,
    ) -> None:
        self.predictor = predictor
        self.objective = objective

    def evaluate(
        self,
        optimization_input: OptimizationInput,
        scenario: OptimizationScenario,
    ) -> OptimizationScenario:

        quality = self.predictor.predict(
            optimization_input,
            scenario,
        )

        violations: list[str] = []

        # -----------------------------------------------
        # Hard quality constraint
        # -----------------------------------------------

        for metric, risk in quality.risk_exceed_spec.items():
            if risk >= 1.0:
                violations.append(
                    f"{metric}: risk_exceed_spec={risk:.3f}"
                )

        # -----------------------------------------------
        # Reliability
        # -----------------------------------------------

        reliability_score = (
            optimization_input.reliability.severity_index
        )

        reliability_class = (
            optimization_input.reliability.risk_class
        )

        quality_risk = (
            max(quality.risk_exceed_spec.values())
            if quality.risk_exceed_spec
            else 0.0
        )

        objective_value = self.objective.evaluate(
            quality_risk=quality_risk,
            reliability_score=reliability_score,
            throughput_score=0.0,
            energy_score=0.0,
        )

        scenario.evaluation = ScenarioEvaluation(
            quality=quality,
            reliability_score=reliability_score,
            reliability_class=reliability_class,
            throughput_score=0.0,
            energy_score=0.0,
            objective_value=objective_value,
            feasible=not violations,
            violations=violations,
        )

        return scenario