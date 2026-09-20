from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas.agents_schemas import (
    OptimizationInput,
    ProcessSnapshot,
)

from .models import OptimizationScenario, PredictedQuality


class QualityPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        optimization_input: OptimizationInput,
        scenario: OptimizationScenario,
    ) -> PredictedQuality:
        raise NotImplementedError

class CurrentStatePredictor(QualityPredictor):
    """
    Заглушка до появления обученной counterfactual-модели.

    Она НЕ делает вид, что умеет предсказывать эффект
    управляющего воздействия.

    Поэтому просто возвращает текущий прогноз.
    """

    def predict(
        self,
        optimization_input: OptimizationInput,
        scenario: OptimizationScenario,
    ) -> PredictedQuality:

        return PredictedQuality(
            values=dict(
                optimization_input.quality.quality_forecast
            ),
            risk_exceed_spec=dict(
                optimization_input.quality.risk_exceed_spec
            ),
            confidence=optimization_input.quality.confidence,
        )