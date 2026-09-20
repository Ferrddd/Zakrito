from __future__ import annotations

from itertools import combinations

from src.schemas.agents_schemas import ProcessSnapshot

from .bounds import VariableBounds
from .catalog import ControlDefinition
from .models import ControlChange, OptimizationScenario


class ScenarioGenerator:
    def __init__(
        self,
        step_fraction: float = 0.02,
        max_pair_controls: int = 2,
    ) -> None:
        self.step_fraction = step_fraction
        self.max_pair_controls = max_pair_controls

    def generate(
        self,
        snapshot: ProcessSnapshot,
        controls: tuple[ControlDefinition, ...],
        bounds: dict[str, VariableBounds],
    ) -> list[OptimizationScenario]:

        scenarios: list[OptimizationScenario] = []

        # --------------------------------------------------
        # 1. No-op scenario
        # --------------------------------------------------

        scenarios.append(
            OptimizationScenario(
                scenario_id="baseline",
                changes=[],
            )
        )

        # --------------------------------------------------
        # 2. Single-control scenarios
        # --------------------------------------------------

        for control in controls:
            current = snapshot.telemetry.get(control.tag)

            if current is None:
                continue

            variable_bounds = bounds.get(control.tag)

            if variable_bounds is None:
                continue

            delta = self._step(variable_bounds, current)

            for direction in (-1.0, 1.0):
                proposed = current + direction * delta

                proposed = min(
                    max(proposed, variable_bounds.lower),
                    variable_bounds.upper,
                )

                if proposed == current:
                    continue

                scenarios.append(
                    OptimizationScenario(
                        scenario_id=(
                            f"{control.tag}_"
                            f"{'down' if direction < 0 else 'up'}"
                        ),
                        changes=[
                            ControlChange(
                                tag=control.tag,
                                description=control.description,
                                unit=control.unit,
                                current_value=current,
                                proposed_value=proposed,
                                delta=proposed - current,
                                lower_bound=variable_bounds.lower,
                                upper_bound=variable_bounds.upper,
                            )
                        ],
                    )
                )

        # --------------------------------------------------
        # 3. Pair scenarios
        # --------------------------------------------------

        controls_with_values = [
            control
            for control in controls
            if control.tag in snapshot.telemetry
            and control.tag in bounds
        ]

        for first, second in combinations(
            controls_with_values,
            self.max_pair_controls,
        ):
            first_current = snapshot.telemetry[first.tag]
            second_current = snapshot.telemetry[second.tag]

            first_bounds = bounds[first.tag]
            second_bounds = bounds[second.tag]

            first_delta = self._step(
                first_bounds,
                first_current,
            )

            second_delta = self._step(
                second_bounds,
                second_current,
            )

            changes = [
                ControlChange(
                    tag=first.tag,
                    description=first.description,
                    unit=first.unit,
                    current_value=first_current,
                    proposed_value=min(
                        max(
                            first_current + first_delta,
                            first_bounds.lower,
                        ),
                        first_bounds.upper,
                    ),
                    delta=0.0,
                    lower_bound=first_bounds.lower,
                    upper_bound=first_bounds.upper,
                ),
                ControlChange(
                    tag=second.tag,
                    description=second.description,
                    unit=second.unit,
                    current_value=second_current,
                    proposed_value=min(
                        max(
                            second_current + second_delta,
                            second_bounds.lower,
                        ),
                        second_bounds.upper,
                    ),
                    delta=0.0,
                    lower_bound=second_bounds.lower,
                    upper_bound=second_bounds.upper,
                ),
            ]

            for change in changes:
                change.delta = (
                    change.proposed_value - change.current_value
                )

            scenarios.append(
                OptimizationScenario(
                    scenario_id=f"{first.tag}_{second.tag}_up",
                    changes=changes,
                )
            )

        return scenarios

    def _step(
        self,
        bounds: VariableBounds,
        current: float,
    ) -> float:
        span = bounds.upper - bounds.lower

        if span <= 0:
            return 0.0

        return span * self.step_fraction