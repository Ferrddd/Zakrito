from pathlib import Path

from .optimization.bounds import (
    build_historical_bounds,
    load_historical_stats,
)
from .optimization.catalog import CONTROL_CATALOG


class OptimizationAgent:
    def __init__(
        self,
        historical_stats_path: str | Path = (
            "data/converted/historical_stats.csv"
        ),
    ) -> None:

        historical_stats = load_historical_stats(
            historical_stats_path
        )

        control_tags = {
            control.tag
            for control in CONTROL_CATALOG
        }

        self.bounds = build_historical_bounds(
            historical_stats=historical_stats,
            tags=control_tags,
        )

        print("CONTROL TAGS:")
        print(sorted(control_tags))

        print("\nBOUNDS:")
        for tag, bound in sorted(self.bounds.items()):
            print(
                tag,
                "->",
                bound.lower,
                bound.upper,
                bound.median,
            )