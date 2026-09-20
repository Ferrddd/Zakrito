from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .tag_resolver import build_tag_mapping


@dataclass(frozen=True)
class VariableBounds:
    tag: str
    lower: float
    upper: float
    median: float
    source: str = "historical_stats.csv"


def load_historical_stats(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Historical stats file not found: {path}"
        )

    stats = pd.read_csv(path)

    required_columns = {"tag", "min", "max", "median"}
    missing = required_columns - set(stats.columns)

    if missing:
        raise ValueError(
            "Historical stats is missing columns: "
            f"{sorted(missing)}"
        )

    return stats


def build_historical_bounds(
    historical_stats: pd.DataFrame,
    tags: set[str],
) -> dict[str, VariableBounds]:

    available_tags = set(
        historical_stats["tag"].astype(str)
    )

    tag_mapping = build_tag_mapping(
        catalog_tags=tags,
        available_tags=available_tags,
    )

    stats = historical_stats.set_index("tag")

    result: dict[str, VariableBounds] = {}

    for catalog_tag in tags:
        actual_tag = tag_mapping.get(catalog_tag)

        if actual_tag is None:
            continue

        row = stats.loc[actual_tag]

        lower = float(row["min"])
        upper = float(row["max"])
        median = float(row["median"])

        if lower >= upper:
            continue

        result[catalog_tag] = VariableBounds(
            tag=catalog_tag,
            lower=lower,
            upper=upper,
            median=median,
        )

    return result

