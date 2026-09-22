
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import BASE_DIR, load_config
from src.utils.tags_catalog import load_tags_catalog, resolve_column

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    monotone_directions: dict[str, int]       # column -> {-1,0,1}, для violation-классификатора
    sulfur_analyzer_columns: tuple[str, ...]  # чёрный список: те же измерения, что и таргет

    target_tag: str
    target_unit: str
    target_limit: float

    horizons_points: tuple[int, ...]
    grid_freq: str
    lags_points: tuple[int, ...]
    window_points: tuple[int, ...]

    holdout_size: str
    cv_n_folds: int
    cv_fold_size: str
    cv_expanding: bool
    embargo_safety: float

    quantiles: tuple[float, ...]
    violation_min_precision: float
    lgbm_params: dict
    early_stopping_rounds: int

    models_dir: Path
    train_export: Path
    test_export: Path
    metrics_export: Path


def _resolve_path(p: str, base_dir: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else base_dir / path


def resolve_column_name(name: str, columns: Iterable[str]) -> str:
    """Имя из каталога ('T5_hdt') -> реальное имя колонки ('T5' или 'T5_hdt').

    Фолбэк на имя без суффикса разрешён, только если у него нет «двойника» из
    другой установки: 'T11' без суффикса не подменит 'T11_hdt', если есть 'T11_avt'
    (значит, суффиксы были, и колонка просто отсутствует).
    """
    cols = set(columns)
    if name in cols:
        return name
    base, sep, suffix = name.rpartition("_")
    if sep and suffix in ("hdt", "avt"):
        other = "avt" if suffix == "hdt" else "hdt"
        if base in cols and f"{base}_{other}" not in cols:
            return base
    raise KeyError(
        f"Колонка '{name}' из конфига не найдена в данных (нет ни '{name}', ни "
        f"безсуффиксного варианта). Проверь tag_id/installation в конфиге.")


def resolve_cfg_columns(cfg: FeatureConfig, columns: Iterable[str]) -> FeatureConfig:
    """Привязывает monotone и чёрный список анализаторов к реальным колонкам."""
    columns = list(columns)
    monotone = {}
    for name, direction in cfg.monotone_directions.items():
        real = resolve_column_name(name, columns)
        if real != name:
            logger.info("monotone: %s -> %s", name, real)
        monotone[real] = direction
    analyzers = []
    for name in cfg.sulfur_analyzer_columns:
        real = resolve_column_name(name, columns)
        if real != name:
            logger.info("чёрный список анализаторов: %s -> %s", name, real)
        analyzers.append(real)
    return dataclasses.replace(cfg, monotone_directions=monotone,
                               sulfur_analyzer_columns=tuple(analyzers))


def load_feature_config(config_path: str | Path = "src/agents/quality/config.yaml",
                        base_dir: Path = BASE_DIR) -> FeatureConfig:
    cfg = load_config(config_path)
    catalog = load_tags_catalog(cfg["tags_catalog"])

    def column(entry: dict) -> str:
        return resolve_column(catalog, entry["installation"], entry["tag_id"])

    monotone = {column(m): int(m["direction"]) for m in cfg.get("monotone", [])}
    sulfur_cols = tuple(column(t) for t in cfg["sulfur_analyzer_tags"])

    target = cfg["target"]

    return FeatureConfig(
        monotone_directions=monotone,
        sulfur_analyzer_columns=sulfur_cols,
        target_tag=target["tag"],
        target_unit=target["unit"],
        target_limit=float(target["limit"]),
        horizons_points=tuple(cfg["horizons_points"]),
        grid_freq=cfg["grid_freq"],
        lags_points=tuple(cfg["lags_points"]),
        window_points=tuple(cfg["window_points"]),
        holdout_size=cfg["holdout_size"],
        cv_n_folds=int(cfg["cv_n_folds"]),
        cv_fold_size=cfg["cv_fold_size"],
        cv_expanding=bool(cfg["cv_expanding"]),
        embargo_safety=float(cfg["embargo_safety"]),
        quantiles=tuple(cfg["quantiles"]),
        violation_min_precision=float(cfg["violation_min_precision"]),
        lgbm_params=cfg["lgbm_params"],
        early_stopping_rounds=int(cfg["early_stopping_rounds"]),
        models_dir=_resolve_path(cfg["models_dir"], base_dir),
        train_export=_resolve_path(cfg["train_export"], base_dir),
        test_export=_resolve_path(cfg["test_export"], base_dir),
        metrics_export=_resolve_path(cfg["metrics_export"], base_dir),
    )