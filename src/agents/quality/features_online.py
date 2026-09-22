

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Создаются внутри add_engineered, во входном состоянии их нет.
DERIVED_TAGS = {"wabt", "h2_to_feed", "load_rel", "is_startup"}


def base_tag(col: str) -> str:
    """'T5_hdt__lag6' -> 'T5_hdt' (так же, как build_monotone_constraints)."""
    return col.split("__")[0]


def to_regular_grid(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Регулярная сетка: пропущенные такты = NaN-строки, но block_id протягивается,
    чтобы позиции внутри блока (а значит и лаги в точках) соответствовали времени."""
    out = df.asfreq(freq)
    if "block_id" in out.columns:
        out["block_id"] = out["block_id"].ffill()
    return out


class OnlineFeatureBuilder:
    def __init__(self, raw_cols: list[str], cols: list[str], lags: list[int], windows: list[int],
                 grid_freq: str = "10min",
                 feature_fns: tuple[Callable, Callable] | None = None,
                 feed_col: str | None = None, feed_median: float | None = None):
        """feature_fns=(add_engineered, add_lag_features) — для тестов; по умолчанию
        импортируются из src.data_pipeline.features."""
        if len(raw_cols) != len(cols):
            raise ValueError("raw_cols и cols должны быть одной длины (порядок = порядок модели)")
        self.raw_cols = raw_cols
        self.cols = cols
        self.lags = tuple(lags)
        self.windows = tuple(windows)
        self.grid_freq = grid_freq
        self.feed_col = feed_col
        self.feed_median = feed_median
        # что нужно на входе (без производных признаков)
        self.base_tags = sorted({base_tag(c) for c in raw_cols} - DERIVED_TAGS)
        # что реально лагируется: только колонки, у которых в модели есть суффикс
        self.lag_sources = sorted({base_tag(c) for c in raw_cols if "__" in c})
        self._fns = feature_fns
        self._warned_block = False
        if "load_rel" in raw_cols and feed_median is None:
            logger.warning("В feature_spec нет feed_median — load_rel онлайн может отличаться от "
                           "обучающего (медиана по буферу). Переобучите модель и проверьте "
                           "tests/test_online_parity.py")

    @classmethod
    def from_spec_file(cls, path: Path, feature_fns: tuple[Callable, Callable] | None = None
                       ) -> OnlineFeatureBuilder:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_cols = spec.get("raw_cols") or spec["cols"]
        return cls(raw_cols, spec["cols"], spec["lags_points"], spec["window_points"],
                   spec.get("grid_freq", "10min"), feature_fns,
                   spec.get("feed_col"), spec.get("feed_median"))

    @property
    def min_history_points(self) -> int:
        """Сколько точек нужно, чтобы самый длинный лаг/окно были определены."""
        return max(max(self.lags, default=0), max(self.windows, default=0)) + 1

    def _get_fns(self) -> tuple[Callable, Callable]:
        if self._fns is None:
            from src.data_pipeline.features import add_engineered, add_lag_features
            self._fns = (add_engineered, add_lag_features)
        return self._fns

    def build(self, history: pd.DataFrame) -> pd.DataFrame:
        """history — регулярная сетка (индекс = время, шаг grid_freq), последняя строка = «сейчас».
        Возвращает DataFrame из одной строки с колонками self.cols."""
        add_engineered, add_lag_features = self._get_fns()
        df = history.copy()
        if "block_id" not in df.columns:
            if not self._warned_block:
                logger.warning("В состоянии нет block_id — считаю всю историю одним блоком "
                               "(лаги могут пересечь останов). Передавайте block_id из витрины.")
                self._warned_block = True
            df["block_id"] = 0

        df = add_engineered(df)
        if self.feed_col and self.feed_median is not None and self.feed_col in df.columns \
                and "load_rel" in df.columns:
            df["load_rel"] = df[self.feed_col] / (self.feed_median + 1e-6)

        sources = [t for t in self.lag_sources if t in df.columns]
        absent = sorted(set(self.lag_sources) - set(sources))
        if absent:
            logger.debug("В истории нет %d колонок для лагов (будут NaN): %s", len(absent), absent[:5])
        df = add_lag_features(df, sources, lags=self.lags, windows=self.windows, n_jobs=1)
        last = df.iloc[[-1]].reindex(columns=self.raw_cols)
        last.columns = self.cols
        return last.astype(float)