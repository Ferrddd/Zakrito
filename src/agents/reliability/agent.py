from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_pipeline.feature_config import resolve_column_name
from src.schemas.agents_schemas import ProcessSnapshot, ReliabilityAssessment, RiskClass

logger = logging.getLogger(__name__)

_SERVICE = {"block_id", "is_valid", "on_grid", "anomaly_share", "hours_since_block_start",
            "is_startup", ""}
_MIN_SCALE = 1e-9


@dataclass(frozen=True)
class ReliabilityConfig:
    feed_tags: tuple[str, ...] = ("T11", "F26")     # массовый расход предпочтительнее
    temp_tag: str = "T5"
    dp_tag: str = "F19"
    load_overload_span: float = 0.15                # +15% к медиане -> load_score = 1
    temp_z_start: float = 1.0                       # мёртвая зона по температуре, σ
    temp_z_span: float = 3.0
    dp_trend_hours: float = 6.0
    dp_min_points: int = 6
    dp_min_span_hours: float = 2.0
    dp_z_start: float = 1.0
    dp_z_span: float = 3.0
    z_threshold: float = 5.0                        # как в loader (pipeline_models.z_threshold)
    anomaly_start: float = 0.05
    anomaly_span: float = 0.20
    weights: dict[str, float] = field(default_factory=lambda: {
        "load": 0.30, "temp": 0.30, "dp": 0.25, "anomaly": 0.15})
    warning_at: float = 0.45
    critical_at: float = 0.75
    downtime_ratio: float = 0.10                    # как downtime_load_ratio в loader
    history_hours: float = 24.0


@dataclass
class ReliabilityBaseline:
    feed_col: str | None = None
    feed_median: float | None = None
    temp_col: str | None = None
    temp_fit: tuple[float, float, float] | None = None    # (a, b, sigma): T5 = a + b*load_rel
    dp_col: str | None = None
    dp_sigma: float | None = None                          # робастный σ 6-часового изменения
    channels: dict[str, tuple[float, float]] = field(default_factory=dict)  # col -> (median, scale)


def _find(cols: Iterable[str], tags: Iterable[str]) -> str | None:
    cols = list(cols)
    for t in tags:
        try:
            return resolve_column_name(f"{t}_hdt", cols)
        except KeyError:
            continue
    return None


def _robust_scale(s: pd.Series) -> float:
    mad = float((s - s.median()).abs().median()) * 1.4826
    return mad if mad > _MIN_SCALE else float(s.std(ddof=0))


def _is_channel(col: str, s: pd.Series) -> bool:
    return not (col in _SERVICE or col.startswith(("mask_", "target_", "lims_")) or "__" in col
                or ":" in col or s.dtype.kind not in "if")


class ReliabilityAgent:
    name = "reliability"
    is_stub = False

    def __init__(self, baseline: ReliabilityBaseline | None = None,
                 cfg: ReliabilityConfig | None = None):
        self.cfg = cfg or ReliabilityConfig()
        self.baseline = baseline or ReliabilityBaseline()
        self._hist: deque[tuple[pd.Timestamp, float | None]] = deque()   # (время, значение dp-тега)

    # ------------------------------------------------------------------ обучение базовой линии
    @classmethod
    def fit(cls, df: pd.DataFrame, cfg: ReliabilityConfig | None = None) -> ReliabilityAgent:
        """Базовая линия «нормы» по истории. Передавайте историю ДО момента прогона."""
        cfg = cfg or ReliabilityConfig()
        b = ReliabilityBaseline()
        valid_mask = df["is_valid"].fillna(False) if "is_valid" in df.columns else pd.Series(True, index=df.index)
        valid = df[valid_mask]
        if valid.empty:
            logger.warning("Надёжность: нет валидной истории — базовая линия пуста")
            return cls(b, cfg)

        b.feed_col = _find(df.columns, cfg.feed_tags)
        if b.feed_col:
            b.feed_median = float(valid[b.feed_col].median())

        b.temp_col = _find(df.columns, (cfg.temp_tag,))
        if b.temp_col and b.feed_col and b.feed_median:
            xy = pd.DataFrame({"x": valid[b.feed_col] / b.feed_median, "y": valid[b.temp_col]}).dropna()
            if len(xy) > 100 and xy["x"].std() > 1e-6:
                slope, intercept = np.polyfit(xy["x"], xy["y"], 1)
                sigma = _robust_scale(xy["y"] - (intercept + slope * xy["x"]))
                if sigma > _MIN_SCALE:
                    b.temp_fit = (float(intercept), float(slope), float(sigma))

        b.dp_col = _find(df.columns, (cfg.dp_tag,))
        if b.dp_col:
            steps = round(cfg.dp_trend_hours * 6)
            d = df[b.dp_col].where(valid_mask).diff(steps).dropna()
            if len(d) > 100 and _robust_scale(d) > _MIN_SCALE:
                b.dp_sigma = _robust_scale(d)

        for col in valid.columns:
            if _is_channel(col, valid[col]):
                s = valid[col].dropna()
                if len(s) > 100:
                    scale = _robust_scale(s)
                    if scale > _MIN_SCALE:
                        b.channels[col] = (float(s.median()), scale)
        logger.info("Надёжность: feed=%s temp=%s(%s) dp=%s каналов=%d", b.feed_col, b.temp_col,
                    "fit" if b.temp_fit else "нет fit", b.dp_col, len(b.channels))
        return cls(b, cfg)

    def warm_up(self, df: pd.DataFrame) -> None:
        """Прогрев буфера тренда историей перед первым циклом."""
        col = self.baseline.dp_col
        if col is None or col not in df.columns:
            return
        for ts, v in df[col].items():
            self._push(pd.Timestamp(str(ts)), None if pd.isna(v) else float(v))

    # ------------------------------------------------------------------ оценка
    def assess(self, snapshot: ProcessSnapshot) -> ReliabilityAssessment:
        b, cfg = self.baseline, self.cfg
        tel = snapshot.telemetry
        ts = pd.Timestamp(snapshot.timestamp)
        if b.dp_col:
            self._push(ts, tel.get(b.dp_col))

        scores: dict[str, float] = {}
        factors: list[str] = []
        downtime = False
        dp_slope: float | None = None

        # load
        feed = tel.get(b.feed_col) if b.feed_col else None
        load_rel: float | None = None
        if feed is not None and b.feed_median:
            load_rel = feed / b.feed_median
            downtime = feed < cfg.downtime_ratio * b.feed_median
            scores["load"] = _clip((load_rel - 1.0) / cfg.load_overload_span)
            if scores["load"] >= 0.3:
                factors.append(f"загрузка {load_rel:.2f}× от медианы истории")
        else:
            factors.append("нет расхода сырья или базовой линии — нагрузка не оценена")

        # temp
        t = tel.get(b.temp_col) if b.temp_col else None
        if t is not None and b.temp_fit and load_rel is not None:
            a, slope, sigma = b.temp_fit
            z = (t - (a + slope * load_rel)) / sigma
            scores["temp"] = _clip((z - cfg.temp_z_start) / cfg.temp_z_span)
            if scores["temp"] >= 0.3:
                factors.append(f"{b.temp_col} выше нормы для текущей нагрузки на {z:.1f}σ (прокси WABT)")
        else:
            factors.append("температура слоя (T5) не оценена — нет данных или базовой линии")

        # dp
        dp_slope = self._dp_slope()
        if dp_slope is not None and b.dp_sigma:
            z = dp_slope * cfg.dp_trend_hours / b.dp_sigma
            scores["dp"] = _clip((z - cfg.dp_z_start) / cfg.dp_z_span)
            if scores["dp"] >= 0.3:
                factors.append(f"рост {b.dp_col} {dp_slope:+.3g}/ч за {cfg.dp_trend_hours:g} ч "
                               f"({z:.1f}σ, прокси роста ΔP)")
        else:
            factors.append("тренд ΔP (F19) не оценён — мало истории или нет базовой линии")

        # anomaly
        share = self._anomaly_share(tel)
        if share is not None:
            scores["anomaly"] = _clip((share - cfg.anomaly_start) / cfg.anomaly_span)
            if scores["anomaly"] >= 0.3:
                factors.append(f"{share:.0%} каналов телеметрии вне нормы (|z|>{cfg.z_threshold:g})")

        severity = self._severity(scores)
        risk = (RiskClass.critical if severity >= cfg.critical_at
                else RiskClass.warning if severity >= cfg.warning_at else RiskClass.normal)
        if downtime:
            factors.append("установка в простое (расход сырья < "
                           f"{cfg.downtime_ratio:.0%} медианы)")
            if risk == RiskClass.normal:
                risk = RiskClass.warning
        if not scores:
            factors.append("оценка тяжести режима НЕ выполнена: нет ни одной доступной компоненты")

        return ReliabilityAssessment(
            timestamp=snapshot.timestamp, severity_index=severity, dp_trend_slope=dp_slope,
            downtime_score=share, risk_class=risk, downtime_flag=downtime, limiting_factors=factors)

    # ------------------------------------------------------------------ внутреннее
    def _severity(self, scores: dict[str, float]) -> float:
        if not scores:
            return 0.0
        w = self.cfg.weights
        wsum = sum(w[k] for k in scores)
        mean = sum(w[k] * v for k, v in scores.items()) / wsum
        return _clip(0.5 * mean + 0.5 * max(scores.values()))

    def _push(self, ts: pd.Timestamp, value: float | None) -> None:
        if self._hist and ts < self._hist[-1][0]:
            self._hist.clear()                     # новый прогон/перемотка назад
        if self._hist and ts == self._hist[-1][0]:
            self._hist.pop()                       # повторная оценка того же момента
        self._hist.append((ts, value))
        horizon = ts - pd.Timedelta(hours=self.cfg.history_hours)
        while self._hist and self._hist[0][0] < horizon:
            self._hist.popleft()

    def _dp_slope(self) -> float | None:
        """МНК-наклон dp-тега за последние dp_trend_hours (ед./ч); по времени, не по номеру строки."""
        if not self._hist:
            return None
        end = self._hist[-1][0]
        start = end - pd.Timedelta(hours=self.cfg.dp_trend_hours)
        pts = [(t, v) for t, v in self._hist if t >= start and v is not None]
        if len(pts) < self.cfg.dp_min_points:
            return None
        hours = np.array([(t - pts[0][0]).total_seconds() / 3600 for t, _ in pts])
        if hours[-1] < self.cfg.dp_min_span_hours:
            return None
        return float(np.polyfit(hours, np.array([v for _, v in pts]), 1)[0])

    def _anomaly_share(self, tel: dict[str, float]) -> float | None:
        checked = out = 0
        for col, (med, scale) in self.baseline.channels.items():
            v = tel.get(col)
            if v is None:
                continue
            checked += 1
            out += abs(v - med) / scale > self.cfg.z_threshold
        return out / checked if checked >= 10 else None


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))