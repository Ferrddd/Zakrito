"""Агент надёжности v1: тяжесть режима гидроочистки по прокси-признакам.

Реализует протокол ReliabilityAgent из src/orchestrator/contracts.py
(name, is_stub=False, assess(snapshot) -> ReliabilityAssessment).

Что считается (каждый компонент — score в [0, 1]):
  dp           закоксовывание/забивка слоя Р-202: рост перепада давления W10.
               Если обучена модель нормального поведения (nbm.py, `make train_reliability`) —
               по остатку «факт - модель» (уровень и тренд); иначе — по упрощённому правилу
               (ΔP, нормированный на нагрузку: тренд за окно И уровень относительно нормы);
  temperature  превышение нормы по температурам реакторов (T5, P8), робастный z;
  anomaly      доля тегов телеметрии с |robust z| > порога (поле downtime_score);
  load         перегрузка по расходу сырья относительно медианы.

severity_index = max( взвешенная сумма, hard_floor_k * max(dp, temperature) ).
Второй член нужен, чтобы один сильный физический сигнал не «размывался» остальными.

Классы: normal / warning / critical. critical выставляется только если держится
critical_persistence циклов подряд — оркестратор на critical отказывается от
рекомендаций (эскалация оператору), одиночный выброс КИП не должен этого делать.

ДОПУЩЕНИЯ (явно, по ТЗ п.4): размеченных отказов нет, пороги — модельные, не
промышленные пределы; нормы считаются по скользящей истории самого агента.
Агент честно молчит (normal + пояснение), пока истории меньше min_history_points.

Агент состоятельный: копит историю снимков, которые ему передаёт оркестратор,
поэтому перед демо его нужно прогреть warm_up(df) (см. scripts/run_orchestrator.py).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.reliability.nbm import NormalBehaviorModel, mask_sentinels
from src.schemas.agents_schemas import ProcessSnapshot, ReliabilityAssessment, RiskClass
from src.utils.config import BASE_DIR, load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "src/agents/reliability/config.yaml"
_MAD_TO_SIGMA = 1.4826
FACTOR_MIN_SCORE = 0.25     # компонент от этого score попадает в limiting_factors


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 при x<=lo, 1 при x>=hi, линейно между."""
    return _clip01((x - lo) / (hi - lo)) if hi > lo else float(x >= hi)


@dataclass
class _Baseline:
    names: list[str]              # колонки для оценки доли аномальных тегов
    med: np.ndarray
    scale: np.ndarray
    med_by_name: dict[str, float]
    scale_by_name: dict[str, float]
    feed_med: float | None
    dp_med: float | None          # медиана нормированного ΔP
    dp_scale: float | None


class RuleBasedReliabilityAgent:
    name = "reliability"
    version = "1.1"
    is_stub = False

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._rows: deque[tuple[datetime, dict[str, float]]] = deque(maxlen=int(cfg["history_points"]))
        self._base: _Baseline | None = None
        self._since_refresh = 0
        self._crit_streak = 0
        self._col_cache: dict[str, str] = {}
        self._sentinels = tuple(float(x) for x in cfg.get("sentinel_values", []))
        self._nbm: NormalBehaviorModel | None = None
        self._res: deque[tuple[datetime, float, float]] = deque(maxlen=600)   # (время, факт ΔP, модель)
        self._load_nbm()

    @classmethod
    def load(cls, config_path: str | Path = DEFAULT_CONFIG) -> RuleBasedReliabilityAgent:
        return cls(load_config(config_path))

    # ------------------------------------------------------------------ модель нормального поведения
    @property
    def dp_mode(self) -> str:
        return "nbm" if self._nbm is not None else "rules"

    def _load_nbm(self) -> None:
        nb = self.cfg.get("nbm") or {}
        if not nb.get("enabled"):
            return
        path = Path(nb["model_path"])
        path = path if path.is_absolute() else BASE_DIR / path
        try:
            model = NormalBehaviorModel.load(path)
        except Exception:
            logger.exception("Не удалось загрузить модель NBM %s — ΔP по правилам", path)
            return
        if model is None:
            logger.warning("Модель NBM не найдена (%s) — ΔP по упрощённым правилам. Обучение: make train_reliability", path)
            return
        skill = model.metrics.get("holdout_skill_vs_persistence")
        if skill is None or not np.isfinite(skill) or skill < float(nb["min_skill"]):
            logger.warning("NBM не лучше наивного прогноза на holdout (skill=%s < %s) — ΔP по правилам", skill, nb["min_skill"])
            return
        self._nbm = model
        logger.info("NBM загружена: skill=%.3f, R2=%.3f", skill, model.metrics.get("holdout_r2", float("nan")))

    def _prefill_residuals(self, df: pd.DataFrame) -> None:
        nb = self._nbm
        if nb is None:
            return
        if any(c not in df.columns for c in nb.base_cols + [nb.target_col]):
            return
        n = int(self.cfg["nbm"]["prefill_points"])
        tail = df.iloc[-(n + nb.lag + nb.roll + 2):]
        pred, actual = nb.predict_actual(tail)
        feed = tail[nb.feed_col].to_numpy(dtype=float) if nb.feed_col and nb.feed_col in tail.columns else None
        feed_med = self._base.feed_med if self._base else None
        for i in range(max(0, len(tail) - n), len(tail)):
            if not (np.isfinite(actual[i]) and np.isfinite(pred[i])):
                continue
            if feed is not None and feed_med and not feed[i] >= self.cfg["downtime_load_ratio"] * feed_med:
                continue
            self._res.append((pd.Timestamp(tail.index[i]).to_pydatetime(), float(actual[i]), float(pred[i])))

    def _nbm_dp(self, ts: datetime, tel: dict[str, float], dp_col: str,
                factors: list[str]) -> tuple[float, float | None] | None:
        """Оценка ΔP по остатку динамической NBM: (score, темп роста ΔP сверх режима, доля/сут) или None."""
        cfg, nb = self.cfg, self._nbm
        assert nb is not None
        need = nb.base_cols + [nb.target_col]
        absent = [c for c in need if c not in tel]
        if absent:
            factors.append(f"Модель ΔP недоступна (нет тегов: {', '.join(absent)}) — использован упрощённый расчёт")
            return None
        # свежий кусок истории на регулярной 10-минутной сетке (оркестратор может идти шагом > 10 мин)
        step = pd.Timedelta("10min")
        n = nb.lag + nb.roll + 2
        t_end = pd.Timestamp(ts)
        rows = [(pd.Timestamp(t), r) for t, r in self._rows if pd.Timestamp(t) >= t_end - n * step]
        if len(rows) < 2:
            return None
        hist = pd.DataFrame([r for _, r in rows], index=pd.DatetimeIndex([t for t, _ in rows]))
        hist = hist[~hist.index.duplicated()].sort_index().reindex(columns=need)
        grid = pd.date_range(end=t_end, periods=n, freq=step)
        hist = hist.reindex(grid, method="nearest", tolerance=pd.Timedelta("35min"))
        pred_arr, actual_arr = nb.predict_actual(hist)
        pred, actual = float(pred_arr[-1]), float(actual_arr[-1])
        if not (np.isfinite(pred) and np.isfinite(actual)) or pred <= 0:
            factors.append("Мало истории ΔP для модели (нужно ~6 ч) — использован упрощённый расчёт")
            return None
        if self._res and self._res[-1][0] == ts:
            self._res.pop()
        self._res.append((ts, actual, pred))

        recent = list(self._res)[-int(cfg["nbm"]["smooth_points"]):]
        rel_med = float(np.median([(a - p) / p for _, a, p in recent if p > 0]))
        sigma = max(nb.resid_scale, float(cfg["nbm"]["min_rel_sigma"]))
        z = rel_med / sigma
        lag_h = nb.lag * step.total_seconds() / 3600.0
        rate_per_day = rel_med * 24.0 / lag_h        # при постоянном темпе r остаток за лаг L = r * L
        score = _ramp(z, cfg["z_soft"], cfg["z_hard"])
        if score >= FACTOR_MIN_SCORE:
            name = cfg["names"][cfg["tags"]["dp"]]
            factors.append(f"{name} ({dp_col}) выше ожидаемого по режиму на {rel_med * 100:+.1f}% за {lag_h:.0f} ч "
                           f"({z:+.1f}σ), темп ≈ {rate_per_day * 100:+.0f}%/сут — рост сопротивления слоя "
                           f"(признак закоксовывания)")
        return score, rate_per_day

    # ------------------------------------------------------------------ история
    def warm_up(self, df: pd.DataFrame) -> None:
        """Прогрев историей витрины (индекс — время). Берутся последние history_points строк."""
        tail = mask_sentinels(df.iloc[-int(self.cfg["history_points"]):], self._sentinels)
        num = tail.select_dtypes("number")
        for ts, rec in zip(num.index, num.to_dict("records")):
            self._rows.append((pd.Timestamp(ts).to_pydatetime(), {str(k): float(v) for k, v in rec.items()}))
        self._refresh()
        self._prefill_residuals(tail)
        logger.info("Агент надёжности прогрет: %d точек, база=%s, ΔP: %s", len(self._rows), self._base is not None, self.dp_mode)

    def _push(self, ts: datetime, tel: dict[str, float]) -> None:
        if self._rows and ts <= self._rows[-1][0]:
            return  # повтор/откат времени — историю не портим
        self._rows.append((ts, {k: float(v) for k, v in tel.items()}))
        self._since_refresh += 1
        if self._base is None or self._since_refresh >= int(self.cfg["refresh_every"]):
            self._refresh()

    def _resolve(self, tag_ids: str | list[str], columns: Any) -> str | None:
        ids = [tag_ids] if isinstance(tag_ids, str) else list(tag_ids)
        for t in ids:
            for suf in self.cfg["suffixes"]:
                cand = f"{t}{suf}"
                if cand in columns:
                    return cand
        return None

    def _is_anomaly_column(self, c: str) -> bool:
        cfg = self.cfg
        return not (c in cfg["exclude_exact"] or ":" in c or "__" in c
                    or c.startswith(tuple(cfg["exclude_prefixes"])))

    def _robust(self, s: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        med = s.median()
        mad = (s - med).abs().median() * _MAD_TO_SIGMA
        scale = np.maximum(mad, self.cfg["min_scale_rel"] * med.abs())
        return med, scale.clip(lower=1e-9)

    def _refresh(self) -> None:
        cfg = self.cfg
        self._since_refresh = 0
        if len(self._rows) < int(cfg["min_history_points"]):
            self._base = None
            return
        df = pd.DataFrame([r for _, r in self._rows])
        feed_col = self._resolve(cfg["tags"]["feed"], df.columns)
        mask = np.ones(len(df), dtype=bool)
        feed_med = None
        if feed_col is not None:
            feed = df[feed_col].to_numpy(dtype=float)
            fm = float(np.nanmedian(feed)) if np.isfinite(feed).any() else 0.0
            if fm > 0:
                mask = np.where(np.isnan(feed), True, feed >= cfg["baseline_min_load_ratio"] * fm)
                if mask.sum() < len(df) // 4:   # простоев больше 75% истории — не фильтруем
                    mask = np.ones(len(df), dtype=bool)
                feed_med = float(np.nanmedian(feed[mask]))
        on = df[mask]

        cols = [c for c in on.columns if self._is_anomaly_column(c) and on[c].notna().sum() >= 10]
        med, scale = self._robust(on[cols])
        ok = [c for c in cols if scale[c] > 1e-9 and not np.isnan(med[c])]

        dp_med = dp_scale = None
        dp_col = self._resolve(cfg["tags"]["dp"], df.columns)
        if dp_col is not None and feed_med:
            dpn = self._normalize_dp(on[dp_col].to_numpy(dtype=float),
                                     (on[feed_col].to_numpy(dtype=float) / feed_med) if feed_col else None)
            dpn = dpn[np.isfinite(dpn)]
            if len(dpn) >= 10:
                dp_med = float(np.median(dpn))
                dp_scale = max(float(np.median(np.abs(dpn - dp_med)) * _MAD_TO_SIGMA),
                               cfg["dp_min_scale_rel"] * abs(dp_med), 1e-9)

        self._base = _Baseline(
            names=ok, med=med[ok].to_numpy(dtype=float), scale=scale[ok].to_numpy(dtype=float),
            med_by_name={c: float(med[c]) for c in ok}, scale_by_name={c: float(scale[c]) for c in ok},
            feed_med=feed_med, dp_med=dp_med, dp_scale=dp_scale)

    def _normalize_dp(self, dp: np.ndarray, load_rel: np.ndarray | None) -> np.ndarray:
        if load_rel is None:
            return dp
        with np.errstate(divide="ignore", invalid="ignore"):
            return dp / np.power(np.where(load_rel > 0.05, load_rel, np.nan), self.cfg["dp_load_exponent"])

    # ------------------------------------------------------------------ оценка
    def assess(self, snapshot: ProcessSnapshot) -> ReliabilityAssessment:
        cfg = self.cfg
        ts = snapshot.timestamp
        # значения-маркеры обрыва (307.0) — это «нет данных», а не измерение
        tel = {k: v for k, v in snapshot.telemetry.items()
               if not any(abs(v - s) < 1e-6 for s in self._sentinels)}
        self._push(ts, tel)
        base = self._base

        def result(sev: float, cls: RiskClass, factors: list[str], slope: float | None = None,
                   anomaly: float | None = None, downtime: bool = False) -> ReliabilityAssessment:
            return ReliabilityAssessment(
                timestamp=ts, severity_index=round(float(sev), 4), dp_trend_slope=slope,
                downtime_score=anomaly, risk_class=cls, downtime_flag=downtime, limiting_factors=factors)

        if base is None:
            self._crit_streak = 0
            return result(0.0, RiskClass.normal, [
                (
                f"Мало истории для оценки тяжести режима: {len(self._rows)} из "
                f"{cfg['min_history_points']} точек — оценка не выполнялась"
                )
            ])

        # ---- простой
        feed_col = self._resolve(cfg["tags"]["feed"], tel)
        feed = tel.get(feed_col) if feed_col else None
        load_rel = (feed / base.feed_med) if (feed is not None and base.feed_med) else None
        if load_rel is not None and load_rel < cfg["downtime_load_ratio"]:
            self._crit_streak = 0
            return result(0.0, RiskClass.normal,
                          [(
                              f"Установка в простое: расход сырья {feed:.1f} < "
                           f"{cfg['downtime_load_ratio']:.0%} медианы — оценка тяжести не применяется"
                           )],
                          downtime=True)

        factors: list[str] = []
        missing: list[str] = []
        w = cfg["weights"]
        names = cfg["names"]

        # ---- ΔP: уровень + тренд
        dp_score, slope_out = 0.0, None
        dp_col = self._resolve(cfg["tags"]["dp"], tel)
        nbm_out = (self._nbm_dp(ts, tel, dp_col, factors)
                   if (dp_col is not None and self._nbm is not None and self._nbm.target_col == dp_col) else None)
        if nbm_out is not None:
            dp_score, slope_out = nbm_out
        elif dp_col is None or base.dp_med is None or base.dp_scale is None:
            missing.append(cfg["tags"]["dp"])
        else:
            lr = load_rel if load_rel else 1.0
            dpn_now = float(self._normalize_dp(np.array([tel[dp_col]]), np.array([lr]))[0])
            level_score = 0.0
            if np.isfinite(dpn_now):
                z = (dpn_now - base.dp_med) / base.dp_scale
                level_score = _ramp(z, cfg["z_soft"], cfg["z_hard"])
                if level_score >= FACTOR_MIN_SCORE:
                    factors.append(f"{names[cfg['tags']['dp']]} ({dp_col}) выше нормы: +{z:.1f}σ "
                                   f"от медианы за историю (с поправкой на нагрузку)")
            slope_out, slope_score = self._dp_slope(ts, base, dp_col)
            if slope_out is not None and slope_score >= FACTOR_MIN_SCORE:
                factors.append(f"{names[cfg['tags']['dp']]} растёт {slope_out * 100:.1f}%/сут за "
                               f"{cfg['dp_window_h']} ч (сигнал закоксовывания; "
                               f"максимальная тревога при {cfg['dp_slope_crit_per_day'] * 100:.0f}%/сут)")
            dp_score = max(level_score, slope_score)

        # ---- температуры
        temp_score = 0.0
        for tag in cfg["tags"]["temperatures"]:
            col = self._resolve(tag, tel)
            if col is None or col not in base.med_by_name:
                missing.append(tag)
                continue
            z = (tel[col] - base.med_by_name[col]) / base.scale_by_name[col]
            s = _ramp(z, cfg["z_soft"], cfg["z_hard"])
            temp_score = max(temp_score, s)
            if s >= FACTOR_MIN_SCORE:
                factors.append(f"{names.get(tag, tag)} ({col}) = {tel[col]:.1f}: +{z:.1f}σ выше нормы")

        # ---- доля аномальных тегов
        idx = np.array([tel.get(n, np.nan) for n in base.names], dtype=float)
        valid = np.isfinite(idx)
        anomaly_share = float(np.mean(np.abs(idx[valid] - base.med[valid]) / base.scale[valid] > cfg["anomaly_z"])) \
            if valid.sum() >= 10 else None
        anomaly_score = _ramp(anomaly_share, cfg["anomaly_share_soft"], cfg["anomaly_share_hard"]) \
            if anomaly_share is not None else 0.0
        if anomaly_score >= FACTOR_MIN_SCORE:
            factors.append(f"Аномальны {anomaly_share:.0%} тегов телеметрии (|z| > {cfg['anomaly_z']:g}) — "
                           f"возможен сбой КИП или смена режима")

        # ---- нагрузка
        load_score = _ramp(load_rel, cfg["load_soft"], cfg["load_hard"]) if load_rel is not None else 0.0
        if load_score >= FACTOR_MIN_SCORE:
            factors.append(f"Перегрузка по сырью: {load_rel:.0%} от медианной загрузки")

        if missing:
            factors.append("Нет данных по тегам оборудования: " + ", ".join(missing))

        # ---- свёртка
        weighted = (w["dp"] * dp_score + w["temperature"] * temp_score
                    + w["anomaly"] * anomaly_score + w["load"] * load_score)
        severity = _clip01(max(weighted, cfg["hard_floor_k"] * max(dp_score, temp_score)))

        raw = (RiskClass.critical if severity >= cfg["critical_threshold"]
               else RiskClass.warning if severity >= cfg["warning_threshold"] else RiskClass.normal)
        self._crit_streak = self._crit_streak + 1 if raw == RiskClass.critical else 0
        cls = raw
        if raw == RiskClass.critical and self._crit_streak < int(cfg["critical_persistence"]):
            cls = RiskClass.warning
            factors.append(f"Критический уровень {self._crit_streak} из {cfg['critical_persistence']} "
                           f"циклов подряд — эскалация после подтверждения")
        if not factors:
            factors.append("Режим в пределах нормы по ΔP реактора, температурам, нагрузке и качеству данных"
                           if cls == RiskClass.normal else
                           f"Умеренное отклонение сразу по нескольким показателям (severity={severity:.2f})")
        return result(severity, cls, factors, slope=slope_out, anomaly=anomaly_share)

    def _dp_slope(self, ts: datetime, base: _Baseline, dp_col: str) -> tuple[float | None, float]:
        """Относительный тренд нормированного ΔP, доля/сутки, и его score."""
        cfg = self.cfg
        t_min = ts.timestamp() - cfg["dp_window_h"] * 3600
        feed_col = self._resolve(cfg["tags"]["feed"], self._rows[-1][1]) if self._rows else None
        xs: list[float] = []
        ys: list[float] = []
        for t, r in reversed(self._rows):
            tt = t.timestamp()
            if tt < t_min:
                break
            dp = r.get(dp_col)
            if dp is None or not np.isfinite(dp):
                continue
            lr = 1.0
            if feed_col and base.feed_med:
                f = r.get(feed_col)
                if f is None or not np.isfinite(f) or f < cfg["downtime_load_ratio"] * base.feed_med:
                    continue
                lr = f / base.feed_med
            y = float(self._normalize_dp(np.array([dp]), np.array([lr]))[0])
            if np.isfinite(y):
                xs.append((tt - t_min) / 86400.0)   # сутки
                ys.append(y)
        if len(xs) < int(cfg["dp_min_points"]) or (max(xs) - min(xs)) * 24 < 0.5 * cfg["dp_window_h"]:
            return None, 0.0
        if not base.dp_med:
            return None, 0.0
        slope_per_day = float(np.polyfit(xs, ys, 1)[0])
        rel = slope_per_day / abs(base.dp_med)
        return rel, _clip01(rel / cfg["dp_slope_crit_per_day"]) if rel > 0 else 0.0
