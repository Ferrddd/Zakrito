"""Тесты агента надёжности на синтетике: нет закоксовывания / есть / скачок / простой / мало истории."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.agents.reliability.agent import RuleBasedReliabilityAgent
from src.schemas.agents_schemas import DataFreshness, ProcessSnapshot, RiskClass

FEED, TEMP5, TEMP8 = 150.0, 350.0, 340.0
EXTRA = [f"X{i}_hdt" for i in range(30)]   # «прочая» телеметрия для доли аномальных тегов


def make_df(n: int = 600, start: str = "2026-06-01", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="10min")
    d = {
        "T11_hdt": FEED + rng.normal(0, 1.0, n),
        "W10": 0.5 * (1 + rng.normal(0, 0.005, n)),
        "T5": TEMP5 + rng.normal(0, 0.5, n),
        "P8": TEMP8 + rng.normal(0, 0.5, n),
    }
    for c in EXTRA:
        d[c] = 100 + rng.normal(0, 2.0, n)
    return pd.DataFrame(d, index=idx)


def snap(ts: Any, row: dict[str, float]) -> ProcessSnapshot:
    return ProcessSnapshot(timestamp=pd.Timestamp(ts).to_pydatetime(), telemetry=row,
                           data_freshness=DataFreshness())


def base_row(**over: float) -> dict[str, float]:
    row = {"T11_hdt": FEED, "W10": 0.5, "T5": TEMP5, "P8": TEMP8, **{c: 100.0 for c in EXTRA}}
    row.update(over)
    return row


def warmed() -> tuple[RuleBasedReliabilityAgent, pd.Timestamp]:
    df = make_df()
    agent = RuleBasedReliabilityAgent.load()
    agent.warm_up(df)
    return agent, df.index[-1]


def test_agent_is_real_not_stub() -> None:
    a = RuleBasedReliabilityAgent.load()
    assert a.is_stub is False and a.name == "reliability"


def test_not_enough_history_is_honest() -> None:
    a = RuleBasedReliabilityAgent.load()
    r = a.assess(snap("2026-06-01", base_row()))
    assert r.risk_class == RiskClass.normal and r.severity_index == 0.0
    assert "Мало истории" in r.limiting_factors[0]


def test_stable_regime_stays_normal() -> None:
    agent, last = warmed()
    rng = np.random.default_rng(1)
    classes = []
    for i in range(1, 60):
        row = base_row(T11_hdt=FEED + rng.normal(0, 1), W10=0.5 * (1 + rng.normal(0, 0.005)),
                       T5=TEMP5 + rng.normal(0, 0.5), P8=TEMP8 + rng.normal(0, 0.5))
        classes.append(agent.assess(snap(last + pd.Timedelta(minutes=10 * i), row)).risk_class)
    assert RiskClass.critical not in classes
    assert classes.count(RiskClass.normal) >= 55        # не создаём лишних тревог


def test_coking_trend_reaches_critical_and_explains_why() -> None:
    agent, last = warmed()
    res = None
    for i in range(1, 200):                              # ~33 часа роста ΔP на ~15%/сутки
        dp = 0.5 * (1 + 0.15 * (i * 10 / 1440))
        res = agent.assess(snap(last + pd.Timedelta(minutes=10 * i), base_row(W10=dp)))
    assert res is not None
    assert res.risk_class == RiskClass.critical and res.severity_index >= 0.8
    assert res.dp_trend_slope is not None and res.dp_trend_slope > 0.05
    assert any("растёт" in f or "выше нормы" in f for f in res.limiting_factors)


def test_single_temperature_spike_is_only_warning() -> None:
    agent, last = warmed()
    r = agent.assess(snap(last + pd.Timedelta(minutes=10), base_row(T5=TEMP5 + 15, P8=TEMP8 + 15)))
    assert r.risk_class == RiskClass.warning            # критический уровень не подтверждён (persistence)
    assert any("подряд" in f for f in r.limiting_factors)


def test_persistent_temperature_excursion_escalates() -> None:
    agent, last = warmed()
    cls = [agent.assess(snap(last + pd.Timedelta(minutes=10 * i),
                             base_row(T5=TEMP5 + 15, P8=TEMP8 + 15))).risk_class for i in range(1, 6)]
    assert cls[0] == RiskClass.warning and cls[-1] == RiskClass.critical


def test_downtime_is_flagged_not_escalated() -> None:
    agent, last = warmed()
    r = agent.assess(snap(last + pd.Timedelta(minutes=10), base_row(T11_hdt=2.0, W10=0.0, T5=200.0)))
    assert r.downtime_flag is True and r.risk_class == RiskClass.normal
    assert "простое" in r.limiting_factors[0]


def test_many_anomalous_tags_raise_anomaly_score() -> None:
    agent, last = warmed()
    row = base_row(**{c: 500.0 for c in EXTRA})
    r = agent.assess(snap(last + pd.Timedelta(minutes=10), row))
    assert r.downtime_score is not None and r.downtime_score > 0.5
    assert r.severity_index > 0.0 and any("Аномальны" in f for f in r.limiting_factors)


def test_missing_equipment_tag_is_reported_not_guessed() -> None:
    agent, last = warmed()
    row = base_row()
    del row["W10"]
    r = agent.assess(snap(last + pd.Timedelta(minutes=10), row))
    assert any("Нет данных по тегам оборудования" in f and "W10" in f for f in r.limiting_factors)
    assert r.risk_class == RiskClass.normal


def test_sentinel_307_is_treated_as_missing_not_anomaly() -> None:
    """307.0 — маркер обрыва канала в сырых КИП (max у большинства тегов): не измерение и не аномалия."""
    agent, last = warmed()
    r = agent.assess(snap(last + pd.Timedelta(minutes=10), base_row(**{c: 307.0 for c in EXTRA})))
    assert r.risk_class == RiskClass.normal
    assert not any("Аномальны" in f for f in r.limiting_factors)
