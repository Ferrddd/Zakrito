"""Модель нормального поведения ΔP: обучение на синтетике и работа агента поверх неё."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.reliability.agent import RuleBasedReliabilityAgent
from src.agents.reliability.nbm import NormalBehaviorModel
from src.agents.reliability.train import train
from src.schemas.agents_schemas import DataFreshness, ProcessSnapshot, RiskClass
from src.utils.config import load_config

STEP = pd.Timedelta("10min")


def physics(
    feed: Any,
    t5: Any,
    p8: Any,
    f19: Any,
    f15: Any,
    p24: Any,
    f9: Any,
) -> Any:
    """ΔP слоя = f(нагрузка, давление, квенч, ВСГ, температуры) — то, что должна выучить NBM."""
    return (
        0.5
        * (feed / 150) ** 1.2
        * (1 + 0.002 * (t5 - 350) + 0.001 * (p8 - 340))
        * (1 + 0.1 * (f15 - 20) / 20)
        * (1 - 0.05 * (f19 - 30) / 30)
        * (1 + 0.03 * (p24 - 10) / 10)
        + 0.0 * f9
    )


def make_df(
    n: int = 7000,
    seed: int = 0,
    coking_from: int | None = None,
    coking_per_day: float = 0.0,
) -> pd.DataFrame:
    """Синтетика как в реальных данных: у ΔP есть СКРЫТЫЙ уровень (меняется скачками раз в ~3 суток
    в 0.6-1.6 раза), поверх которого режим (нагрузка, температуры...) даёт быстрые изменения."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-05-01", periods=n, freq="10min")
    t = np.arange(n)
    feed = 150 + 15 * np.sin(t / 40) + rng.normal(0, 1.0, n)
    t5 = 350 + 4 * np.sin(t / 55 + 1) + rng.normal(0, 0.5, n)
    p8 = 340 + 4 * np.cos(t / 65) + rng.normal(0, 0.5, n)
    f19 = 30 + 2 * np.sin(t / 90) + rng.normal(0, 0.2, n)
    f15 = 20 + 3 * np.cos(t / 70) + rng.normal(0, 0.3, n)
    p24 = 10 + 1.0 * np.sin(t / 80 + 2) + rng.normal(0, 0.1, n)
    f9 = 5 + rng.normal(0, 0.2, n)
    levels = rng.uniform(0.6, 1.6, n // 500 + 2)
    hidden = np.repeat(levels, 500)[:n]
    w10 = (
        2.5
        * hidden
        * physics(feed, t5, p8, f19, f15, p24, f9)
        * (1 + rng.normal(0, 0.01, n))
    )
    if coking_from is not None:
        days = np.clip((t - coking_from) / 144.0, 0, None)
        w10 = w10 * (1 + coking_per_day * days)
    return pd.DataFrame(
        {
            "T11_hdt": feed,
            "T5": t5,
            "P8": p8,
            "F19_hdt": f19,
            "F15": f15,
            "P24": p24,
            "F9_hdt": f9,
            "W10": w10,
            "is_valid": True,
            "hours_since_block_start": 500.0,
        },
        index=idx,
    )


def small_cfg(model_path: str) -> dict[str, Any]:
    cfg = load_config("src/agents/reliability/config.yaml")
    cfg["nbm"].update(
        {
            "holdout_size": "10D",
            "model_path": model_path,
            "prefill_points": 200,
            "params": {
                **cfg["nbm"]["params"],
                "n_estimators": 200,
                "min_child_samples": 50,
            },
        }
    )
    return cfg


def hidden_level(df: pd.DataFrame, ts: Any) -> float:
    """Скрытый уровень ΔП в момент ts, восстановленный из исходной синтетики (для теста «нагрузка вверх»)."""
    row = df.loc[ts]
    base = 2.5 * physics(
        row["T11_hdt"],
        row["T5"],
        row["P8"],
        row["F19_hdt"],
        row["F15"],
        row["P24"],
        row["F9_hdt"],
    )
    return float(row["W10"] / base)


def snap(ts: Any, row: pd.Series) -> ProcessSnapshot:
    return ProcessSnapshot(
        timestamp=pd.Timestamp(ts).to_pydatetime(),
        telemetry={
            k: float(v) for k, v in row.items() if k not in ("is_valid",)
        },
        data_freshness=DataFreshness(),
    )


def trained_agent(tmp: str) -> tuple[RuleBasedReliabilityAgent, dict[str, Any]]:
    cfg = small_cfg(str(Path(tmp) / "nbm.joblib"))
    nbm = train(make_df(), cfg)
    nbm.save(cfg["nbm"]["model_path"])
    return RuleBasedReliabilityAgent(cfg), cfg


def test_nbm_learns_physics_and_saves() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = small_cfg(str(Path(tmp) / "nbm.joblib"))
        nbm = train(make_df(), cfg)
        assert nbm.metrics["holdout_skill_vs_persistence"] > 0.2  # лучше наивного «ΔП не менялся»
        assert nbm.metrics["holdout_r2"] > 0.8
        nbm.save(cfg["nbm"]["model_path"])
        loaded = NormalBehaviorModel.load(cfg["nbm"]["model_path"])
        assert loaded is not None and loaded.base_cols == nbm.base_cols
        assert not any("hours_since" in c for c in loaded.columns)  # время в признаки не входит


def test_agent_uses_nbm_when_model_present() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = trained_agent(tmp)
        assert agent.dp_mode == "nbm"


def test_load_surge_is_not_coking() -> None:
    """Рост нагрузки поднимает ΔP «по физике» — остаток нулевой, тревоги нет."""
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = trained_agent(tmp)
        df = make_df(seed=1)
        agent.warm_up(df.iloc[:6500])
        res = None
        for ts, row in df.iloc[6500:6560].iterrows():
            r = row.copy()
            r["T11_hdt"] = 178.0  # нагрузка выше всего, что видела модель
            r["W10"] = 2.5 * hidden_level(df, ts) * physics(
                178.0,
                r["T5"],
                r["P8"],
                r["F19_hdt"],
                r["F15"],
                r["P24"],
                r["F9_hdt"],
            )
            res = agent.assess(snap(ts, r))
        assert res is not None
        assert res.risk_class == RiskClass.normal, res.limiting_factors


def test_fast_unexplained_dp_growth_is_detected() -> None:
    """Быстрый рост ΔП (~100%/сут) сверх режима: остаток модели за 24 ч устойчиво выше нуля."""
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = trained_agent(tmp)
        df = make_df(seed=2, coking_from=6500, coking_per_day=1.0)
        agent.warm_up(df.iloc[:6500])
        classes, factors = [], []
        for ts, row in df.iloc[6500:6900].iterrows():  # ~2.8 суток
            res = agent.assess(snap(ts, row))
            classes.append(res.risk_class)
            factors += res.limiting_factors
        assert RiskClass.warning in classes or RiskClass.critical in classes
        assert any("ожидаемого по режиму" in f for f in factors)
        assert res.dp_trend_slope is not None and res.dp_trend_slope > 0.3


def test_hidden_level_jumps_do_not_raise_alarms() -> None:
    """Скачки скрытого уровня ΔП (как в реальных данных) — не закоксовывание: тревог быть не должно."""
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = trained_agent(tmp)
        df = make_df(seed=5)
        agent.warm_up(df.iloc[:6000])
        classes = [
            agent.assess(snap(ts, row)).risk_class
            for ts, row in df.iloc[6000:6900].iterrows()
        ]
        assert classes.count(RiskClass.normal) >= 0.95 * len(classes)


def test_missing_model_feature_falls_back_to_rules() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent, _ = trained_agent(tmp)
        df = make_df(seed=3)
        agent.warm_up(df.iloc[:6500])
        ts, row = df.index[6500], df.iloc[6500].drop("F15")
        res = agent.assess(snap(ts, row))
        assert any("Модель ΔP недоступна" in f for f in res.limiting_factors)