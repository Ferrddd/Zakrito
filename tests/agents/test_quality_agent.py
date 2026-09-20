"""Юнит-тесты агента качества БЕЗ данных и обученных моделей: фейковые модели и фичи.
Проверяют логику агента (режимы, отказы, what-if, статус ПАК), а не качество прогноза."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agents.quality.agent import ModeBundle, QualityAgent
from src.agents.quality.features_online import OnlineFeatureBuilder
from src.schemas.process_state import ProcessState

TARGET = "24-2000:Mg.Sulfur"
T0 = datetime(2026, 6, 1, 0, 0)


def fake_engineered(df: pd.DataFrame) -> pd.DataFrame:
    return df


def fake_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...], windows: tuple[int, ...],
              n_jobs: int = 1) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        for lag in lags:
            df[f"{c}__lag{lag}"] = df[c].shift(lag)
        for w in windows:
            df[f"{c}__mean{w}"] = df[c].rolling(w, min_periods=1).mean()
    return df


class FakeModel:
    feature_cols = ["T5_hdt", "T5_hdt__lag3"]

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        base = X["T5_hdt"].to_numpy() / 100.0   # T5=350 -> 3.5 ppm
        return {0.1: base - 1, 0.5: base, 0.9: base + 1}

    def explain(self, X: pd.DataFrame, top_n: int = 5) -> pd.Series:
        return pd.Series({"T5_hdt": 0.5})


class FakeClf:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return (X["T5_hdt"].to_numpy() > 500).astype(float)   # T5>500 -> «нарушение»


def make_agent(mode_list=("with_analyzer", "blind"), delta=0.0) -> QualityAgent:
    cfg = SimpleNamespace(target_tag=TARGET, target_unit="ppm", target_limit=10.0, grid_freq="10min")
    builder = OnlineFeatureBuilder(["T5_hdt", "T5_hdt__lag3"], ["T5_hdt", "T5_hdt__lag3"], [3], [6],
                                   feature_fns=(fake_engineered, fake_lags))
    bundles = [ModeBundle(3, m, FakeModel(), FakeClf(), 0.5, builder, delta) for m in mode_list]
    return QualityAgent(cfg, bundles, "10min", max_history_points=50)


def state(i: int, t5: float = 350.0, bad=0.0, frozen=0.0, age=5.0) -> ProcessState:
    tags = {"T5_hdt": t5, f"{TARGET}__bad": bad, f"{TARGET}__frozen": frozen, f"{TARGET}__age_min": age}
    return ProcessState(timestamp=T0 + timedelta(minutes=10 * i), tags=tags)


def warm(agent: QualityAgent, n: int = 20) -> None:
    for i in range(n):
        agent.observe(state(i))


def test_warmup_abstains():
    a = make_agent()
    r = a.evaluate(state(0))
    assert r.abstain and "истории" in r.reason


def test_with_analyzer_mode_and_prediction():
    a = make_agent()
    warm(a)
    r = a.evaluate(state(20))
    assert not r.abstain and r.predictions[0].source_model == "with_analyzer"
    assert r.predictions[0].p50 == pytest.approx(3.5)
    assert not r.predictions[0].alert and r.drivers[0].tag == "T5_hdt"


def test_blind_when_pak_bad_and_nan_flags_do_not_crash():
    a = make_agent()
    warm(a)
    r = a.evaluate(state(20, bad=1.0))
    assert r.data_quality.pak_status == "out_of_range" and r.predictions[0].source_model == "blind"
    r = a.evaluate(state(21, bad=float("nan")))       # NaN не должен считаться «frozen»/«ok»
    assert r.data_quality.pak_status == "missing"


def test_frozen_status():
    a = make_agent()
    warm(a)
    assert a.evaluate(state(20, frozen=1.0)).data_quality.pak_status == "frozen"


def test_stale_pak_abstains():
    a = make_agent()
    warm(a)
    r = a.evaluate(state(20, bad=1.0, age=2000.0))
    assert r.abstain and "устарело" in r.reason


def test_what_if_changes_prediction_and_does_not_persist():
    a = make_agent()
    warm(a)
    base = a.evaluate(state(20)).predictions[0]
    hot = a.evaluate(state(20), overrides={"T5_hdt": 600.0}).predictions[0]
    assert hot.alert and hot.p50 == pytest.approx(6.0) and not base.alert
    again = a.evaluate(state(20)).predictions[0]      # override не должен «прилипнуть» к буферу
    assert again.p50 == pytest.approx(3.5)


def test_what_if_unknown_tag_abstains():
    a = make_agent()
    warm(a)
    assert a.evaluate(state(20), overrides={"NOPE": 1.0}).abstain


def test_conformal_delta_widens_interval_and_clips_at_zero():
    a = make_agent(delta=0.5)
    warm(a)
    p = a.evaluate(state(20)).predictions[0]
    assert p.p10 == pytest.approx(2.0) and p.p90 == pytest.approx(5.0)


def test_missing_tags_abstain():
    a = make_agent()
    warm(a)
    s = ProcessState(timestamp=T0 + timedelta(minutes=200),
                     tags={f"{TARGET}__bad": 0.0, f"{TARGET}__frozen": 0.0})   # нет T5_hdt
    assert a.evaluate(s).abstain


def test_capabilities():
    caps = make_agent().capabilities()
    assert caps.horizons_min == [30] and "T5_hdt" in caps.required_tags


def test_block_id_forward_filled_over_gaps():
    from src.agents.quality.features_online import to_regular_grid
    idx = pd.to_datetime(["2026-06-01 00:00", "2026-06-01 00:30"])
    df = pd.DataFrame({"block_id": [7.0, 7.0], "x": [1.0, 2.0]}, index=idx)
    out = to_regular_grid(df, "10min")
    assert len(out) == 4 and out["block_id"].tolist() == [7.0] * 4 and out["x"].isna().sum() == 2