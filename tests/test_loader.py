import math

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.loader import (
    DataPreparer,
    # DataPreparer methods used via instance
    _mask_min_duration,
    frozen_runs,
    robust_z,
)


def test_robust_z_variable_and_constant():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 10, 10, 10, 10]})
    z = robust_z(df)
    # variable column should have median approximately 0
    assert pytest.approx(0.0, abs=1e-12) == z["a"].median()
    # constant column becomes NaN due to zero scale in implementation
    assert z["b"].isna().all()


def test_frozen_runs_detects_long_runs_and_ignores_nans():
    s = pd.Series([1, 1, 1, 2, 2, 2, np.nan, 3, 3])
    res = frozen_runs(s, min_points=3)
    # first six points are runs of length >=3, nans and short runs are False
    expected = [True, True, True, True, True, True, False, False, False]
    assert res.tolist() == expected


def test_mask_min_duration_filters_short_spikes():
    mask = pd.Series([False, True, True, False, True, True, True, False])
    out = _mask_min_duration(mask, min_points=3)
    assert out.tolist() == [False, False, False, False, True, True, True, False]


def test_drop_service_columns_removes_unnamed_and_constant():
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=3),
        "Unnamed: 0": [1, 2, 3],
        "const": [5, 5, 5],
        "val": [1, 2, 3],
    })
    res = DataPreparer._drop_service_columns(df)
    assert "Unnamed: 0" not in res.columns
    assert "const" not in res.columns
    assert "val" in res.columns


def test_normalize_grid_inserts_missing_timestamps_and_on_grid_flag():
    prep = DataPreparer()
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2021-01-01 00:00"),
            pd.Timestamp("2021-01-01 00:15"),  # off-grid for 10min frequency
            pd.Timestamp("2021-01-01 00:20"),
        ],
        "val": [1.0, np.nan, 3.0],
    })
    out = prep.normalize_grid(df)
    # index should be 00:00, 00:10, 00:20
    assert out.index[0] == pd.Timestamp("2021-01-01 00:00")
    assert out.index[1] == pd.Timestamp("2021-01-01 00:10")
    assert out.index[2] == pd.Timestamp("2021-01-01 00:20")
    # on_grid True only for rows that had any non-NaN original values
    assert out.loc[pd.Timestamp("2021-01-01 00:00"), "on_grid"] == True
    assert out.loc[pd.Timestamp("2021-01-01 00:10"), "on_grid"] == False
    assert out.loc[pd.Timestamp("2021-01-01 00:20"), "on_grid"] == True


def test_attach_lims_merges_last_result_and_age_minutes():
    prep = DataPreparer()
    # left dataframe (telemetry) with datetime index
    left = pd.DataFrame({"date": [
        pd.Timestamp("2021-01-01 00:00"),
        pd.Timestamp("2021-01-01 00:10"),
        pd.Timestamp("2021-01-01 00:20"),
    ]}).set_index("date")

    # long LIMS-like table: dates and values for indicator 'Mg.Sulfur'
    long = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-01 00:05"), pd.Timestamp("2021-01-01 00:15")],
        "value": [1.0, 2.0],
        "installation": ["inst", "inst"],
        "indicator": ["Mg.Sulfur", "Mg.Sulfur"],
    })

    attached = prep.attach_lims(left, long, {"sulfur": "Mg.Sulfur"})
    # after attach_lims index is date
    assert "lims_sulfur" in attached.columns
    assert math.isnan(attached.iloc[0]["lims_sulfur"])  # before any lims
    assert attached.iloc[1]["lims_sulfur"] == 1.0
    assert attached.iloc[2]["lims_sulfur"] == 2.0
    # ages in minutes should be 5 and 5 for the last two rows
    assert attached.iloc[1]["lims_sulfur_age_min"] == pytest.approx(5.0)
    assert attached.iloc[2]["lims_sulfur_age_min"] == pytest.approx(5.0)
