import numpy as np
import pandas as pd
import pytest
from windai.data.build import build_base_table, build_wind_cube, check_base_table
from windai.data.parks import parks_table, capacity_at
from windai.data.timing import last_power_limit

# Real data, a handful of issues. Building the training cube streams all of met_forecast.parquet once (~1 min).
ISSUES = {
    "2021-01-15 12:00": ("train", 24),    # 12Z, winter
    "2024-03-29 09:00": ("val", 23),      # D+2 = 31 Mar 2024, spring DST
    "2024-10-25 09:00": ("test", 25),     # D+2 = 27 Oct 2024, autumn DST
    "2023-06-13 12:00": (None, 0),        # truncated 12Z issue (max lt 36): dropped
}


@pytest.fixture(scope="module")
def table():
    issues = [pd.Timestamp(t, tz="UTC") for t in ISSUES]
    return build_base_table("train", cube=build_wind_cube("train", issues=issues))


def test_rows_per_issue(table):
    counts = table.groupby("issue_time").size()
    for t, (_, hours) in ISSUES.items():
        assert counts.get(pd.Timestamp(t, tz="UTC"), 0) == 4 * hours


def test_folds(table):
    folds = table.groupby("issue_time")["fold"].first()
    for t, (fold, hours) in ISSUES.items():
        if hours:
            assert folds[pd.Timestamp(t, tz="UTC")] == fold


def test_columns_and_features(table):
    for col in ["issue_time", "issue_date", "issue_hour", "fold", "zone", "target_time", "target_hour_utc",
                "target_hour_local", "lt", "last_power", "last_power_time", "last_cf", "capacity_mw",
                "capacity_cov_mw", "power", "cf"] + [f"ws_{m:02d}" for m in range(15)]:
        assert col in table.columns
    ws = table[[f"ws_{m:02d}" for m in range(15)]].to_numpy()
    assert np.isfinite(ws).all() and (ws >= 0).all() and (ws < 50).all()


# Capacity is the issue-time capacity of all parks; the capacity of parks with weather data is no larger
def test_capacity_at_issue_time(table):
    parks = parks_table()
    for (issue, zone), g in table.groupby(["issue_time", "zone"]):
        assert (g["capacity_mw"] == capacity_at(parks, zone, [issue])[0]).all()
    assert (table["capacity_cov_mw"] <= table["capacity_mw"]).all()


def test_last_power_within_information_set(table):
    limits = table["issue_time"].map(last_power_limit)
    assert (table["last_power_time"] <= limits).all()
    assert (table["last_power_time"] == limits).all()      # no gaps at these issues: exactly 08Z on D


def test_cf_plausible(table):
    cf = table["cf"].dropna()
    assert (cf >= 0).all() and (cf <= 1).all()


# The asserts catch a broken table
def test_check_catches_leak(table):
    broken = table.copy()
    broken.loc[0, "last_power_time"] = broken.loc[0, "issue_time"]          # after the 08Z limit
    with pytest.raises(AssertionError, match="after the cutoff"):
        check_base_table(broken)


def test_eval_split():
    issue = pd.Timestamp("2025-09-22 09:00", tz="UTC")
    table = build_base_table("eval", cube=build_wind_cube("eval", issues=[issue]))
    assert len(table) == 4 * 24 and set(table["fold"]) == {"windai"}
    assert table["power"].notna().all()
