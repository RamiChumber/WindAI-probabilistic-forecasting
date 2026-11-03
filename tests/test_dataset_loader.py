import pytest
import pandas as pd
import datetime
import re
from windai.data.dataset_loader import (load_power, load_parks, load_met_nowcast, load_met_forecast,
                                        _filter_time, VALID_ZONES, N_MEMBERS)

# Tests for _filter_time()
def test_filter_time_inputs():
    df = pd.DataFrame({"x": range(48)}, index=pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"))
    naive = _filter_time(df, pd.Timestamp("2024-01-01 06:00"), pd.Timestamp("2024-01-01 12:00"))
    string = _filter_time(df, "2024-01-01 06:00", "2024-01-01 12:00")
    aware = _filter_time(df, pd.Timestamp("2024-01-01 07:00", tz="Europe/Oslo"),
                         pd.Timestamp("2024-01-01 13:00", tz="Europe/Oslo"))
    assert naive.equals(string)
    assert naive.equals(aware)
    assert naive.index.min() == pd.Timestamp("2024-01-01 06:00", tz="UTC")
    assert naive.index.max() == pd.Timestamp("2024-01-01 11:00", tz="UTC")
    assert len(naive.index) == 6

# Tests for load_power()
def test_power_format():
    power_df = load_power()
    assert power_df.index.name == "time"
    assert isinstance(power_df.index, pd.DatetimeIndex)
    assert power_df.index.tz == datetime.timezone.utc
    assert list(power_df.columns.values) == VALID_ZONES
    assert len(power_df.index) == 46011

def test_power_zone_filter():
    power_df = load_power(zone="ELSPOT NO3")
    assert list(power_df.columns.values) == ['ELSPOT NO3']
    assert power_df.index.tz == datetime.timezone.utc

def test_power_invalid_zone():
    with pytest.raises(ValueError, match=re.escape(f"Please input a valid bidding zone ({str(VALID_ZONES)}), or leave 'zone' blank")):
        load_power(zone="NO3")

def test_power_double_time_filter():
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-02")
    power_df = load_power(start=start, end=end)
    assert power_df.index.min() == start.tz_localize("UTC")
    assert power_df.index.max() == end.tz_localize("UTC") - pd.Timedelta(1, 'h')
    assert end.tz_localize("UTC") not in power_df.index
    assert len(power_df.index) == 24

def test_power_start_filter():
    start = pd.Timestamp("2024-02-01")
    power_df = load_power(start=start)
    control_df = load_power()
    control_df = control_df.loc[control_df.index >= start.tz_localize("UTC")]
    assert power_df.index.min() == start.tz_localize("UTC")
    assert power_df.index.max() == load_power().index.max()
    assert len(power_df.index) == len(control_df.index)

def test_power_end_filter():
    end = pd.Timestamp("2024-03-01")
    power_df = load_power(end=end)
    control_df = load_power()
    control_df = control_df.loc[control_df.index < end.tz_localize("UTC")]
    assert power_df.index.max() == end.tz_localize("UTC") - pd.Timedelta(1, 'h')
    assert power_df.index.min() == load_power().index.min()
    assert len(power_df.index) == len(control_df.index)

def test_power_eval_split():
    power_df = load_power(split="eval")
    assert power_df.index.name == "time"
    assert power_df.index.tz == datetime.timezone.utc
    assert list(power_df.columns.values) == VALID_ZONES
    assert list(load_power(zone="ELSPOT NO3", split="eval").columns.values) == ['ELSPOT NO3']

# Tests for load_parks()
def test_park_format():
    park_df = load_parks()
    assert len(park_df.index) == 76
    assert list(park_df.columns) == ["bidding_area", "substation_name", "operating_power_max", "prod_start_new", "eic_code"]
    assert park_df["prod_start_new"].dt.tz == datetime.timezone.utc

def test_park_valid_zones():
    park_df = load_parks()
    assert park_df["bidding_area"].isin(VALID_ZONES).all()

def test_park_zone_filter():
    park_df = load_parks(zone="ELSPOT NO3")
    control_df = load_parks()
    assert (park_df["bidding_area"] == "ELSPOT NO3").all()
    assert len(park_df.index) == len(control_df.loc[control_df["bidding_area"] == "ELSPOT NO3"])

def test_park_invalid_zone():
    with pytest.raises(ValueError, match=re.escape(f"Please input a valid bidding zone ({VALID_ZONES}), or leave 'zone' blank")):
        load_parks("NO3")

# Tests for load_met_nowcast()
def test_nowcast_format():
    nowcast_df = load_met_nowcast()
    assert nowcast_df.index.name == "time"
    assert isinstance(nowcast_df.index, pd.DatetimeIndex)
    assert nowcast_df.index.tz == datetime.timezone.utc
    assert "windpark" in nowcast_df.columns
    assert len(nowcast_df.index) == 2806427
    assert nowcast_df["windpark"].nunique() == 61

def test_nowcast_canonical_names():
    parks = set(load_met_nowcast()["windpark"].unique())
    assert "Tonst" not in parks and "Sørfj" not in parks
    assert parks <= set(load_parks()["substation_name"])

def test_nowcast_park_filter():
    nowcast_df = load_met_nowcast(windparks="Tonst. Vindpark")
    assert len(nowcast_df.index) > 0
    assert (nowcast_df["windpark"] == "Tonst. Vindpark").all()

def test_nowcast_several_parks():
    parks = ["Tonst. Vindpark", "Sørfj. Vindkraft", "Bessakerfjellet"]
    nowcast_df = load_met_nowcast(windparks=parks)
    assert set(nowcast_df["windpark"].unique()) == set(parks)

def test_nowcast_invalid_park():
    with pytest.raises(ValueError, match="Invalid windparks"):
        load_met_nowcast(windparks="Tonst")

def test_nowcast_eval_split():
    nowcast_df = load_met_nowcast(split="eval")
    assert nowcast_df.index.name == "time"
    assert nowcast_df.index.tz == datetime.timezone.utc
    assert not nowcast_df.reset_index().duplicated(subset=["windpark", "time"]).any()

# Tests for load_met_forecast()
def test_forecast_format():
    forecast_df = load_met_forecast(variables="ws10m", start="2024-01-01", end="2024-01-03")
    members = [f"ws10m_{i:02d}" for i in range(N_MEMBERS)]
    assert list(forecast_df.columns) == ["windpark", "time_ref", "time", "lt"] + members
    assert forecast_df["time_ref"].dt.tz == datetime.timezone.utc
    assert forecast_df["time"].dt.tz == datetime.timezone.utc

def test_forecast_time_filter():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-03", tz="UTC")
    forecast_df = load_met_forecast(variables="ws10m", start=start, end=end)
    assert forecast_df["time_ref"].min() >= start
    assert forecast_df["time_ref"].max() < end
    assert forecast_df["time_ref"].nunique() == 2

def test_forecast_park_filter():
    forecast_df = load_met_forecast(windparks="Sørfj. Vindkraft", variables="ws10m", start="2024-01-01", end="2024-01-03")
    assert len(forecast_df.index) > 0
    assert (forecast_df["windpark"] == "Sørfj. Vindkraft").all()

def test_forecast_invalid_park():
    with pytest.raises(ValueError, match="Invalid windparks"):
        load_met_forecast(windparks="Sørfj", variables="ws10m", start="2024-01-01", end="2024-01-03")

def test_forecast_invalid_variable():
    with pytest.raises(ValueError, match="Invalid variables"):
        load_met_forecast(variables="ws100m")

def test_forecast_eval_split():
    eval_df = load_met_forecast(variables="ws10m", start="2025-06-01", end="2025-06-03", split="eval")
    train_df = load_met_forecast(variables="ws10m", start="2024-01-01", end="2024-01-02")
    assert list(eval_df.columns) == list(train_df.columns)
    assert eval_df["time_ref"].dt.tz == datetime.timezone.utc
    assert (eval_df["time_ref"].dt.hour == 9).all()
    assert eval_df["time_ref"].nunique() == 2
