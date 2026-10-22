import pytest
import pandas as pd
import datetime
import re
from windai.data.dataset_loader import load_power, valid_zones, target

def test_format():
    power_df = load_power()
    assert power_df.index.name == "time"
    assert isinstance(power_df.index, pd.DatetimeIndex)
    assert power_df.index.tz == datetime.timezone.utc
    assert list(power_df.columns.values) == valid_zones
    assert len(power_df.index) == 46011

def test_zone_filter():
    power_df = load_power(zone="ELSPOT NO3")
    assert list(power_df.columns.values) == ['ELSPOT NO3']
    assert power_df.index.tz == datetime.timezone.utc
    
def test_invalid_zone():
    with pytest.raises(ValueError, match=re.escape(f"Please input a valid bidding zone ({str(valid_zones)}), or leave 'zone' blank")):
        load_power(zone="NO3")

def test_double_time_filter():
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-02")
    power_df = load_power(start=start, end=end)
    assert power_df.index.min() == start.tz_localize("UTC")
    assert power_df.index.max() == end.tz_localize("UTC") - pd.Timedelta(1, 'h')
    assert end.tz_localize("UTC") not in power_df.index
    assert len(power_df.index) == 24

def test_start_filter():
    start = pd.Timestamp("2024-02-01")
    power_df = load_power(start=start)
    control_df = load_power()
    control_df = control_df.loc[control_df.index >= start.tz_localize("UTC")]
    assert power_df.index.min() == start.tz_localize("UTC")
    assert power_df.index.max() == load_power().index.max()
    assert len(power_df.index) == len(control_df.index)
                          
def test_end_filter():
    end = pd.Timestamp("2024-03-01")
    power_df = load_power(end=end)
    control_df = load_power()
    control_df = control_df.loc[control_df.index < end.tz_localize("UTC")]
    assert power_df.index.max() == end.tz_localize("UTC") - pd.Timedelta(1, 'h')
    assert power_df.index.min() == load_power().index.min()
    assert len(power_df.index) == len(control_df.index)