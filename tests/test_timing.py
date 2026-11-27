import datetime
import pandas as pd
import pytest
from windai.data.timing import (TZ, issue_date_local, target_hours_utc, target_lead_times, info_cutoff,
                                last_power_limit, fold_of)


def utc(s):
    return pd.Timestamp(s, tz="UTC")


# D+2 00-23 local: 09Z issues need lt 38-61 in winter (CET) and 37-60 in summer (CEST);
# 12Z issues need lt 35-58 and 34-57 (C1)
@pytest.mark.parametrize("issue, first, last", [
    ("2024-01-15 09:00", 38, 61),
    ("2024-07-15 09:00", 37, 60),
    ("2021-01-15 12:00", 35, 58),
    ("2021-07-15 12:00", 34, 57),
])
def test_lead_times(issue, first, last):
    lts = target_lead_times(utc(issue))
    assert lts[0] == first and lts[-1] == last and len(lts) == 24


# The target day starts at local midnight of D+2 and runs hour by hour
def test_target_hours_are_local_day():
    hours = target_hours_utc(utc("2024-01-15 09:00")).tz_convert(TZ)
    assert hours[0] == pd.Timestamp("2024-01-17 00:00", tz=TZ)
    assert hours[-1] == pd.Timestamp("2024-01-17 23:00", tz=TZ)


# DST on D+2: 31 Mar 2024 has 23 local hours, 27 Oct 2024 has 25 (D10)
def test_dst_days():
    assert len(target_hours_utc(utc("2024-03-29 09:00"))) == 23
    assert len(target_hours_utc(utc("2024-10-25 09:00"))) == 25
    autumn = target_hours_utc(utc("2024-10-25 09:00"))
    assert autumn.is_unique                  # the repeated local 02:00 is two distinct UTC hours


# Information set (D6): 09Z on local day D for both regimes; power labelled up to 08Z may be used
@pytest.mark.parametrize("issue", ["2024-01-15 09:00", "2021-01-15 12:00"])
def test_info_cutoff(issue):
    day = issue[:10]
    assert info_cutoff(utc(issue)) == utc(f"{day} 09:00")
    assert last_power_limit(utc(issue)) == utc(f"{day} 08:00")


def test_issue_date_is_local():
    assert issue_date_local(utc("2024-01-15 23:30")) == datetime.date(2024, 1, 16)


# Folds by local issue date (D2)
def test_fold_of():
    dates = [datetime.date(2023, 3, 31), datetime.date(2023, 4, 1), datetime.date(2024, 4, 1), datetime.date(2019, 1, 1)]
    assert list(fold_of(dates)) == ["train", "val", "test", None]
