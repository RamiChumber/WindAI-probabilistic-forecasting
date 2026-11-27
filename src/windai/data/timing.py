import datetime
import numpy as np
import pandas as pd

# Time helpers. Everything is UTC internally; Europe/Oslo local time appears only here.

TZ = "Europe/Oslo"

# Folds by local issue date D, [start, end)
FOLDS = {
    "train": (datetime.date(2020, 1, 1), datetime.date(2023, 4, 1)),
    "val": (datetime.date(2023, 4, 1), datetime.date(2024, 4, 1)),
    "test": (datetime.date(2024, 4, 1), datetime.date(2025, 4, 1)),
}


def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


# The issue day D: the local calendar date of the issue time
def issue_date_local(issue_time):
    return _utc(issue_time).tz_convert(TZ).date()


# UTC timestamps of D+2, 00:00-23:00 local time. Calendar-day arithmetic (DateOffset) in local time, so the
# target day has 23 hours on the spring DST day and 25 on the autumn one.
def target_hours_utc(issue_time):
    start = pd.Timestamp(issue_date_local(issue_time), tz=TZ) + pd.DateOffset(days=2)
    hours = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left")
    return hours.tz_convert("UTC")


# Lead times (hours after the issue time) of the target hours
def target_lead_times(issue_time):
    return ((target_hours_utc(issue_time) - _utc(issue_time)) / pd.Timedelta(hours=1)).astype(int).to_numpy()


# Information set: every issue, also a 12Z one, knows what is known at 09Z on its local day D, so persistence
# means the same in both issue regimes.
def info_cutoff(issue_time):
    return pd.Timestamp(issue_date_local(issue_time), tz="UTC") + pd.Timedelta(hours=9)


# The latest power label a forecast may use: the hour labelled cutoff - 1 h is complete at the cutoff under
# either timestamp convention, [t, t+1) or (t-1, t].
def last_power_limit(issue_time):
    return info_cutoff(issue_time) - pd.Timedelta(hours=1)


# Fold of each local issue date; dates outside every fold get None
def fold_of(dates):
    dates = pd.Series(list(dates))
    out = np.full(len(dates), None, dtype=object)
    for name, (start, end) in FOLDS.items():
        out[((dates >= start) & (dates < end)).to_numpy()] = name
    return out
