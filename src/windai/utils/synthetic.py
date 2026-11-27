import numpy as np
import pandas as pd
from windai.data.timing import TZ, issue_date_local, target_hours_utc, target_lead_times, last_power_limit, fold_of

ZONES = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
N_MEMBERS = 15


# Stylised power curve: capacity factor as a function of wind speed (cut-in ~3 m/s, rated ~12 m/s).
def _power_curve(ws):
    return 1 / (1 + np.exp(-(ws - 7.5) / 1.3))


# A small fake version of the build_dataset() table (windai.data.build + p_00..p_14), for tests and for running
# the experiment script without the real caches. Same columns and conventions as the real table: one row per
# (issue_time, zone, UTC target_time), D+2 in local time (23/24/25 hours), folds by local issue date, members
# p_XX and the target cf in capacity factor, power and capacity in MW, capacity fixed at issue time (D1).
# Daily 09Z issues. Numbers are made up; only the schema and rough behaviour are realistic: the ensemble is
# under-dispersive and biased, power is bounded by [0, capacity] and capacity grows over time.
def make_synthetic_dataset(n_issues=365, start="2022-10-01", seed=0):
    rng = np.random.default_rng(seed)
    issues = pd.date_range(start, periods=n_issues, freq="D", tz="UTC") + pd.Timedelta(hours=9)
    capacity0 = {"ELSPOT NO1": 400.0, "ELSPOT NO2": 900.0, "ELSPOT NO3": 1700.0, "ELSPOT NO4": 1100.0}

    # Columns are collected as plain arrays and turned into one DataFrame at the end (much faster than one
    # small DataFrame per issue and zone)
    cols = {k: [] for k in ["issue_time", "issue_date", "issue_hour", "zone", "target_time", "lt",
                            "last_power", "last_power_time", "power", "capacity_mw", "cf", "last_cf"]}
    ws_all, p_all = [], []
    for issue in issues:
        hours = target_hours_utc(issue)
        lts = target_lead_times(issue)
        n = len(hours)
        season = 2.0 * np.cos(2 * np.pi * (hours.dayofyear.to_numpy() - 15) / 365)
        for zone in ZONES:
            weather = rng.gamma(4.0, 1.8)
            ws_true = np.clip(weather + season + rng.normal(0, 1.0, n), 0, None)
            ws_members = np.clip(ws_true[:, None] + 0.5 + rng.normal(0, 1.0, (n, N_MEMBERS)), 0, None)
            capacity = capacity0[zone] * (1 + 0.15 * (issue - issues[0]).days / 365)
            cf_true = np.clip(_power_curve(ws_true) + rng.normal(0, 0.05, n), 0, 1)
            last_cf = float(np.clip(_power_curve(weather) + rng.normal(0, 0.1), 0, 1))
            for key, value in [("issue_time", issue), ("issue_date", issue_date_local(issue)),
                               ("issue_hour", issue.hour), ("zone", zone), ("last_power", last_cf * capacity),
                               ("last_power_time", last_power_limit(issue)), ("capacity_mw", capacity),
                               ("last_cf", last_cf)]:
                cols[key].append(np.repeat(np.array([value], dtype=object), n))
            cols["target_time"].append(hours)
            cols["lt"].append(lts)
            cols["power"].append(cf_true * capacity)
            cols["cf"].append(cf_true)
            ws_all.append(ws_members)
            p_all.append(_power_curve(ws_members))

    df = pd.DataFrame({k: np.concatenate([np.asarray(v, dtype=object) if k != "target_time" else v.to_numpy()
                                          for v in vs]) for k, vs in cols.items()})
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    df["last_power_time"] = pd.to_datetime(df["last_power_time"], utc=True)
    for c in ["issue_hour", "lt"]:
        df[c] = df[c].astype(int)
    for c in ["last_power", "power", "capacity_mw", "cf", "last_cf"]:
        df[c] = df[c].astype(float)
    df["target_hour_utc"] = df["target_time"].dt.hour
    df["target_hour_local"] = df["target_time"].dt.tz_convert(TZ).dt.hour
    df["capacity_cov_mw"] = 0.99 * df["capacity_mw"]
    df["fold"] = fold_of(df["issue_date"])
    members = pd.DataFrame(np.hstack([np.vstack(ws_all), np.vstack(p_all)]),
                           columns=[f"ws_{i:02d}" for i in range(N_MEMBERS)] + [f"p_{i:02d}" for i in range(N_MEMBERS)])
    df = pd.concat([df, members], axis=1)
    return df.sort_values(["issue_time", "zone", "target_time"], ignore_index=True)
