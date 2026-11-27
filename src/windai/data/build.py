"""Builds the model table: one row per (issue, zone, target hour). Decisions D1-D12 in notes/decisions.md.

Three products:
  wind cube   member wind ws10m per (issue, park, lt, member), lt 33-62    data/processed/wind_cube_{split}.npz
  base table  keys, targets, capacity, persistence, zone wind; no curve    data/processed/dataset_{split}.parquet
  p_00..p_14  zone CF per member from a fitted power curve, computed per fit (first_stage.add_power_ensemble)

Usage (repo root, tdt4501 env):
    python -m windai.data.build --split train
    python -m windai.data.build --split eval
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from windai.paths import get_data_root
from windai.data.dataset_loader import (load_power, load_met_forecast, MEPS_FORECAST, N_MEMBERS, VALID_ZONES)
from windai.data.parks import parks_table, weather_parks, capacity_at, capacity_matrix
from windai.data.timing import (TZ, issue_date_local, target_hours_utc, target_lead_times, last_power_limit,
                                fold_of)
from windai.data.first_stage import fit_power_curve, add_power_ensemble, zone_weighted
from windai.data.splits import FIT_BOUNDARY

PROCESSED = Path(__file__).resolve().parents[3] / "data" / "processed"
CUBE_LTS = np.arange(33, 63)          # D+2 +- 1 h for both regimes: 12Z needs 33-59, 09Z 36-62 (C1)
WS_COLS = [f"ws10m_{m:02d}" for m in range(N_MEMBERS)]


def _ns(times):
    """UTC DatetimeIndex in ns resolution, so indexes built from different sources compare equal."""
    times = pd.DatetimeIndex(times)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    return times.as_unit("ns")


# ---------------------------------------------------------------- wind cube

def build_wind_cube(split="train", issues=None):
    """Member wind per (issue, park, lt, member) as float32, NaN where the forecast has no value.

    `met_forecast.parquet` (4.7 GB, 7 row groups of ~1M rows) is read one column of one row group at a time:
    pyarrow decodes whole column chunks, so reading all 18 columns of a row group at once peaks at ~1 GB, while
    one column at a time stays near the size of the cube (the machine has little free RAM). Only lts 33-62 are
    kept.
    """
    parks = sorted(weather_parks())
    park_pos = {p: i for i, p in enumerate(parks)}
    if split == "train":
        path = get_data_root() / MEPS_FORECAST
        f = pq.ParquetFile(path, read_dictionary=["sid"])
        if issues is None:
            issues = pd.concat([pc.unique(f.read_row_group(g, columns=["time_ref"]).column(0)).to_pandas()
                                for g in range(f.num_row_groups)]).unique()
        issues = _ns(pd.Series(issues).sort_values().unique())
        issue_ns = issues.tz_convert(None).asi8
        cube = np.full((len(issues), len(parks), len(CUBE_LTS), N_MEMBERS), np.nan, dtype=np.float32)
        for g in range(f.num_row_groups):
            keys = f.read_row_group(g, columns=["sid", "time_ref", "lt"])
            lt = keys.column("lt").to_numpy()
            ref = keys.column("time_ref").cast(pa.timestamp("ns")).to_numpy().astype("int64")
            ii = np.clip(np.searchsorted(issue_ns, ref), 0, len(issue_ns) - 1)
            keep = (lt >= CUBE_LTS[0]) & (lt <= CUBE_LTS[-1]) & (issue_ns[ii] == ref)
            if not keep.any():
                continue
            sid = keys.column("sid").combine_chunks()            # dictionary-encoded: codes + park names once
            lookup = np.array([park_pos[s] for s in sid.dictionary.to_pylist()])
            target = (ii[keep], lookup[sid.indices.to_numpy()[keep]], lt[keep] - CUBE_LTS[0])
            del keys, sid
            for m, col in enumerate(WS_COLS):
                values = f.read_row_group(g, columns=[col]).column(0).to_numpy()
                cube[target[0], target[1], target[2], m] = values[keep]
    else:
        df = load_met_forecast(variables="ws10m", split="eval")
        df = df.loc[(df["lt"] >= CUBE_LTS[0]) & (df["lt"] <= CUBE_LTS[-1])]
        all_issues = _ns(np.sort(df["time_ref"].unique()))
        issues = all_issues if issues is None else _ns(issues)
        df = df.loc[df["time_ref"].isin(issues)]
        cube = np.full((len(issues), len(parks), len(CUBE_LTS), N_MEMBERS), np.nan, dtype=np.float32)
        ii = issues.get_indexer(_ns(df["time_ref"]))
        pi = df["windpark"].map(park_pos).to_numpy()
        cube[ii, pi, df["lt"].to_numpy() - CUBE_LTS[0], :] = df[WS_COLS].to_numpy(dtype=np.float32)
    return {"ws": cube, "issues": issues, "parks": np.array(parks, dtype=object), "lts": CUBE_LTS}


def save_cube(cube, path):
    np.savez(path, ws=cube["ws"], issues=cube["issues"].tz_convert(None).asi8, parks=cube["parks"].astype(str),
             lts=cube["lts"])


def load_cube(path):
    with np.load(path) as z:
        return {"ws": z["ws"], "issues": _ns(pd.to_datetime(z["issues"])), "parks": z["parks"].astype(object),
                "lts": z["lts"]}


# ---------------------------------------------------------------- base table

# Issues whose forecasts cover D+2 +- 1 h for every park and member (D10 drops the truncated ones, C1)
def usable_issues(cube):
    keep = []
    for i, issue in enumerate(cube["issues"]):
        lts = target_lead_times(issue)
        need = np.arange(lts.min() - 1, lts.max() + 2) - CUBE_LTS[0]
        if need.min() >= 0 and need.max() < len(CUBE_LTS) and np.isfinite(cube["ws"][i][:, need]).all():
            keep.append(i)
    return np.array(keep, dtype=int)


def _last_power(series, limit):
    """Latest non-missing value with label <= limit, and its time (D6). (NaN, NaT) if there is none."""
    s = series.dropna()
    pos = s.index.searchsorted(limit, side="right") - 1
    return (s.iloc[pos], s.index[pos]) if pos >= 0 else (np.nan, pd.NaT)


def build_base_table(split="train", issues=None, cube=None, parks=None, power=None):
    cube = build_wind_cube(split, issues) if cube is None else cube
    parks = parks_table() if parks is None else parks
    power = load_power(split=split) if power is None else power
    keep = usable_issues(cube)
    issue_times = cube["issues"][keep]

    frames = []
    for issue in issue_times:
        hours = target_hours_utc(issue)
        lts = target_lead_times(issue)
        limit = last_power_limit(issue)
        d = issue_date_local(issue)
        for zone in VALID_ZONES:
            last, last_time = _last_power(power[zone], limit)
            frames.append(pd.DataFrame({
                "issue_time": issue, "issue_date": d, "issue_hour": issue.hour, "zone": zone,
                "target_time": hours, "target_hour_utc": hours.hour, "target_hour_local": hours.tz_convert(TZ).hour,
                "lt": lts, "last_power": last, "last_power_time": last_time,
                "power": power[zone].reindex(hours).to_numpy(),
            }))
    df = pd.concat(frames, ignore_index=True)
    df["issue_time"] = _ns(df["issue_time"])
    df["target_time"] = _ns(df["target_time"])
    df["last_power_time"] = pd.to_datetime(df["last_power_time"], utc=True).dt.as_unit("ns")
    df["fold"] = "windai" if split == "eval" else fold_of(df["issue_date"])

    # Capacity at issue time (D1): all parks for the target CF, covered parks for the ensemble (D5)
    df["capacity_mw"] = np.nan
    df["capacity_cov_mw"] = np.nan
    for zone in VALID_ZONES:
        rows = (df["zone"] == zone).to_numpy()
        times = df.loc[rows, "issue_time"]
        df.loc[rows, "capacity_mw"] = capacity_at(parks, zone, times)
        df.loc[rows, "capacity_cov_mw"] = capacity_at(parks, zone, times, covered_only=True)
    df["cf"] = df["power"] / df["capacity_mw"]
    df["last_cf"] = df["last_power"] / df["capacity_mw"]

    # Zone wind per member at t: capacity-weighted mean over covered parks in operation at issue time (D11)
    ws_cols = [f"ws_{m:02d}" for m in range(N_MEMBERS)]
    ws_out = np.full((len(df), N_MEMBERS), np.nan)
    cube_parks = list(cube["parks"])
    for zone in VALID_ZONES:
        covered = parks.loc[(parks["zone"] == zone) & parks["has_weather"] & parks["park"].isin(cube_parks)]
        pidx = [cube_parks.index(p) for p in covered["park"]]
        capw = capacity_matrix(covered, issue_times)
        zone_ws = zone_weighted(cube["ws"][np.ix_(keep, pidx)], capw)     # (issues, lts, members)
        rows = np.flatnonzero((df["zone"] == zone).to_numpy())
        ii = issue_times.get_indexer(df["issue_time"].iloc[rows])
        ws_out[rows] = zone_ws[ii, df["lt"].iloc[rows].to_numpy() - CUBE_LTS[0]]
    df[ws_cols] = ws_out

    df = df.sort_values(["issue_time", "zone", "target_time"], ignore_index=True)
    check_base_table(df)
    return df


def check_base_table(df):
    """Leakage, consistency and row-count asserts (02_pipeline P4, D2, D6, D10)."""
    keys = ["issue_time", "zone", "target_time"]
    assert not df.duplicated(keys).any(), "duplicate (issue_time, zone, target_time) rows"
    lead = ((df["target_time"] - df["issue_time"]) / pd.Timedelta(hours=1)).astype(int)
    assert (lead == df["lt"]).all(), "target_time - issue_time does not match lt"
    limits = df["issue_time"].map(last_power_limit)
    has_last = df["last_power_time"].notna()
    assert (df.loc[has_last, "last_power_time"] <= limits[has_last]).all(), "last_power from after the cutoff"
    per_issue = df.groupby("issue_time").size()
    expected = pd.Series({t: len(VALID_ZONES) * len(target_hours_utc(t)) for t in per_issue.index})
    assert (per_issue == expected).all(), "rows per issue != 4 zones x local hours on D+2"
    assert df["fold"].notna().all(), "issue outside every fold"
    ws_cols = [c for c in df.columns if c.startswith("ws_")]
    assert np.isfinite(df[ws_cols].to_numpy()).all(), "missing zone wind"


# ---------------------------------------------------------------- caches and the full dataset

def cube_path(split):
    return PROCESSED / f"wind_cube_{split}.npz"


def table_path(split):
    return PROCESSED / f"dataset_{split}.parquet"


def build_and_cache(split="train"):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cube = build_wind_cube(split)
    save_cube(cube, cube_path(split))
    table = build_base_table(split, cube=cube)
    table.to_parquet(table_path(split), index=False)
    return cube, table


def load_base(split="train"):
    df = pd.read_parquet(table_path(split))
    df["issue_date"] = pd.to_datetime(df["issue_date"]).dt.date
    return df


# The model table for one fit: the cached base table plus p_00..p_14 from a curve fitted on the training period
# of that fit ("val": train fold, "test": train + val, "windai": everything up to 31 Mar 2025; D2, D7).
def load_dataset(split="train", fit="val", curve=None):
    base, cube = load_base(split), load_cube(cube_path(split))
    curve = fit_power_curve(FIT_BOUNDARY[fit]) if curve is None else curve
    return add_power_ensemble(base, cube, curve), curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    args = parser.parse_args()
    cube, table = build_and_cache(args.split)
    print(f"cube: {cube['ws'].shape}, {cube['ws'].nbytes / 1e6:.0f} MB -> {cube_path(args.split)}")
    print(f"issues: {len(cube['issues'])}, usable: {table['issue_time'].nunique()}")
    print(f"table: {len(table)} rows -> {table_path(args.split)}")
    print(table.groupby("fold")["issue_time"].nunique().rename("issues per fold"))
