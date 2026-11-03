import pyarrow.parquet as pq
import pandas as pd
import datetime
import re
from windai.paths import get_data_root, list_files

VALID_ZONES = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
TARGET = "wind_power_per_bidzone.parquet"
PARKS = "windparks_bidzone.csv"
MEPS_NOWCAST = "met_nowcast.parquet"
MEPS_FORECAST = "met_forecast.parquet"
EVAL_TARGET = "evaluation/wind_power"
EVAL_NOWCAST = "evaluation/met_nowcast_daily"
EVAL_FORECAST = "evaluation/met_forecast_daily"
SPLITS = ["train", "eval"]
MEPS_VARIABLES = ["ws10m", "wd10m", "t2m", "rh2m", "mslp", "g10m"]
N_MEMBERS = 15
FORECAST_KEYS = ["windpark", "time_ref", "time", "lt"]
STORED_TO_CANONICAL_NAMES = { "Tonst": "Tonst. Vindpark", "Sørfj": "Sørfj. Vindkraft"}
CANONICAL_TO_STORED_NAMES = {v:k for (k,v) in STORED_TO_CANONICAL_NAMES.items()}

# Shared helper functions

# Converts a timestamp (naive is taken as UTC) to an UTC-aware timestamp
def _to_utc(ts):
    ts = pd.Timestamp(ts)
    if ts.tz:
        if ts.tz != datetime.timezone.utc:
            ts = ts.tz_convert("UTC")
    else:
        ts = ts.tz_localize("UTC")
    return ts

# Time filter giving rows for [start, end)
def _filter_time(df, start, end):
    if start:
        start = _to_utc(start)
    if end:
        end = _to_utc(end)
    if start and end:
        return df.loc[(df.index >= start) & (df.index < end)]
    elif start:
        return df.loc[df.index >= start]
    elif end:
        return df.loc[df.index < end]
    else:
        return df

# Function for checking whether a bidding zone is valid
def _check_zone(zone):
    if zone not in VALID_ZONES:
        raise ValueError(f"Please input a valid bidding zone ({VALID_ZONES}), or leave 'zone' blank")

def _check_split(split):
    if split not in SPLITS:
        raise ValueError(f"Please input a valid split ({SPLITS})")

# Raises if any of the requested names are not among the valid ones
def _check_names(requested, valid, kind):
    invalid = [x for x in requested if x not in valid]
    if invalid:
        raise ValueError(f"Invalid {kind}: {invalid}. Valid {kind}: {sorted(valid)}")

# Lets the user pass a single name or several
def _as_list(x):
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)

# Concatenates the daily parquet files in an evaluation folder. Some files overlap,
# so rows that are identical (including the index) are only kept once
def _read_daily(subdir):
    df = pd.concat([pd.read_parquet(f) for f in list_files(subdir)])
    return df[~df.reset_index().duplicated().to_numpy()]

# Dataset loaders

# Loads the dataset with the target variable
def load_power(zone=None, start=None, end=None, split="train"):
    _check_split(split)
    if zone:
        _check_zone(zone)
    if split == "train":
        pq_file = pq.ParquetFile(get_data_root() / TARGET)
        if zone:
            pq_table = pq_file.read(columns=[zone], use_pandas_metadata=True)
        else:
            pq_table = pq_file.read(use_pandas_metadata=True)
        df = pq_table.to_pandas()
    else:
        df = _read_daily(EVAL_TARGET)
        if zone:
            df = df[[zone]]

    df.index.name = "time"
    df.index = df.index.tz_localize("UTC")

    return _filter_time(df, start, end)


# Loads the MEPS forecast dataset in wide format, one column per ensemble member.
# start/end filter on time_ref (issue time). Units: K, Pa and fraction
def load_met_forecast(windparks=None, variables=None, start=None, end=None, split="train"):
    _check_split(split)
    windparks = _as_list(windparks)
    variables = _as_list(variables)
    if variables is not None:
        _check_names(variables, MEPS_VARIABLES, "variables")
    else:
        variables = MEPS_VARIABLES
    member_cols = [f"{v}_{i:02d}" for v in variables for i in range(N_MEMBERS)]

    if split == "train":
        forecast_path = get_data_root() / MEPS_FORECAST
        filters = []
        if windparks is not None:
            valid = pq.read_table(forecast_path, columns=["sid"]).column("sid").unique().to_pylist()
            _check_names(windparks, valid, "windparks")
            filters.append(("sid", "in", windparks))
        if start:
            filters.append(("time_ref", ">=", _to_utc(start).tz_localize(None)))
        if end:
            filters.append(("time_ref", "<", _to_utc(end).tz_localize(None)))
        df = pd.read_parquet(forecast_path, columns=["sid", "time_ref", "time", "lt"] + member_cols,
                             filters=filters or None)
        df = df.rename(columns={"sid": "windpark"})
        df["time_ref"] = df["time_ref"].dt.tz_localize("UTC")
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        # File names are meps_windai_YYYYMMDDTHHZ.csv, with the time_ref of the forecast
        files = [(f, pd.Timestamp(f.name[12:23]).tz_localize("UTC")) for f in list_files(EVAL_FORECAST)]
        if start:
            files = [(f, t) for f, t in files if t >= _to_utc(start)]
        if end:
            files = [(f, t) for f, t in files if t < _to_utc(end)]
        stored_cols = [f"{v}_{i}" for v in variables for i in range(N_MEMBERS)]
        dfs = [pd.read_csv(f, encoding="utf-8", usecols=["sid", "time_ref", "time", "lt"] + stored_cols) for f, _ in files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={"sid": "windpark"})
        df = df.rename(columns=lambda c: re.sub(r"_(\d+)$", lambda m: f"_{int(m.group(1)):02d}", c))
        df["time_ref"] = pd.to_datetime(df["time_ref"], utc=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        if windparks is not None:
            _check_names(windparks, df["windpark"].unique(), "windparks")
            df = df.loc[df["windpark"].isin(windparks)].reset_index(drop=True)

    return df[FORECAST_KEYS + member_cols]

# Loads the MEPS nowcast dataset. Units: K, Pa and fraction
def load_met_nowcast(windparks=None, start=None, end=None, split="train"):
    _check_split(split)
    windparks = _as_list(windparks)
    if split == "train":
        nowcast_path = get_data_root() / MEPS_NOWCAST
        if windparks is not None:
            stored_parks = pd.read_parquet(nowcast_path, columns=["windpark"])["windpark"].unique()
            valid = [STORED_TO_CANONICAL_NAMES.get(p, p) for p in stored_parks]
            _check_names(windparks, valid, "windparks")
            requested = [CANONICAL_TO_STORED_NAMES.get(p, p) for p in windparks]
            df = pd.read_parquet(nowcast_path, filters=[("windpark", "in", requested)])
        else:
            df = pd.read_parquet(nowcast_path)
    else:
        df = _read_daily(EVAL_NOWCAST)
        if windparks is not None:
            valid = [STORED_TO_CANONICAL_NAMES.get(p, p) for p in df["windpark"].unique()]
            _check_names(windparks, valid, "windparks")

    df["windpark"] = df["windpark"].replace(STORED_TO_CANONICAL_NAMES)
    if split == "eval" and windparks is not None:
        df = df.loc[df["windpark"].isin(windparks)]
    df.index.name = "time"
    df.index = df.index.tz_localize("UTC")

    return _filter_time(df, start, end)

# Loads the park-to-zone mapping, with option to filter by bidding zone/area
def load_parks(zone=None):
    if zone:
        _check_zone(zone)
    df = pd.read_csv(get_data_root() / PARKS, encoding="utf-8")
    df["prod_start_new"] = pd.to_datetime(df["prod_start_new"], utc=True)
    if zone:
        return df.loc[df["bidding_area"] == zone]
    else:
        return df
