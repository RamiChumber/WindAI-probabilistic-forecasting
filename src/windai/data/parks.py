from functools import lru_cache
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from windai.paths import get_data_root
from windai.data.dataset_loader import load_parks, MEPS_NOWCAST, STORED_TO_CANONICAL_NAMES

# Parks and installed capacity (D1, D5).


# Names of the parks with weather data. The forecast and the nowcast have exactly the same 61 parks (C3), and the
# nowcast is much smaller to read; its park column is read dictionary-encoded (one code per row, names once).
@lru_cache(maxsize=1)
def weather_parks():
    table = pq.read_table(get_data_root() / MEPS_NOWCAST, columns=["windpark"], read_dictionary=["windpark"])
    stored = table.column("windpark").combine_chunks().dictionary.to_pylist()
    return frozenset(STORED_TO_CANONICAL_NAMES.get(n, n) for n in stored)


# One row per park. The metadata lists some parks under several EIC codes, each row repeating the park's full
# capacity, so it is deduplicated by name, not by EIC code (Øyfjell1/2 share a code but are two phases) (D1, C3).
def parks_table():
    df = load_parks().drop_duplicates("substation_name")
    out = pd.DataFrame({
        "park": df["substation_name"].to_numpy(),
        "zone": df["bidding_area"].to_numpy(),
        "capacity_mw": df["operating_power_max"].to_numpy(dtype=float),
        "prod_start": df["prod_start_new"].to_numpy(),
    })
    out["prod_start"] = pd.to_datetime(out["prod_start"], utc=True)
    out["has_weather"] = out["park"].isin(weather_parks())
    return out.sort_values(["zone", "park"]).reset_index(drop=True)


# Matrix of each park's capacity in operation at each time, shape (len(times), len(parks)): capacity_mw where
# prod_start <= t, else 0. Capacity is constant from start-up (no decommissioning or derating, D1).
def capacity_matrix(parks, times):
    times = pd.DatetimeIndex(times)
    times = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    # Compare as naive UTC datetime64, which is vectorised (tz-aware arrays become slow object arrays)
    starts = parks["prod_start"].dt.tz_convert(None).to_numpy()
    started = starts[None, :] <= times.tz_convert(None).to_numpy()[:, None]
    return np.where(started, parks["capacity_mw"].to_numpy()[None, :], 0.0)


# Installed capacity of a zone at each time. covered_only=True counts only parks with weather data, the
# denominator of the zone ensemble (D5); the default counts all parks, the denominator of the target CF (D1).
def capacity_at(parks, zone, times, covered_only=False):
    sel = parks["zone"] == zone
    if covered_only:
        sel &= parks["has_weather"]
    return capacity_matrix(parks.loc[sel], times).sum(axis=1)
