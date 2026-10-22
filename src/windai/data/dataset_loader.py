import pyarrow.parquet as pq
import pandas as pd
from windai.paths import get_data_root

valid_zones = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
target = "wind_power_per_bidzone.parquet"
parks = "windparks_bidzone.csv"

# Loads the dataset with the target variable. Time filtering gives rows for [start, end). Takes naive timestamps
def load_power(zone=None, start=None, end=None):
    pq_file = pq.ParquetFile(get_data_root() / target)
    if zone:
        if zone not in valid_zones:
            raise ValueError(f"Please input a valid bidding zone ({valid_zones}), or leave 'zone' blank")
        pq_table = pq_file.read(columns=[zone], use_pandas_metadata=True)
    else:
        pq_table = pq_file.read(use_pandas_metadata=True)
    
    df = pq_table.to_pandas()
    df.index.name = "time"
    df.index = df.index.tz_localize("UTC")
    
    if start:
        start = pd.Timestamp(start)
        start = start.tz_localize("UTC")
    if end:
        end = pd.Timestamp(end)
        end = end.tz_localize("UTC")
    if start and end:
        return df.loc[(df.index >= start) & (df.index < end)]
    elif start:
        return df.loc[df.index >= start]
    elif end:
        return df.loc[df.index < end]
    else:
        return df


# Loads the MEPS forecast dataset
def load_meps_forecast(sids=None, columns=None, time_ref_range=None):
    return

# Loads the MEPS nowcast dataset
def load_meps_nowcast(sids=None, columns=None, time_ref=None):
    return

# Loads the park-to-zone mapping
def load_parks(zone=None):
    if zone:
        if zone not in valid_zones:
            raise ValueError(f"Please input a valid bidding zone ({valid_zones}), or leave 'zone' blank")
    df = pd.read_csv(get_data_root() / parks, encoding="utf-8")
    df["prod_start_new"] = pd.to_datetime(df["prod_start_new"], utc=True)
    if zone:
        return df.loc[df["bidding_area"] == zone]
    else:
        return df