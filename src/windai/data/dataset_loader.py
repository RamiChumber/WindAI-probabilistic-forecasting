import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from windai.paths import get_data_root, list_files

def load_power(zone=None, time_ref=None):
    flist = list_files()
    power_dataset = [x.absolute() for x in flist if x.name == "wind_power_per_bidzone.parquet"]
    pq_file = pq.ParquetFile(power_dataset[0])
    pq_table = pq_file.read()
    return pq_table.to_pandas()

def load_meps_forecast(sids=None, columns=None, time_ref_range=None):
    return

def load_meps_nowcast(sids=None, columns=None, time_ref=None):
    return

def load_parks():
    return