import datetime
import pandas as pd

# Selecting training and evaluation rows for a fit (D2, D12).
# A "fit" names what is evaluated: "val" (fit on train), "test" (refit on train + val), "windai" (refit on
# everything up to 31 Mar 2025, scored on the competition period).

FIT_BOUNDARY = {
    "val": datetime.date(2023, 4, 1),
    "test": datetime.date(2024, 4, 1),
    "windai": datetime.date(2025, 4, 1),
}


# Training rows for a fit: issues before the boundary whose target day D+2 also lies before it, so no training
# row uses power observed after the evaluated fold has started (the purge drops 2 issues per boundary, C2).
# Rows without observed power are dropped (D10). The WindAI period is never training data.
def training_rows(df, fit):
    boundary = pd.Timestamp(FIT_BOUNDARY[fit])
    target_day = pd.to_datetime(df["issue_date"]) + pd.Timedelta(days=2)
    keep = (target_day < boundary) & df["power"].notna() & (df["fold"] != "windai")
    return df.loc[keep.to_numpy()].reset_index(drop=True)


# Rows to forecast and score for a fit; rows with missing power stay (scoring drops them, D10)
def evaluation_rows(df, fit):
    return df.loc[(df["fold"] == fit).to_numpy()].reset_index(drop=True)
