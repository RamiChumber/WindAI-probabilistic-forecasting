import datetime
import pandas as pd

# Selecting training and evaluation rows for a fit.
# A "fit" names what is evaluated: "val" (fit on train), "test" (refit on train + val), "windai" (refit on
# everything up to 31 Mar 2025, scored on the competition period).

FIT_BOUNDARY = {
    "val": datetime.date(2023, 4, 1),
    "test": datetime.date(2024, 4, 1),
    "windai": datetime.date(2025, 4, 1),
}


# Training rows for a fit: issues before the boundary whose target day D+2 also lies before it, so no training
# row uses power observed after the evaluated fold has started (this purges 2 issues per boundary). Rows without
# observed power are dropped. The WindAI period is never training data.
def training_rows(df, fit):
    boundary = pd.Timestamp(FIT_BOUNDARY[fit])
    target_day = pd.to_datetime(df["issue_date"]) + pd.Timedelta(days=2)
    keep = (target_day < boundary) & df["power"].notna() & (df["fold"] != "windai")
    return df.loc[keep.to_numpy()].reset_index(drop=True)


# End (exclusive, local issue date) of the fold each fit evaluates
FOLD_END = {
    "val": datetime.date(2024, 4, 1),
    "test": datetime.date(2025, 4, 1),
    "windai": datetime.date(2100, 1, 1),
}


# History for sequential models, which refit for every issue on the observations known at that issue's time.
# Unlike training_rows, this deliberately includes observations from inside the evaluated fold: a model
# forecasting issue D may use issues up to D-3, whose targets are fully observed by then, and it must enforce
# that itself. Rows after the evaluated fold and rows without observed power are excluded.
def history_rows(df, fit):
    end = pd.Timestamp(FOLD_END[fit])
    keep = (pd.to_datetime(df["issue_date"]) < end) & df["power"].notna()
    if fit != "windai":
        keep &= df["fold"] != "windai"
    return df.loc[keep.to_numpy()].reset_index(drop=True)


# Rows to forecast and score for a fit; rows with missing power stay (scoring drops them)
def evaluation_rows(df, fit):
    return df.loc[(df["fold"] == fit).to_numpy()].reset_index(drop=True)
