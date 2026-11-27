import datetime
import numpy as np
import pandas as pd
from windai.data.splits import training_rows, evaluation_rows
from windai.utils.synthetic import make_synthetic_dataset


def test_purge_at_val_boundary(synthetic_df):
    train = training_rows(synthetic_df, "val")
    # Issues on 30 and 31 Mar have targets in the val fold: purged (D2)
    assert train["issue_date"].max() == datetime.date(2023, 3, 29)
    assert (train["target_time"] < pd.Timestamp("2023-03-31 22:00", tz="UTC")).all()   # 1 Apr 00:00 local
    assert set(train["fold"]) == {"train"}


def test_missing_power_not_trained_on(synthetic_df):
    df = synthetic_df.copy()
    df.loc[:9, "power"] = np.nan
    assert len(training_rows(df, "val")) == len(training_rows(synthetic_df, "val")) - 10


def test_evaluation_rows_keep_missing_power(synthetic_df):
    df = synthetic_df.copy()
    rows = evaluation_rows(df, "val")
    df.loc[df["fold"] == "val", "power"] = np.nan
    assert len(evaluation_rows(df, "val")) == len(rows)


# Refit for test: train + val, purged at 1 Apr 2024; the WindAI period is never training data
def test_test_fit_uses_train_and_val():
    df = make_synthetic_dataset(n_issues=500, start="2023-01-01")
    df.loc[df["issue_date"] >= datetime.date(2024, 4, 15), "fold"] = "windai"   # stand-in rows
    train = training_rows(df, "test")
    assert set(train["fold"]) == {"train", "val"}
    assert train["issue_date"].max() == datetime.date(2024, 3, 29)
