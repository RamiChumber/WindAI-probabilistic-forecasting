import numpy as np
import pandas as pd
from windai.evaluation import scoring

PINBALL_LEVELS = [0.1, 0.5, 0.9]
COVERAGE_WIDTHS = [0.5, 0.8, 0.9, 0.95]     # 95 % bounds are interpolated (scoring.quantile_at)


# Per-row scores of a forecast against observations y (same row order as forecast.keys).
# Rows where y is missing are dropped. Optional columns only appear if the forecast provides them.
def score_rows(forecast, y):
    y = np.asarray(y, dtype=float)
    if y.shape != (len(forecast),):
        raise ValueError(f"y has shape {y.shape}, expected ({len(forecast)},)")
    rows = forecast.keys.copy()
    rows["issue_hour"] = rows["issue_time"].dt.hour       # regime (12Z / 09Z), for grouping (D3)
    rows["y"] = y
    rows["sq_error"] = (forecast.point - y) ** 2
    rows["abs_error"] = np.abs(forecast.point - y)
    rows["crps_q"] = scoring.crps_quantiles(forecast.quantiles, forecast.levels, y)
    for tau in PINBALL_LEVELS:
        i = scoring._level_index(forecast.levels, tau)
        rows[f"pinball_{tau}"] = scoring.pinball(forecast.quantiles[:, i], y, tau)
    for width in COVERAGE_WIDTHS:
        rows[f"cov_{int(width * 100)}"] = scoring.coverage(forecast.quantiles, forecast.levels, y, width)
    if forecast.samples is not None:
        rows["crps_sample"] = scoring.crps_sample(forecast.samples, y)
    if forecast.crps_fn is not None:
        rows["crps_exact"] = forecast.crps_fn(y)
    return rows.loc[~np.isnan(y)].reset_index(drop=True)


# Summary table: one row per group in `by` plus an "all" row. RMSE is the square root of the mean squared
# error within each group (as in the competition), the other scores are plain means.
def score(forecast, y, by=("zone",)):
    rows = score_rows(forecast, y)
    by = list(by)
    metrics = [c for c in rows.columns if c not in forecast.keys.columns and c not in ("y", "issue_hour")]

    def summarise(df):
        out = df[metrics].mean()
        out["rmse"] = np.sqrt(out.pop("sq_error"))
        out = out.rename({"abs_error": "mae"})
        out["n"] = len(df)
        return out

    table = rows.groupby(by).apply(summarise, include_groups=False)
    overall = summarise(rows).to_frame().T
    overall.index = pd.MultiIndex.from_tuples([("all",) * len(by)], names=by) if len(by) > 1 \
        else pd.Index(["all"], name=by[0])
    table = pd.concat([table, overall])
    first = ["n", "rmse", "mae", "crps_q"]
    return table[first + [c for c in table.columns if c not in first]]
