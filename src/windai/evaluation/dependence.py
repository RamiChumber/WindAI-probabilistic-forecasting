import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

# Dependence in forecast errors. A calibrated forecast gives PIT values u that are uniform; their normal scores
# z = Phi^-1(u) are then standard normal, so correlations of z across hours, days and zones measure dependence
# that the marginal forecasts leave out. Daily series are handled on a calendar-day axis: a missing issue day
# is a gap, never a shift of the lags.

MAX_LAG = 14


def normal_scores(pit):
    """z = Phi^-1(u), with u kept away from 0 and 1 so the scores stay finite."""
    return ndtri(np.clip(np.asarray(pit, dtype=float), 1e-6, 1 - 1e-6))


def z_matrix(df, z):
    """Per zone, a DataFrame of issue day x local hour of D+2 (columns 0-23). On the 23-hour DST day the
    missing hour is NaN; on the 25-hour day the repeated hour keeps its first occurrence. df holds the rows
    the scores z belong to (columns zone, issue_date, target_hour_local), in the same order."""
    d = pd.DataFrame({"zone": df["zone"].to_numpy(), "issue_date": pd.to_datetime(df["issue_date"]).to_numpy(),
                      "hour": df["target_hour_local"].to_numpy(), "z": np.asarray(z, dtype=float)})
    d = d.drop_duplicates(subset=["zone", "issue_date", "hour"], keep="first")
    out = {}
    for zone, g in d.groupby("zone"):
        out[zone] = g.pivot(index="issue_date", columns="hour", values="z").reindex(columns=range(24))
    return out


def daily(series):
    """A date-indexed series on a complete calendar-day axis (missing days become NaN)."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))


def acf(series, max_lag=MAX_LAG):
    """Autocorrelation at lags 1..max_lag days of a date-indexed series, from the pairs where both days are
    observed, standardised by the overall mean and variance."""
    x = daily(series).to_numpy(dtype=float)
    dev = x - np.nanmean(x)
    var = np.nanmean(dev * dev)
    rho = np.full(max_lag, np.nan)
    for k in range(1, max_lag + 1):
        prod = dev[k:] * dev[:-k]
        if np.isfinite(prod).any():
            rho[k - 1] = np.nanmean(prod) / var
    return rho


def n_eff(rho, n):
    """Effective sample size n / (1 + 2 sum rho_k), summing the lags up to the first non-positive rho_k; the
    cut keeps the noisy far lags from dominating the sum."""
    total = 0.0
    for r in rho:
        if not np.isfinite(r) or r <= 0:
            break
        total += r
    return n / (1 + 2 * total)


def long_run_cov(S, lag, dates=None):
    """Newey-West (Bartlett kernel) estimate of the long-run covariance of a series of vectors S (n, p), after
    centring: Gamma_0 + sum_{j=1..lag} (1 - j / (lag + 1)) (Gamma_j + Gamma_j^T). lag = 0 is the plain
    covariance, the estimate that assumes independence. With dates (one per row), the series is put on a
    calendar-day axis: rows of the same day are summed into one block (e.g. the hour that occurs twice on the
    25-hour DST day), and missing days are filled with zeros, which contribute nothing. The sums are divided
    by the number of rows."""
    S = np.asarray(S, dtype=float)
    S = S - S.mean(axis=0)
    n = len(S)
    if dates is not None:
        days = pd.to_datetime(pd.Series(dates)).dt.normalize()
        pos = ((days - days.min()) / pd.Timedelta(days=1)).astype(int).to_numpy()
        full = np.zeros((pos.max() + 1, S.shape[1]))
        np.add.at(full, pos, S)
        S = full
    B = S.T @ S / n
    for j in range(1, lag + 1):
        G = S[j:].T @ S[:-j] / n
        B += (1 - j / (lag + 1)) * (G + G.T)
    return B


def numerical_hessian(grad, theta, eps=1e-5):
    """Hessian from central differences of an analytic gradient function, symmetrised."""
    theta = np.asarray(theta, dtype=float)
    p = len(theta)
    H = np.empty((p, p))
    for i in range(p):
        e = np.zeros(p)
        e[i] = eps
        H[:, i] = (grad(theta + e) - grad(theta - e)) / (2 * eps)
    return 0.5 * (H + H.T)


def sandwich_ratios(A, B):
    """How much wider the sandwich variance A^-1 B A^-1 is than the inverse Fisher information A^-1 (the
    variance when errors are independent and the model is correct): the per-parameter ratio of the diagonals,
    and the largest eigenvalue of A^-1 B."""
    A_inv = np.linalg.inv(A)
    sandwich = A_inv @ B @ A_inv
    ratio = np.diag(sandwich) / np.diag(A_inv)
    return ratio, float(np.max(np.real(np.linalg.eigvals(A_inv @ B))))


def diebold_mariano(loss_a, loss_b, dates, lag=7):
    """Diebold-Mariano test of equal expected loss for two forecasts of the same rows. The loss differences
    a - b are averaged per day first (the hours of a day are strongly dependent), and the variance of their
    mean is the Newey-West long-run variance over calendar days, which allows for dependence across days.
    Returns the mean difference, the same relative to forecast b, the t statistic and the two-sided p-value
    (normal approximation), and the number of days. A negative difference favours a."""
    diff = pd.Series(np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float),
                     index=pd.to_datetime(pd.Series(dates)).dt.normalize())
    day = diff.groupby(level=0).mean()
    n = len(day)
    var = long_run_cov(day.to_numpy()[:, None], lag, dates=day.index)[0, 0]
    stat = day.mean() / np.sqrt(var / n)
    mean_b = np.mean(loss_b)
    return {"mean_diff": day.mean(), "rel_diff": day.mean() / mean_b if mean_b != 0 else np.nan, "stat": stat,
            "p": 2 * ndtr(-abs(stat)), "n_days": n}


def benjamini_hochberg(p):
    """Benjamini-Hochberg adjusted p-values: rejecting where the adjusted value is below alpha keeps the
    expected share of false discoveries among the rejections at most alpha (for independent or positively
    dependent tests). NaNs stay NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]       # monotone from the largest p downwards
    vals = np.empty(m)
    vals[order] = np.minimum(adjusted, 1.0)
    out[ok] = vals
    return out
