import numpy as np
import pandas as pd
from scipy.optimize import minimize
from windai.models.base import Model
from windai.models import censored_normal as cn
from windai.evaluation.forecast import Forecast, QUANTILE_LEVELS

ALL_MEMBERS = [f"p_{m:02d}" for m in range(15)]
PERTURBED = ALL_MEMBERS[1:]

# EMOS (Gneiting et al., 2005): a parametric predictive distribution whose location and scale depend linearly on
# ensemble statistics, fitted on training data.
# Here, for capacity factor, per zone and UTC target hour:
#   Y* ~ N(mu, sigma^2),  mu = a + b * mean(p_00..p_14),  sigma^2 = c^2 + d^2 * var(p_01..p_14)
#   Y  = min(max(Y*, 0), 1)                         (censored normal: power is bounded by 0 and capacity)
# The location uses all 15 members. The scale uses only members 01-14: they are exchangeable, while member 00,
# the unperturbed control run, sits in the middle of the ensemble and would understate the spread. The squares
# keep sigma positive without constraints. c^2 is the variance left when the members agree; d^2 scales the
# ensemble variance.
# Fitted by maximum likelihood (default) or by minimising the mean CRPS. The forecast is converted to MW with
# the row's capacity at issue time; the exact CRPS and PIT of the censored normal are attached.


def ensemble_stats(df):
    """Mean of all 15 members and sample variance of members 01-14 (capacity factor)."""
    return df[ALL_MEMBERS].to_numpy().mean(axis=1), df[PERTURBED].to_numpy().var(axis=1, ddof=1)


def location_scale(theta, m, s2):
    a, b, c, d = theta
    return a + b * m, np.sqrt(c * c + d * d * s2)


def fit_censored(y, m, s2, objective="ml", x0=None):
    """Fits (a, b, c, d) of one censored-normal EMOS model. Without x0, starts from least squares of y on the
    ensemble mean, with the residual SD as c and d = 1. Returns (theta, converged, loss)."""
    if x0 is None:
        X = np.column_stack([np.ones_like(m), m])
        (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
        x0 = np.array([a, b, np.std(y - X @ np.array([a, b])), 1.0])

    if objective == "ml":
        # Mean negative log-likelihood with its analytic gradient (chain rule through mu = a + b m and
        # sigma = sqrt(c^2 + d^2 s2)), which saves the optimiser most of its function evaluations
        def loss(theta):
            a, b, c, d = theta
            mu, sigma = location_scale(theta, m, s2)
            d_mu, d_sigma = cn.log_likelihood_grad(y, mu, sigma)
            grad = -np.array([d_mu.mean(), (d_mu * m).mean(),
                              (d_sigma * c / sigma).mean(), (d_sigma * d * s2 / sigma).mean()])
            return -cn.log_likelihood(y, mu, sigma).mean(), grad
        res = minimize(loss, x0, method="L-BFGS-B", jac=True)
    else:
        def loss(theta):
            mu, sigma = location_scale(theta, m, s2)
            return cn.crps(y, mu, sigma).mean()
        res = minimize(loss, x0, method="L-BFGS-B")
    theta = res.x.copy()
    theta[2:] = np.abs(theta[2:])               # c and d enter squared; report them as non-negative
    return theta, res.success, res.fun


def _check_objective(objective):
    if objective not in ("ml", "crps"):
        raise ValueError("objective must be 'ml' or 'crps'")
    return objective


def censored_forecast(df, theta):
    """Forecast in MW from per-row parameters theta (n, 4), with the exact CRPS and PIT attached."""
    m, s2 = ensemble_stats(df)
    mu, sigma = location_scale(np.asarray(theta, dtype=float).T, m, s2)
    cap = df["capacity_mw"].to_numpy()
    quantiles = cn.quantile(QUANTILE_LEVELS[None, :], mu[:, None], sigma[:, None]) * cap[:, None]
    point = cn.mean(mu, sigma) * cap
    # The CRPS scales with the unit: CRPS in MW = capacity x CRPS of the CF forecast at y / capacity
    crps_fn = lambda y: cap * cn.crps(np.asarray(y, dtype=float) / cap, mu, sigma)
    pit_fn = lambda y, rng=None: cn.pit(np.asarray(y, dtype=float) / cap, mu, sigma, rng)
    return Forecast(keys=df, point=point, quantiles=quantiles, crps_fn=crps_fn, pit_fn=pit_fn)


# Static EMOS: one fit per zone and UTC hour on all training rows.
class EMOS(Model):
    name = "emos"

    def __init__(self, objective="ml"):
        self.objective = _check_objective(objective)

    def fit(self, train_df):
        d = train_df.dropna(subset=["cf"])
        m, s2 = ensemble_stats(d)
        y = np.clip(d["cf"].to_numpy(), 0.0, 1.0)
        rows = []
        for (zone, hour), idx in d.groupby(["zone", "target_hour_utc"]).indices.items():
            theta, ok, loss = fit_censored(y[idx], m[idx], s2[idx], self.objective)
            rows.append({"zone": zone, "target_hour_utc": hour, "a": theta[0], "b": theta[1], "c": theta[2],
                         "d": theta[3], "converged": ok, "loss": loss, "n": len(idx)})
        self.params = pd.DataFrame(rows).set_index(["zone", "target_hour_utc"]).sort_index()
        return self

    def summary(self):
        return self.params.copy()

    def predict(self, df):
        keys = pd.MultiIndex.from_arrays([df["zone"], df["target_hour_utc"]])
        p = self.params.reindex(keys)[["a", "b", "c", "d"]]
        if p.isna().any().any():
            raise ValueError("no fitted EMOS parameters for some (zone, hour) in df")
        return censored_forecast(df, p.to_numpy())


# Rolling EMOS: the same model, refitted for every issue D on the observations known at D's issue time.
# The window ends at issue D-3, the latest whose target day (D-1) is fully observed by 09Z on D, and spans
# `window_days` calendar days (None: everything up to D-3, an expanding window). Short windows pool hours
# h-1, h, h+1 to have enough values per fit. Each refit starts from the previous issue's parameters.
# fit() takes the history (windai.data.splits.history_rows), which deliberately includes observations from the
# evaluated period; the D-3 rule is enforced here.
class RollingEMOS(Model):
    sequential = True
    LAG_DAYS = 3

    def __init__(self, window_days=90, pool_hours=None, objective="ml", min_values=30):
        self.window_days = window_days
        self.pool_hours = (window_days is not None and window_days <= 90) if pool_hours is None else pool_hours
        self.objective = _check_objective(objective)
        self.min_values = min_values
        self.name = f"emos_rolling_{window_days}" if window_days else "emos_expanding"

    def fit(self, history):
        d = history.dropna(subset=["cf"])
        m, s2 = ensemble_stats(d)
        self._hist = {}
        for zone, idx in d.groupby("zone").indices.items():
            g = d.iloc[idx]
            order = np.argsort(pd.to_datetime(g["issue_date"]).to_numpy(), kind="stable")
            self._hist[zone] = {
                "date": pd.to_datetime(g["issue_date"]).to_numpy().astype("datetime64[D]")[order],
                "hour": g["target_hour_utc"].to_numpy()[order],
                "y": np.clip(g["cf"].to_numpy(), 0.0, 1.0)[order],
                "m": m[idx][order], "s2": s2[idx][order],
            }
        return self

    def _hours(self, h):
        return [(h - 1) % 24, h, (h + 1) % 24] if self.pool_hours else [h]

    def predict(self, df):
        issue_dates = pd.to_datetime(df["issue_date"]).to_numpy().astype("datetime64[D]")
        records, prev = [], {}
        for day in np.unique(issue_dates):
            end = day - np.timedelta64(self.LAG_DAYS, "D")
            start = None if self.window_days is None else end - np.timedelta64(self.window_days - 1, "D")
            for zone in np.unique(df["zone"]):
                h = self._hist[zone]
                lo = 0 if start is None else np.searchsorted(h["date"], start, side="left")
                hi = np.searchsorted(h["date"], end, side="right")
                for hour in range(24):
                    sel = lo + np.flatnonzero(np.isin(h["hour"][lo:hi], self._hours(hour)))
                    key = (zone, hour)
                    if len(sel) >= self.min_values:
                        theta, ok, _ = fit_censored(h["y"][sel], h["m"][sel], h["s2"][sel], self.objective,
                                                    x0=prev.get(key))
                        fallback = False
                    elif key in prev:                       # too few values (data gap): keep the last fit
                        theta, ok, fallback = prev[key], True, True
                    else:                                   # nothing yet: use everything up to D-3
                        sel = np.flatnonzero(np.isin(h["hour"][:hi], self._hours(hour)))
                        theta, ok, _ = fit_censored(h["y"][sel], h["m"][sel], h["s2"][sel], self.objective)
                        fallback = True
                    prev[key] = theta
                    records.append({"issue_date": day, "zone": zone, "target_hour_utc": hour, "a": theta[0],
                                    "b": theta[1], "c": theta[2], "d": theta[3], "n": len(sel), "converged": ok,
                                    "fallback": fallback})
        self.params_by_issue = pd.DataFrame(records)
        keys = pd.MultiIndex.from_arrays([issue_dates, df["zone"], df["target_hour_utc"]])
        p = self.params_by_issue.set_index(["issue_date", "zone", "target_hour_utc"]).reindex(keys)
        return censored_forecast(df, p[["a", "b", "c", "d"]].to_numpy())
