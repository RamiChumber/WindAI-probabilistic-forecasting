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


class EMOS(Model):
    name = "emos"

    def __init__(self, objective="ml"):
        if objective not in ("ml", "crps"):
            raise ValueError("objective must be 'ml' or 'crps'")
        self.objective = objective

    def _fit_one(self, y, m, s2):
        # Start from least squares of y on the ensemble mean, with the residual SD as c and d = 1
        X = np.column_stack([np.ones_like(m), m])
        (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
        c = np.std(y - X @ np.array([a, b]))
        x0 = np.array([a, b, c, 1.0])

        if self.objective == "ml":
            def loss(theta):
                mu, sigma = location_scale(theta, m, s2)
                return -cn.log_likelihood(y, mu, sigma).mean()
        else:
            def loss(theta):
                mu, sigma = location_scale(theta, m, s2)
                return cn.crps(y, mu, sigma).mean()

        res = minimize(loss, x0, method="L-BFGS-B")
        theta = res.x.copy()
        theta[2:] = np.abs(theta[2:])           # c and d enter squared; report them as non-negative
        return theta, res.success, res.fun

    def fit(self, train_df):
        d = train_df.dropna(subset=["cf"])
        m, s2 = ensemble_stats(d)
        y = np.clip(d["cf"].to_numpy(), 0.0, 1.0)
        rows = []
        for (zone, hour), idx in d.groupby(["zone", "target_hour_utc"]).indices.items():
            theta, ok, loss = self._fit_one(y[idx], m[idx], s2[idx])
            rows.append({"zone": zone, "target_hour_utc": hour, "a": theta[0], "b": theta[1], "c": theta[2],
                         "d": theta[3], "converged": ok, "loss": loss, "n": len(idx)})
        self.params = pd.DataFrame(rows).set_index(["zone", "target_hour_utc"]).sort_index()
        return self

    def summary(self):
        return self.params.copy()

    def predict(self, df):
        keys = pd.MultiIndex.from_arrays([df["zone"], df["target_hour_utc"]])
        p = self.params.reindex(keys)
        if p[["a", "b", "c", "d"]].isna().any().any():
            raise ValueError("no fitted EMOS parameters for some (zone, hour) in df")
        m, s2 = ensemble_stats(df)
        mu, sigma = location_scale(p[["a", "b", "c", "d"]].to_numpy().T, m, s2)
        cap = df["capacity_mw"].to_numpy()

        quantiles = cn.quantile(QUANTILE_LEVELS[None, :], mu[:, None], sigma[:, None]) * cap[:, None]
        point = cn.mean(mu, sigma) * cap
        # The CRPS scales with the unit: CRPS in MW = capacity x CRPS of the CF forecast at y / capacity
        crps_fn = lambda y: cap * cn.crps(np.asarray(y, dtype=float) / cap, mu, sigma)
        pit_fn = lambda y, rng=None: cn.pit(np.asarray(y, dtype=float) / cap, mu, sigma, rng)
        return Forecast(keys=df, point=point, quantiles=quantiles, crps_fn=crps_fn, pit_fn=pit_fn)
