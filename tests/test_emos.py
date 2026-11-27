import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats
from windai.models import censored_normal as cn
from windai.models.EMOS import EMOS, ALL_MEMBERS


# Numerical CRPS: integral of (F(x) - 1{x >= y})^2 over the support [0, 1] (plus any part of y outside it)
def crps_numeric(y, mu, sigma):
    f = lambda x: (cn.cdf(x, mu, sigma) - (x >= y)) ** 2
    lo, hi = min(0.0, y), max(1.0, y)
    pts = [p for p in (0.0, y, 1.0) if lo <= p <= hi]
    return integrate.quad(f, lo, hi, points=pts, limit=200)[0]


@pytest.mark.parametrize("mu, sigma, y", [
    (0.4, 0.1, 0.35), (0.4, 0.1, 0.0), (0.4, 0.1, 1.0), (0.05, 0.1, 0.02), (0.95, 0.08, 1.0),
    (-0.2, 0.15, 0.0), (1.3, 0.2, 0.9), (0.5, 0.6, 0.7), (0.3, 0.02, 0.8),
])
def test_crps_matches_numerical_integration(mu, sigma, y):
    assert cn.crps(y, mu, sigma) == pytest.approx(crps_numeric(y, mu, sigma), abs=1e-7)


# Far from 0 and 1 the censoring has no effect: the plain normal CRPS
def test_crps_equals_normal_far_from_bounds():
    mu, sigma, y = 0.5, 0.03, 0.52
    z = (y - mu) / sigma
    normal = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert cn.crps(y, mu, sigma) == pytest.approx(normal, rel=1e-9)


# Observations outside [0, 1] add their distance to the interval
def test_crps_outside_support():
    assert cn.crps(1.2, 0.5, 0.1) == pytest.approx(cn.crps(1.0, 0.5, 0.1) + 0.2)


def test_mean_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for mu, sigma in [(0.5, 0.2), (0.05, 0.1), (0.97, 0.1), (-0.1, 0.3)]:
        draws = np.clip(rng.normal(mu, sigma, 400_000), 0, 1)
        assert cn.mean(mu, sigma) == pytest.approx(draws.mean(), abs=2e-3)


def test_quantiles_clipped_and_monotone():
    q = cn.quantile(np.linspace(0.01, 0.99, 99), 0.05, 0.2)
    assert (q >= 0).all() and (q <= 1).all() and (np.diff(q) >= 0).all()
    assert q[0] == 0.0                                   # P(Y = 0) = Phi(-0.25) > 1 %


# Draws from the forecast itself, including the point masses at 0 and 1, give a uniform PIT
def test_pit_uniform_when_calibrated():
    rng = np.random.default_rng(1)
    mu = rng.uniform(-0.1, 1.1, 50_000)
    sigma = rng.uniform(0.05, 0.3, 50_000)
    y = np.clip(rng.normal(mu, sigma), 0, 1)
    assert ((y == 0).mean() > 0.05) and ((y == 1).mean() > 0.05)
    counts, _ = np.histogram(cn.pit(y, mu, sigma, rng=2), bins=10, range=(0, 1))
    np.testing.assert_allclose(counts / len(y), 0.1, atol=0.006)


def test_log_likelihood_point_masses():
    assert cn.log_likelihood(0.0, 0.2, 0.1) == pytest.approx(np.log(stats.norm.cdf(-2.0)))
    assert cn.log_likelihood(1.0, 0.8, 0.1) == pytest.approx(np.log(stats.norm.cdf(-2.0)))
    assert cn.log_likelihood(0.5, 0.5, 0.1) == pytest.approx(stats.norm.logpdf(0.5, 0.5, 0.1))


# Data generated from the EMOS model itself, for one zone and hour
TRUE = (0.05, 0.9, 0.06, 1.5)                          # a, b, c, d


def model_data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    centre = rng.uniform(0.05, 0.9, n)
    spread = rng.uniform(0.01, 0.12, n)
    members = np.clip(centre[:, None] + spread[:, None] * rng.standard_normal((n, 15)), 0, 1)
    m, s2 = members.mean(1), members[:, 1:].var(1, ddof=1)
    a, b, c, d = TRUE
    y = np.clip(rng.normal(a + b * m, np.sqrt(c * c + d * d * s2)), 0, 1)
    df = pd.DataFrame(members, columns=ALL_MEMBERS)
    df["zone"], df["target_hour_utc"], df["cf"] = "Z", 0, y
    return df


@pytest.mark.parametrize("objective", ["ml", "crps"])
def test_fit_recovers_parameters(objective):
    params = EMOS(objective).fit(model_data()).summary().iloc[0]
    assert params["converged"]
    for name, true, tol in zip("abcd", TRUE, (0.02, 0.05, 0.01, 0.25)):
        assert params[name] == pytest.approx(true, abs=tol), name


# The attached CRPS and PIT work in MW: CRPS scales with capacity, the PIT does not change
def test_forecast_functions_in_mw():
    df = model_data(n=500).assign(capacity_mw=200.0, issue_time=pd.Timestamp("2023-01-01", tz="UTC"),
                                  target_time=pd.date_range("2023-01-03", periods=500, freq="h", tz="UTC"))
    model = EMOS().fit(df)
    fc = model.predict(df.drop(columns="cf"))
    y_mw = df["cf"].to_numpy() * 200.0
    p = model.summary().iloc[0]
    m, s2 = df[ALL_MEMBERS].mean(axis=1).to_numpy(), df[ALL_MEMBERS[1:]].var(axis=1, ddof=1).to_numpy()
    mu, sigma = p["a"] + p["b"] * m, np.sqrt(p["c"] ** 2 + p["d"] ** 2 * s2)
    np.testing.assert_allclose(fc.crps_fn(y_mw), 200.0 * cn.crps(df["cf"].to_numpy(), mu, sigma))
    np.testing.assert_allclose(fc.pit_fn(y_mw, rng=0), cn.pit(df["cf"].to_numpy(), mu, sigma, rng=0))
    np.testing.assert_allclose(fc.point, 200.0 * cn.mean(mu, sigma))
