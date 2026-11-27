import datetime
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


# ---------------------------------------------------------------- rolling EMOS

from windai.models.EMOS import RollingEMOS
from windai.data.splits import history_rows, training_rows, evaluation_rows


def issue_rows(df, day):
    return df.loc[df["issue_date"] == day].reset_index(drop=True)


# The forecast for issue D may only depend on issues up to D-3: changing later observations (D-2 onwards, all of
# the evaluated fold included) leaves it unchanged, changing D-3 changes it
def test_rolling_uses_nothing_after_d_minus_3(synthetic_df):
    history = history_rows(synthetic_df, "val")
    day = datetime.date(2023, 4, 20)
    target = issue_rows(synthetic_df, day).drop(columns=["power", "cf"])
    base = RollingEMOS(30).fit(history).predict(target).quantiles

    late = history.copy()
    after = late["issue_date"] >= day - datetime.timedelta(days=2)
    late.loc[after, "cf"] = np.random.default_rng(0).uniform(0, 1, after.sum())
    np.testing.assert_array_equal(RollingEMOS(30).fit(late).predict(target).quantiles, base)

    edge = history.copy()
    at = edge["issue_date"] == day - datetime.timedelta(days=3)
    edge.loc[at, "cf"] = np.random.default_rng(1).uniform(0, 1, at.sum())
    assert not np.allclose(RollingEMOS(30).fit(edge).predict(target).quantiles, base)


# A 30-day window with pooled hours uses exactly the issues D-32 .. D-3 and hours h-1, h, h+1 (wrapping at midnight)
def test_rolling_window_bounds_and_pooling(synthetic_df):
    history = history_rows(synthetic_df, "val")
    day = datetime.date(2023, 4, 20)
    model = RollingEMOS(30).fit(history)
    model.predict(issue_rows(synthetic_df, day).drop(columns=["power", "cf"]))
    p = model.params_by_issue
    for hour, hours in [(5, [4, 5, 6]), (0, [23, 0, 1])]:
        zone = "ELSPOT NO2"
        lo, hi = day - datetime.timedelta(days=32), day - datetime.timedelta(days=3)
        expected = ((history["zone"] == zone) & history["target_hour_utc"].isin(hours)
                    & (history["issue_date"] >= lo) & (history["issue_date"] <= hi)).sum()
        got = p.loc[(p["zone"] == zone) & (p["target_hour_utc"] == hour), "n"].iloc[0]
        assert got == expected


# The expanding window uses all history up to D-3 and doesn't pool hours
def test_expanding_window(synthetic_df):
    history = history_rows(synthetic_df, "val")
    day = datetime.date(2023, 4, 20)
    model = RollingEMOS(None).fit(history)
    model.predict(issue_rows(synthetic_df, day).drop(columns=["power", "cf"]))
    p = model.params_by_issue.iloc[0]
    expected = ((history["zone"] == p["zone"]) & (history["target_hour_utc"] == p["target_hour_utc"])
                & (history["issue_date"] <= day - datetime.timedelta(days=3))).sum()
    assert p["n"] == expected and not model.pool_hours


# For val's first issue (1 Apr), the expanding window holds exactly the purged training rows of the static fit,
# so both give the same parameters
def test_expanding_matches_static_on_first_val_issue(synthetic_df):
    day = datetime.date(2023, 4, 1)
    static = EMOS().fit(training_rows(synthetic_df, "val")).summary()
    rolling = RollingEMOS(None).fit(history_rows(synthetic_df, "val"))
    rolling.predict(issue_rows(synthetic_df, day).drop(columns=["power", "cf"]))
    r = rolling.params_by_issue.set_index(["zone", "target_hour_utc"])
    np.testing.assert_allclose(r[["a", "b", "c", "d"]].to_numpy(),
                               static.loc[r.index, ["a", "b", "c", "d"]].to_numpy(), rtol=1e-6, atol=1e-8)


# Data from 10 Mar to 18 Apr 2024 spans the end of the val fold: the val history keeps March, drops April (test)
def test_history_rows_stop_at_fold_end():
    from windai.utils.synthetic import make_synthetic_dataset
    df = make_synthetic_dataset(n_issues=40, start="2024-03-10")
    h = history_rows(df, "val")
    assert h["issue_date"].max() == datetime.date(2024, 3, 31)
    assert set(h["fold"]) == {"val"} and h["power"].notna().all()
    assert set(history_rows(df, "test")["fold"]) == {"val", "test"}


# The runner hands a sequential model the history, and its forecasts cover exactly the evaluated rows
def test_runner_with_rolling_model(synthetic_df):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    from run_baselines import run
    res = run(synthetic_df, fit="val", models=[lambda: RollingEMOS(30)])
    n_val = evaluation_rows(synthetic_df, "val")["power"].notna().sum()
    assert res.loc[("emos_rolling_30", "all", "all"), "n"] == n_val


# The analytic gradient of the log-likelihood matches finite differences, including the point masses
@pytest.mark.parametrize("y", [0.0, 0.3, 1.0])
def test_log_likelihood_grad(y):
    mu, sigma, eps = 0.2, 0.15, 1e-6
    d_mu, d_sigma = cn.log_likelihood_grad(y, mu, sigma)
    num_mu = (cn.log_likelihood(y, mu + eps, sigma) - cn.log_likelihood(y, mu - eps, sigma)) / (2 * eps)
    num_sigma = (cn.log_likelihood(y, mu, sigma + eps) - cn.log_likelihood(y, mu, sigma - eps)) / (2 * eps)
    assert d_mu == pytest.approx(num_mu, rel=1e-5) and d_sigma == pytest.approx(num_sigma, rel=1e-5)


# ---------------------------------------------------------------- seasonal EMOS

from windai.models.EMOS import SeasonalEMOS, SEASONAL_PARAMS, season_basis, seasonal_location_scale

TRUE_SEASONAL = dict(a0=0.05, a1=0.03, a2=-0.02, b0=0.9, b1=-0.1, b2=0.05, c=0.06, d=1.5, g1=0.4, g2=-0.3)


def seasonal_data(seed=0, n=8000, seasonal=True):
    df = model_data(seed, n)
    df["target_time"] = pd.Timestamp("2021-01-01", tz="UTC") + pd.to_timedelta(
        np.random.default_rng(seed + 1).uniform(0, 3 * 365, n), unit="D")
    if seasonal:
        theta = np.array([TRUE_SEASONAL[k] for k in SEASONAL_PARAMS])
        m, s2 = df[ALL_MEMBERS].mean(axis=1).to_numpy(), df[ALL_MEMBERS[1:]].var(axis=1, ddof=1).to_numpy()
        mu, sigma, _ = seasonal_location_scale(theta, m, s2, season_basis(df["target_time"]))
        df["cf"] = np.clip(np.random.default_rng(seed + 2).normal(mu, sigma), 0, 1)
    return df


def test_seasonal_fit_recovers_parameters():
    p = SeasonalEMOS().fit(seasonal_data()).summary().iloc[0]
    assert p["converged"]
    tol = dict(a0=0.02, a1=0.015, a2=0.015, b0=0.05, b1=0.04, b2=0.04, c=0.01, d=0.25, g1=0.1, g2=0.1)
    for k in SEASONAL_PARAMS:
        assert p[k] == pytest.approx(TRUE_SEASONAL[k], abs=tol[k]), k


# Without a seasonal cycle in the data, the harmonics come out near zero and the rest matches static EMOS
def test_seasonal_without_season_matches_static():
    df = seasonal_data(seasonal=False)
    p = SeasonalEMOS().fit(df).summary().iloc[0]
    s = EMOS().fit(df).summary().iloc[0]
    for k in ("a1", "a2", "b1", "b2"):
        assert abs(p[k]) < 0.03, k
    for k in ("g1", "g2"):
        assert abs(p[k]) < 0.1, k
    for k, ks in (("a0", "a"), ("b0", "b"), ("c", "c"), ("d", "d")):
        assert p[k] == pytest.approx(s[ks], abs=0.03), k


# The analytic gradient of the seasonal objective matches finite differences
def test_seasonal_gradient():
    from scipy.optimize import check_grad
    from windai.models.EMOS import seasonal_nll
    df = seasonal_data(n=500)
    m, s2 = df[ALL_MEMBERS].mean(axis=1).to_numpy(), df[ALL_MEMBERS[1:]].var(axis=1, ddof=1).to_numpy()
    args = (df["cf"].to_numpy(), m, s2, season_basis(df["target_time"]))
    theta = np.array([TRUE_SEASONAL[k] for k in SEASONAL_PARAMS]) + 0.02
    err = check_grad(lambda x: seasonal_nll(x, *args)[0], lambda x: seasonal_nll(x, *args)[1], theta)
    assert err < 1e-5
