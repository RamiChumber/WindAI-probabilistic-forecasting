import numpy as np
import pandas as pd
import pytest
from windai.evaluation import dependence as dep
from windai.models.EMOS import ALL_MEMBERS, fit_censored, score_contributions

PHI = 0.6


def ar1(n=20_000, phi=PHI, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi * phi)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x, index=pd.date_range("2000-01-01", periods=n, freq="D"))


def test_acf_of_ar1():
    rho = dep.acf(ar1(), max_lag=5)
    np.testing.assert_allclose(rho, PHI ** np.arange(1, 6), atol=0.03)


# A missing day is a gap: removing one day leaves the lag-1 and lag-2 estimates essentially unchanged, while
# shifting the later values one day earlier would break them
def test_acf_gap_does_not_shift_lags():
    x = ar1(n=3000)
    rho = dep.acf(x.drop(x.index[1500]), max_lag=2)
    np.testing.assert_allclose(rho, dep.acf(x, max_lag=2), atol=0.01)


def test_n_eff_of_ar1():
    x = ar1()
    expected = len(x) * (1 - PHI) / (1 + PHI)
    assert dep.n_eff(dep.acf(x), len(x)) == pytest.approx(expected, rel=0.1)
    assert dep.n_eff(np.array([-0.1, 0.5]), 100) == 100          # stops at the first non-positive lag


def test_long_run_cov():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((500, 3))
    np.testing.assert_allclose(dep.long_run_cov(S, 0), np.cov(S.T, bias=True))
    x = ar1().to_numpy()[:, None]
    assert dep.long_run_cov(x, 50)[0, 0] == pytest.approx(1 / (1 - PHI) ** 2, rel=0.15)


# With dates, a missing day contributes nothing: the result equals the series with a zero at that day
def test_long_run_cov_with_gap():
    rng = np.random.default_rng(2)
    S = rng.standard_normal((50, 2))
    dates = pd.date_range("2021-01-01", periods=51, freq="D").delete(20)
    got = dep.long_run_cov(S, 3, dates=dates)
    centred = S - S.mean(axis=0)
    filled = np.insert(centred, 20, 0.0, axis=0)
    expected = filled.T @ filled / 50
    for j in range(1, 4):
        G = filled[j:].T @ filled[:-j] / 50
        expected += (1 - j / 4) * (G + G.T)
    np.testing.assert_allclose(got, expected)


# Issues from 23 Mar 2023 cover the spring DST day (26 Mar, 23 hours) and the autumn one is not included
def test_z_matrix_dst():
    from windai.utils.synthetic import make_synthetic_dataset
    df = make_synthetic_dataset(n_issues=10, start="2023-03-20")
    z = np.arange(len(df), dtype=float)
    mats = dep.z_matrix(df, z)
    m = mats["ELSPOT NO1"]
    assert m.shape == (10, 24) and list(m.columns) == list(range(24))
    short_day = pd.Timestamp("2023-03-24")                     # D+2 = 26 March
    assert m.loc[short_day].isna().sum() == 1 and m.drop(index=short_day).notna().all().all()


# Data from the EMOS model itself: the scores average to zero at the ML estimate, and with independent rows and a
# correct model the sandwich variance equals the inverse Fisher information (ratios near 1)
def test_scores_and_sandwich_on_correct_model():
    rng = np.random.default_rng(3)
    n = 6000
    centre, spread = rng.uniform(0.05, 0.9, n), rng.uniform(0.01, 0.12, n)
    members = np.clip(centre[:, None] + spread[:, None] * rng.standard_normal((n, 15)), 0, 1)
    m, s2 = members.mean(1), members[:, 1:].var(1, ddof=1)
    y = np.clip(rng.normal(0.05 + 0.9 * m, np.sqrt(0.06 ** 2 + 1.5 ** 2 * s2)), 0, 1)
    theta = fit_censored(y, m, s2)[0]
    S = score_contributions(theta, y, m, s2)
    assert np.abs(S.mean(axis=0)).max() < 1e-3
    A = dep.numerical_hessian(lambda t: -score_contributions(t, y, m, s2).mean(axis=0), theta)
    ratio, top = dep.sandwich_ratios(A, dep.long_run_cov(S, 0))
    np.testing.assert_allclose(ratio, 1.0, atol=0.15)
    assert top == pytest.approx(1.0, abs=0.25)


# Two rows on the same day form one block: the result equals the series with the two rows added together
def test_long_run_cov_same_day_rows_summed():
    rng = np.random.default_rng(4)
    S = rng.standard_normal((6, 2))
    dates = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"])
    c = S - S.mean(axis=0)
    blocks = np.vstack([c[0], c[1] + c[2], c[3], c[4], c[5]])
    G = blocks[1:].T @ blocks[:-1] / 6
    np.testing.assert_allclose(dep.long_run_cov(S, 1, dates=dates), blocks.T @ blocks / 6 + 0.5 * (G + G.T))


# Under equal expected loss with AR(1)-dependent daily differences, the Newey-West test keeps roughly its 5 %
# size, while ignoring the dependence (lag 0) rejects far too often; a real difference is detected
def test_diebold_mariano_size_and_power():
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    reject = {0: 0, 7: 0}
    reps = 300
    for r in range(reps):
        d = ar1(n=365, phi=0.5, seed=100 + r).to_numpy()
        for lag in reject:
            reject[lag] += dep.diebold_mariano(d, np.zeros(365), dates, lag=lag)["p"] < 0.05
    assert reject[7] / reps < 0.12 and reject[0] / reps > 0.15
    d = ar1(n=365, phi=0.5, seed=7).to_numpy() - 0.6
    res = dep.diebold_mariano(d, np.ones(365), dates)
    assert res["p"] < 0.01 and res["mean_diff"] < 0


# Several rows per day are averaged into one daily difference
def test_diebold_mariano_daily_mean():
    dates = np.repeat(pd.date_range("2023-01-01", periods=3, freq="D"), 2)
    res = dep.diebold_mariano(np.array([1.0, 3.0, 2.0, 4.0, 0.0, 2.0]), np.zeros(6), dates, lag=0)
    assert res["n_days"] == 3 and res["mean_diff"] == pytest.approx(2.0) and np.isfinite(res["stat"])


# Benjamini-Hochberg by hand: p sorted (0.01, 0.02, 0.03, 0.5) with m = 4 gives 0.04, 0.04, 0.04, 0.5
def test_benjamini_hochberg_by_hand():
    adj = dep.benjamini_hochberg([0.03, 0.5, 0.01, 0.02, np.nan])
    np.testing.assert_allclose(adj[:4], [0.04, 0.5, 0.04, 0.04])
    assert np.isnan(adj[4])
