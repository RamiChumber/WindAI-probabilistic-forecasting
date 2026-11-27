import itertools
import numpy as np
import pytest
from scipy import stats
from windai.evaluation.forecast import QUANTILE_LEVELS
from windai.evaluation.scoring import (rmse, mae, pinball, crps_quantiles, crps_sample, pit_from_quantiles,
                                       coverage, quantile_at)


def test_rmse_mae():
    point = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 2.0, 6.0])
    assert rmse(point, y) == pytest.approx(np.sqrt((1 + 0 + 9) / 3))
    assert mae(point, y) == pytest.approx((1 + 0 + 3) / 3)


# Pinball loss by hand: under-prediction costs tau per unit, over-prediction 1 - tau per unit
def test_pinball_by_hand():
    assert pinball(np.array([2.0]), np.array([5.0]), 0.9)[0] == pytest.approx(2.7)
    assert pinball(np.array([5.0]), np.array([2.0]), 0.9)[0] == pytest.approx(0.3)
    assert pinball(np.array([5.0]), np.array([2.0]), 0.5)[0] == pytest.approx(1.5)
    assert pinball(np.array([3.0]), np.array([3.0]), 0.1)[0] == 0


def test_pinball_grid_shape():
    q = np.zeros((4, 99))
    assert pinball(q, np.ones(4), QUANTILE_LEVELS).shape == (4, 99)


# M = 1: the ensemble CRPS is the absolute error
def test_crps_sample_single_member():
    np.testing.assert_allclose(crps_sample(np.array([[3.0], [1.0]]), np.array([1.0, 4.0])), [2.0, 3.0])


# M = 2, members {0, 2}, y = 1: E|X - y| = 1, E|X - X'| = (0 + 2 + 2 + 0) / 4 = 1, CRPS = 1 - 1/2 = 0.5
def test_crps_sample_two_members_by_hand():
    assert crps_sample(np.array([[0.0, 2.0]]), np.array([1.0]))[0] == pytest.approx(0.5)


# The sorted O(M log M) formula must agree with the O(M^2) definition
def test_crps_sample_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 15))
    y = rng.normal(size=5)
    brute = [np.abs(x[i] - y[i]).mean() - 0.5 * np.mean([abs(a - b) for a, b in itertools.product(x[i], x[i])])
             for i in range(5)]
    np.testing.assert_allclose(crps_sample(x, y), brute)


# All quantiles equal (point forecast): CRPS is exactly the absolute error
def test_crps_quantiles_point_mass_is_absolute_error():
    q = np.full((3, 99), 4.0)
    np.testing.assert_allclose(crps_quantiles(q, QUANTILE_LEVELS, np.array([4.0, 1.0, 10.0])), [0.0, 3.0, 6.0])


# Quantiles of N(0, 1) must give roughly the closed-form normal CRPS z(2Phi(z) - 1) + 2phi(z) - 1/sqrt(pi).
# The 99-level grid leaves out the outer tails, so allow a small error.
@pytest.mark.parametrize("y", [-2.0, -0.5, 0.0, 1.0, 3.0])
def test_crps_quantiles_approximates_normal_closed_form(y):
    q = stats.norm.ppf(QUANTILE_LEVELS)[None, :]
    exact = y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi)
    approx = crps_quantiles(q, QUANTILE_LEVELS, np.array([y]))[0]
    assert approx == pytest.approx(exact, rel=0.03, abs=0.005)


def test_coverage():
    q = np.tile(QUANTILE_LEVELS, (3, 1))     # quantiles of U(0, 1): the 80 % interval is [0.1, 0.9]
    y = np.array([0.05, 0.5, 0.9])
    np.testing.assert_array_equal(coverage(q, QUANTILE_LEVELS, y, 0.8), [False, True, True])


# On the grid, quantile_at returns the stored column unchanged
def test_quantile_at_on_grid():
    q = np.random.default_rng(0).normal(size=(4, 99)).cumsum(axis=1)
    np.testing.assert_array_equal(quantile_at(q, QUANTILE_LEVELS, 0.1), q[:, 9])


# Off the grid, by hand: q_0.02 = 0.110 and q_0.03 = 0.130, so q_0.025 = 0.120 (halfway); q_0.022 = 0.114
def test_quantile_at_interpolates_by_hand():
    q = np.zeros((1, 99))
    q[0, 1], q[0, 2] = 0.110, 0.130       # levels 0.02 and 0.03
    assert quantile_at(q, QUANTILE_LEVELS, 0.025)[0] == pytest.approx(0.120)
    assert quantile_at(q, QUANTILE_LEVELS, 0.022)[0] == pytest.approx(0.114)


# For N(0, 1) the interpolated 2.5 % quantile is -1.9673 against the exact -1.9600. In probability that is
# Phi(-1.9673) = 0.0246, so the interpolated 95 % interval covers ~95.1 %: slightly wide (the quantile function
# curves outwards in the tails), and far below the sampling noise of a coverage estimate.
def test_quantile_at_close_to_exact_normal():
    q = stats.norm.ppf(QUANTILE_LEVELS)[None, :]
    lo = stats.norm.cdf(quantile_at(q, QUANTILE_LEVELS, 0.025)[0])
    hi = stats.norm.cdf(quantile_at(q, QUANTILE_LEVELS, 0.975)[0])
    assert lo == pytest.approx(0.025, abs=0.001) and lo < 0.025      # outward, i.e. conservative
    assert hi == pytest.approx(0.975, abs=0.001) and hi > 0.975
    assert hi - lo == pytest.approx(0.95, abs=0.002)


# Beyond the outermost stored levels there is nothing to interpolate between
def test_quantile_at_outside_grid():
    with pytest.raises(ValueError, match="outside the quantile grid"):
        quantile_at(np.zeros((1, 99)), QUANTILE_LEVELS, 0.005)


# U(0, 1): the 95 % interval is [0.025, 0.975], built from interpolated bounds
def test_coverage_95_interpolated():
    q = np.tile(QUANTILE_LEVELS, (4, 1))
    y = np.array([0.024, 0.026, 0.974, 0.976])
    np.testing.assert_array_equal(coverage(q, QUANTILE_LEVELS, y, 0.95), [False, True, True, False])


def test_coverage_beyond_grid():
    with pytest.raises(ValueError, match="outside the quantile grid"):
        coverage(np.zeros((1, 99)), QUANTILE_LEVELS, np.zeros(1), 0.99)


def test_pit_interpolates():
    q = np.tile(QUANTILE_LEVELS, (2, 1))     # U(0, 1): PIT = y (0.505 avoids hitting a quantile exactly)
    np.testing.assert_allclose(pit_from_quantiles(q, QUANTILE_LEVELS, np.array([0.355, 0.505])), [0.355, 0.505])


def test_pit_tails():
    q = np.tile(QUANTILE_LEVELS, (2, 1))
    np.testing.assert_allclose(pit_from_quantiles(q, QUANTILE_LEVELS, np.array([-1.0, 2.0])), [0.005, 0.995])


# Observations drawn from the forecast distribution itself give a uniform PIT
def test_pit_uniform_when_calibrated():
    rng = np.random.default_rng(1)
    y = rng.normal(size=20000)
    q = np.tile(stats.norm.ppf(QUANTILE_LEVELS), (len(y), 1))
    pit = pit_from_quantiles(q, QUANTILE_LEVELS, y, rng=2)
    counts, _ = np.histogram(pit, bins=10, range=(0, 1))
    np.testing.assert_allclose(counts / len(y), 0.1, atol=0.01)


# A point mass hit exactly: the PIT is spread uniformly over (0, 1) instead of piling up in one bin
def test_pit_randomised_at_point_mass():
    q = np.zeros((20000, 99))
    pit = pit_from_quantiles(q, QUANTILE_LEVELS, np.zeros(20000), rng=3)
    counts, _ = np.histogram(pit, bins=10, range=(0, 1))
    np.testing.assert_allclose(counts / 20000, 0.1, atol=0.01)
