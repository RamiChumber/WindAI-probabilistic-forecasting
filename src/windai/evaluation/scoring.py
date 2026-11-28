import numpy as np

# Scoring rules. Everything is vectorised over forecasts: y has shape (n,), quantiles (n, K), samples (n, M).
# Per-row scores are returned as arrays, so they can be averaged over any grouping (zone, lead time, month).
# Lower is better for every score here except coverage, which should match its nominal level.


def rmse(point, y):
    return float(np.sqrt(np.mean((np.asarray(point) - np.asarray(y)) ** 2)))


def mae(point, y):
    return float(np.mean(np.abs(np.asarray(point) - np.asarray(y))))


# Pinball (quantile) loss of quantile forecast q at level tau. It penalises under-prediction with weight tau
# and over-prediction with weight 1 - tau, so its expectation is minimised by the true tau-quantile.
# q: (n,) or (n, K), y: (n,), tau: scalar or (K,). Returns the same shape as q.
def pinball(q, y, tau):
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    if q.ndim == 2:
        y = y[:, None]
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


# CRPS from quantiles. CRPS(F, y) = 2 * integral over tau in (0, 1) of pinball(F^-1(tau), y, tau), so twice
# the mean pinball loss over an evenly spaced grid of levels approximates it. The mean weights each of the 99
# levels by 1/99 rather than by the grid spacing 0.01, so for smooth distributions the estimate is about 1 %
# above the exact CRPS (normal: ratio 1.010 for any y). For a point forecast (all quantiles equal) it gives
# exactly the absolute error. Compare models on the same estimator.
def crps_quantiles(quantiles, levels, y):
    return 2 * pinball(quantiles, y, np.asarray(levels)).mean(axis=1)


# Exact CRPS of the empirical distribution of M samples: E|X - y| - 1/2 E|X - X'| (energy form).
# E|X - X'| over all M^2 pairs is computed from the sorted samples in O(M log M):
#   sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - M - 1) x_(i),  i = 1..M
# This is the standard (not the "fair") ensemble CRPS, which treats the members as the forecast itself.
def crps_sample(samples, y):
    x = np.sort(np.asarray(samples, dtype=float), axis=1)
    y = np.asarray(y, dtype=float)
    m = x.shape[1]
    term1 = np.abs(x - y[:, None]).mean(axis=1)
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = (x * weights).sum(axis=1) / m**2
    return term1 - term2


# PIT value F(y) reconstructed from the quantiles by linear interpolation between neighbouring levels.
# - y below the lowest quantile or above the highest: the middle of the uncovered tail (e.g. 0.005).
# - y equal to one or more quantiles (point masses, e.g. power = 0 or persistence): F jumps at y, so the
#   PIT is drawn uniformly over the jump (randomised PIT). Without this a calibrated forecast with point
#   masses gives a non-uniform PIT histogram.
def pit_from_quantiles(quantiles, levels, y, rng=None):
    rng = np.random.default_rng(rng)
    q = np.asarray(quantiles, dtype=float)
    y = np.asarray(y, dtype=float)
    levels = np.asarray(levels, dtype=float)
    n, k_levels = q.shape
    ext = np.concatenate([[0.0], levels, [1.0]])

    below = (q < y[:, None]).sum(axis=1)       # quantiles strictly below y
    at_or_below = (q <= y[:, None]).sum(axis=1)
    pit = np.empty(n)

    ties = at_or_below > below
    lo = ext[below[ties]]
    hi = ext[at_or_below[ties] + 1]
    pit[ties] = rng.uniform(lo, hi)

    k = below[~ties]
    rows = np.flatnonzero(~ties)
    inner = (k > 0) & (k < k_levels)
    pit_free = np.where(k == 0, levels[0] / 2, (levels[-1] + 1) / 2)
    r, kk = rows[inner], k[inner]
    q_lo, q_hi = q[r, kk - 1], q[r, kk]
    frac = (y[r] - q_lo) / (q_hi - q_lo)
    pit_free[inner] = levels[kk - 1] + frac * (levels[kk] - levels[kk - 1])
    pit[rows] = pit_free
    return pit


# The quantile at level tau for every row, shape (n,). On the grid this is the stored column. Between two grid
# levels it is interpolated linearly, e.g. tau = 0.025 lies halfway between 0.02 and 0.03, so it is the mean of
# those two quantiles. Over a 1 % step the quantile function is close to a straight line, so the error is small.
# Levels outside the grid (below 0.01 or above 0.99) raise: the tails beyond it are not stored.
def quantile_at(quantiles, levels, tau):
    q = np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    on_grid = np.flatnonzero(np.isclose(levels, tau))
    if len(on_grid):
        return q[:, on_grid[0]]
    if not levels[0] < tau < levels[-1]:
        raise ValueError(f"level {tau} is outside the quantile grid [{levels[0]}, {levels[-1]}]")
    k = np.searchsorted(levels, tau)            # levels[k - 1] < tau < levels[k]
    w = (tau - levels[k - 1]) / (levels[k] - levels[k - 1])
    return (1 - w) * q[:, k - 1] + w * q[:, k]


# Whether y falls inside the central interval of the given width, e.g. width=0.8 uses the 0.1 and 0.9
# quantiles. Averaged over rows this is the empirical coverage. Bounds that fall between grid levels are
# interpolated: the 95 % interval uses 0.025 (between 0.02 and 0.03) and 0.975 (between 0.97 and 0.98).
def coverage(quantiles, levels, y, width):
    y = np.asarray(y, dtype=float)
    lower = quantile_at(quantiles, levels, (1 - width) / 2)
    upper = quantile_at(quantiles, levels, (1 + width) / 2)
    return (lower <= y) & (y <= upper)


def _level_index(levels, tau):
    idx = np.flatnonzero(np.isclose(levels, tau))
    if len(idx) == 0:
        raise ValueError(f"level {tau} is not on the quantile grid")
    return idx[0]


# Mean width of the central interval of the given width, per row (same bounds as coverage)
def interval_width(quantiles, levels, width):
    return quantile_at(quantiles, levels, (1 + width) / 2) - quantile_at(quantiles, levels, (1 - width) / 2)


# P(Y > x) per row, read off the quantile grid: the CDF at x is interpolated linearly between the two grid
# quantiles around x. Beyond the grid the tail is not stored, so x below the 0.01 quantile gets F = 0.005 and x
# above the 0.99 quantile gets F = 0.995 (the middle of the unstored tail).
def exceedance_prob(quantiles, levels, x):
    q = np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    x = np.broadcast_to(np.asarray(x, dtype=float), (len(q),))
    k = (q <= x[:, None]).sum(axis=1)                    # number of grid quantiles at or below x
    inner = np.clip(k, 1, len(levels) - 1)
    lo, hi = q[np.arange(len(q)), inner - 1], q[np.arange(len(q)), inner]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(hi > lo, (x - lo) / (hi - lo), 1.0)
    F = levels[inner - 1] + np.clip(w, 0, 1) * (levels[inner] - levels[inner - 1])
    F = np.where(k == 0, levels[0] / 2, np.where(k == len(levels), 1 - (1 - levels[-1]) / 2, F))
    return 1 - F


# Brier score of probability forecasts p for a binary event
def brier(p, event):
    return (np.asarray(p, dtype=float) - np.asarray(event, dtype=float)) ** 2
