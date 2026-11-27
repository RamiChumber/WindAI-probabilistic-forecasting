import numpy as np
from scipy.special import ndtr, ndtri, log_ndtr

# Normal distribution censored to [0, 1]: a latent Y* ~ N(mu, sigma^2) is observed as Y = min(max(Y*, 0), 1).
# Probability below 0 becomes a point mass at 0, probability above 1 a point mass at 1. This is the predictive
# distribution of EMOS for capacity factor. All functions are vectorised; mu, sigma and y broadcast together.

SQRT2 = np.sqrt(2.0)
SQRT_PI = np.sqrt(np.pi)
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _pdf(t):
    return np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)


def _bounds(mu, sigma):
    """Standardised positions of the censoring points 0 and 1."""
    return -mu / sigma, (1.0 - mu) / sigma


def cdf(y, mu, sigma):
    y = np.asarray(y, dtype=float)
    return np.where(y < 0, 0.0, np.where(y >= 1, 1.0, ndtr((y - mu) / sigma)))


# Quantile function: the latent normal's quantile, clipped to [0, 1]. Non-decreasing in tau.
def quantile(tau, mu, sigma):
    return np.clip(mu + sigma * ndtri(tau), 0.0, 1.0)


# Mean of the censored variable: 0 * P(Y=0) + E[Y*; 0 < Y* < 1] + 1 * P(Y=1)
def mean(mu, sigma):
    lo, hi = _bounds(mu, sigma)
    return mu * (ndtr(hi) - ndtr(lo)) + sigma * (_pdf(lo) - _pdf(hi)) + (1.0 - ndtr(hi))


# Antiderivatives used by the CRPS: G' = Phi^2, H' = (1 - Phi)^2
def _G(t):
    p = ndtr(t)
    return t * p * p + 2 * p * _pdf(t) - ndtr(SQRT2 * t) / SQRT_PI


def _H(t):
    return -_G(-t)


# Closed-form CRPS: integral of (F(x) - 1{x >= y})^2 over x. F is 0 below 0 and 1 from 1 on, so only [0, 1]
# contributes when 0 <= y <= 1; in standardised units t = (x - mu) / sigma this is
#   sigma * ( int_lo^z Phi(t)^2 dt + int_z^hi (1 - Phi(t))^2 dt ),   z = clip((y - mu) / sigma, lo, hi).
# An observation outside [0, 1] adds the distance to the interval (F is 0 or 1 there).
def crps(y, mu, sigma):
    y = np.asarray(y, dtype=float)
    lo, hi = _bounds(mu, sigma)
    z = np.clip((y - mu) / sigma, lo, hi)
    inside = sigma * ((_G(z) - _G(lo)) + (_H(hi) - _H(z)))
    return inside + np.maximum(y - 1.0, 0.0) + np.maximum(-y, 0.0)


# Log-likelihood of observations in [0, 1]: density of the latent normal for interior values, probability of
# the point mass for y <= 0 and y >= 1.
def log_likelihood(y, mu, sigma):
    y = np.asarray(y, dtype=float)
    z = (y - mu) / sigma
    interior = -0.5 * z * z - np.log(sigma) - LOG_SQRT_2PI
    return np.where(y <= 0, log_ndtr(-mu / sigma), np.where(y >= 1, log_ndtr((mu - 1.0) / sigma), interior))


# Randomised PIT: F(y) for interior values; at a point mass, uniform over the jump of F (0 to P(Y=0) at y = 0,
# P(Y<1) to 1 at y = 1), so a calibrated forecast gives a uniform PIT despite the point masses.
def pit(y, mu, sigma, rng=None):
    rng = np.random.default_rng(rng)
    y = np.asarray(y, dtype=float)
    lo, hi = _bounds(mu, sigma)
    y, lo, hi = np.broadcast_arrays(y, lo, hi)
    u = rng.uniform(size=y.shape)
    at_zero = u * ndtr(lo)
    at_one = ndtr(hi) + u * (1.0 - ndtr(hi))
    return np.where(y <= 0, at_zero, np.where(y >= 1, at_one, ndtr((y - mu) / sigma)))
