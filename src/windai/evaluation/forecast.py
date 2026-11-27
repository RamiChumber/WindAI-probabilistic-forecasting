from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Common forecast format for every model: a point forecast plus 99 quantiles (0.01-0.99), as in GEFCom2014.
# Models that produce samples (raw ensemble, BNN) can attach them, and models with a closed-form CRPS (EMOS)
# can attach that, so the exact scores from their own literature can be reported alongside.
QUANTILE_LEVELS = np.arange(1, 100) / 100
FORECAST_KEYS = ["issue_time", "zone", "target_time"]


@dataclass
class Forecast:
    keys: pd.DataFrame          # one row per forecast, columns FORECAST_KEYS
    point: np.ndarray           # (n,) point forecast (mean or median), scored with RMSE/MAE
    quantiles: np.ndarray       # (n, K) quantiles at `levels`, non-decreasing along each row
    samples: np.ndarray | None = None   # (n, M) optional draws or ensemble members
    # Optional closed-form CRPS: a function y (n,) -> CRPS (n,). predict() never sees the observations, so the
    # model hands over a function and score() calls it with y.
    crps_fn: Callable[[np.ndarray], np.ndarray] | None = None
    levels: np.ndarray = field(default_factory=lambda: QUANTILE_LEVELS.copy())
    # Optional exact (randomised) PIT: a function (y (n,), rng) -> PIT (n,), for models with a closed-form CDF.
    # Without it, the PIT is reconstructed from the quantiles (scoring.pit_from_quantiles).
    pit_fn: Callable[..., np.ndarray] | None = None

    def __post_init__(self):
        self.keys = self.keys[FORECAST_KEYS].reset_index(drop=True)
        self.point = np.asarray(self.point, dtype=float)
        self.quantiles = np.asarray(self.quantiles, dtype=float)
        self.levels = np.asarray(self.levels, dtype=float)
        n = len(self.keys)

        if self.point.shape != (n,):
            raise ValueError(f"point has shape {self.point.shape}, expected ({n},)")
        if self.quantiles.shape != (n, len(self.levels)):
            raise ValueError(f"quantiles has shape {self.quantiles.shape}, expected ({n}, {len(self.levels)})")
        if not (np.all(np.diff(self.levels) > 0) and self.levels[0] > 0 and self.levels[-1] < 1):
            raise ValueError("levels must be strictly increasing and inside (0, 1)")
        if not (np.isfinite(self.point).all() and np.isfinite(self.quantiles).all()):
            raise ValueError("point and quantiles must be finite")
        # Small negative steps from floating-point noise are allowed; real quantile crossing is not
        if np.any(np.diff(self.quantiles, axis=1) < -1e-9):
            raise ValueError("quantiles must be non-decreasing along each row (quantile crossing)")

        if self.samples is not None:
            self.samples = np.asarray(self.samples, dtype=float)
            if self.samples.ndim != 2 or self.samples.shape[0] != n:
                raise ValueError(f"samples has shape {self.samples.shape}, expected ({n}, M)")

    def __len__(self):
        return len(self.keys)


# Turns samples (n, M) into quantiles (n, K). With few samples (15 members) the empirical quantile function
# is a step function; linear interpolation between order statistics smooths it.
def quantiles_from_samples(samples, levels=QUANTILE_LEVELS):
    samples = np.asarray(samples, dtype=float)
    return np.quantile(samples, levels, axis=1).T
