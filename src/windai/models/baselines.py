import numpy as np
import pandas as pd
from windai.models.base import Model
from windai.evaluation.forecast import Forecast, QUANTILE_LEVELS, quantiles_from_samples

MEMBER_POWER_COLS = [f"p_{i:02d}" for i in range(15)]

# Baselines. The tests in tests/test_models.py define what "done" means.
# Useful building blocks: Forecast(keys=df, point=..., quantiles=..., samples=...), QUANTILE_LEVELS and
# quantiles_from_samples() from windai.evaluation.forecast.
# Units (D9): members p_XX, cf and last_cf are capacity factors; power, last_power and capacity_mw are MW. The
# Forecast is in MW, so CF values are multiplied by the row's capacity_mw (capacity at issue time, D1).


# Persistence: the last power observation known at issue time (column `last_power`) is the forecast for every
# target hour. As a probabilistic forecast it is a point mass, so all 99 quantiles equal `last_power` and its
# CRPS equals its absolute error.
# fit() has nothing to estimate.
class Persistence(Model):
    name = "persistence"

    def fit(self, train_df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError


# Climatology: the empirical distribution of capacity factor (column `cf`) in the training data, per
# zone x month x UTC hour of target_time (`target_hour_utc`, D4). Forecast quantiles = climatological CF
# quantiles x capacity_mw of the row being forecast, so growth in installed capacity is taken into account.
# Point = mean (or median). Still to decide (and note in decisions.md): what to do with a (zone, month, hour)
# cell that has too few training rows.
class Climatology(Model):
    name = "climatology"

    def fit(self, train_df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError


# Raw power ensemble: the 15 members p_00..p_14 (zone CF from the first stage, D5/D7), x capacity_mw, used
# directly as the forecast, with no post-processing. Return the members as `samples` so the exact ensemble
# CRPS is also reported. This is the reference EMOS has to beat.
class RawEnsemble(Model):
    name = "raw_ensemble"

    def fit(self, train_df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError
