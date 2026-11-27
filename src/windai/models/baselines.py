import numpy as np
import pandas as pd
from windai.models.base import Model
from windai.evaluation.forecast import Forecast, QUANTILE_LEVELS, quantiles_from_samples

MEMBER_POWER_COLS = [f"p_{i:02d}" for i in range(15)]

# Baselines. The tests in tests/test_models.py define what "done" means.
# Units: members p_XX, cf and last_cf are capacity factors; power, last_power and capacity_mw are MW. The
# Forecast is in MW, so CF values are multiplied by the row's capacity_mw (capacity at issue time).


def _to_mw(cf, df):
    return np.asarray(cf) * df["capacity_mw"].to_numpy()[:, None] if np.ndim(cf) == 2 \
        else np.asarray(cf) * df["capacity_mw"].to_numpy()


# Probabilistic persistence: the last observation known at issue time (last_cf, labelled <= 08Z on D) plus the
# empirical distribution of training errors cf - last_cf per zone and UTC target hour. The information set is
# fixed at 09Z for every issue, so the target hour also fixes the horizon from the last observation.
# Quantiles are clipped to [0, 1] (CF), which keeps them non-decreasing. Point forecast = last_power.
# Rows without any earlier observation (only the first eval days, before eval power starts on 31 May 2025)
# fall back to the zone's training mean CF.
class Persistence(Model):
    name = "persistence"

    def fit(self, train_df):
        d = train_df.dropna(subset=["cf", "last_cf"])
        err = d["cf"] - d["last_cf"]
        self.error_quantiles = {key: np.quantile(e.to_numpy(), QUANTILE_LEVELS)
                                for key, e in err.groupby([d["zone"], d["target_hour_utc"]])}
        self.zone_mean = d.groupby("zone")["cf"].mean().to_dict()
        return self

    def predict(self, df):
        last = df["last_cf"].to_numpy().copy()
        missing = np.isnan(last)
        last[missing] = df.loc[missing, "zone"].map(self.zone_mean).to_numpy()
        errors = np.stack([self.error_quantiles[(z, h)] for z, h in zip(df["zone"], df["target_hour_utc"])])
        quantiles = np.clip(last[:, None] + errors, 0.0, 1.0)
        point = np.where(missing, last * df["capacity_mw"].to_numpy(), df["last_power"].to_numpy())
        return Forecast(keys=df, point=point, quantiles=_to_mw(quantiles, df))


# Climatology: the empirical distribution of capacity factor in the training data, per zone x month x UTC hour of
# the target, pooling hours h-1, h and h+1 (~270 values per cell with three training years). Forecast
# quantiles = the cell's CF quantiles x capacity_mw of the row being forecast, so capacity growth is taken into
# account. Point = the cell's mean. A cell with fewer than MIN_VALUES values falls back to the same hour over
# all months, then to the whole zone (never needed with the real training folds, which cover every month).
class Climatology(Model):
    name = "climatology"
    MIN_VALUES = 30

    def fit(self, train_df):
        d = train_df.dropna(subset=["cf"])
        self.table = {}
        for zone, g in d.groupby("zone"):
            cf = g["cf"].to_numpy()
            month = g["target_time"].dt.month.to_numpy()
            hour = g["target_hour_utc"].to_numpy()
            for h in range(24):
                near = np.isin(hour, [(h - 1) % 24, h, (h + 1) % 24])
                for m in range(1, 13):
                    for values in (cf[near & (month == m)], cf[near], cf):
                        if len(values) >= self.MIN_VALUES:
                            break
                    self.table[(zone, m, h)] = (np.quantile(values, QUANTILE_LEVELS), values.mean())
        return self

    def predict(self, df):
        cells = [self.table[key] for key in zip(df["zone"], df["target_time"].dt.month, df["target_hour_utc"])]
        quantiles = np.stack([c[0] for c in cells])
        point = np.array([c[1] for c in cells])
        return Forecast(keys=df, point=_to_mw(point, df), quantiles=_to_mw(quantiles, df))


# Raw power ensemble: the 15 members p_00..p_14 (zone CF from the first stage) x capacity_mw, used
# directly as the forecast, with no post-processing. The members are returned as `samples`, so the exact
# ensemble CRPS is reported too. Point = ensemble mean. This is the reference EMOS has to beat.
class RawEnsemble(Model):
    name = "raw_ensemble"

    def fit(self, train_df):
        return self                                   # nothing to estimate

    def predict(self, df):
        samples = _to_mw(df[MEMBER_POWER_COLS].to_numpy(), df)
        return Forecast(keys=df, point=samples.mean(axis=1), quantiles=quantiles_from_samples(samples),
                        samples=samples)
