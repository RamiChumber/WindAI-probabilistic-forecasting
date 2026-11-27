import numpy as np
import pytest
from windai.evaluation.forecast import Forecast, QUANTILE_LEVELS
from windai.models.baselines import Persistence, Climatology, RawEnsemble
from windai.models.EMOS import EMOS, SeasonalEMOS

# These tests define "done" for each model. A model that still raises NotImplementedError is reported as
# xfail ("not implemented yet"), so the number of xfails is your remaining work. Any other error is a failure.
ALL_MODELS = [Persistence, Climatology, RawEnsemble, EMOS, SeasonalEMOS]
MEMBERS = [f"p_{i:02d}" for i in range(15)]


# Fits on the training rows and forecasts the test rows with the target removed (leakage guard)
def fit_predict(model_cls, train_df, test_df):
    model = model_cls()
    try:
        model.fit(train_df)
        return model.predict(test_df.drop(columns=["power", "cf"]))   # the target, in MW and in CF
    except NotImplementedError:
        pytest.xfail(f"{model_cls.__name__} not implemented yet")


@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=lambda c: c.__name__)
def test_contract(model_cls, train_df, test_df):
    forecast = fit_predict(model_cls, train_df, test_df)
    assert isinstance(forecast, Forecast)
    assert len(forecast) == len(test_df)
    # Same rows in the same order as the input
    assert forecast.keys.equals(test_df[["issue_time", "zone", "target_time"]].reset_index(drop=True))
    # Power is bounded by 0 and installed capacity
    capacity = test_df["capacity_mw"].to_numpy()[:, None]
    assert np.all(forecast.quantiles >= -1e-6)
    assert np.all(forecast.quantiles <= capacity + 1e-6)


# Probabilistic persistence: the point forecast is the last observation
def test_persistence_point_is_last_power(train_df, test_df):
    forecast = fit_predict(Persistence, train_df, test_df)
    np.testing.assert_allclose(forecast.point, test_df["last_power"])


# With zero training errors the error distribution is a point mass, so every quantile is the last observation
def test_persistence_without_errors_is_point_mass(train_df, test_df):
    forecast = fit_predict(Persistence, train_df.assign(cf=train_df["last_cf"]), test_df)
    np.testing.assert_allclose(forecast.quantiles, np.repeat(test_df["last_power"].to_numpy()[:, None], 99, axis=1))


# The quantiles are last_cf + the training error quantiles of the row's zone and UTC hour, x capacity
def test_persistence_adds_error_quantiles(train_df, test_df):
    forecast = fit_predict(Persistence, train_df, test_df)
    row = test_df.iloc[0]
    g = train_df[(train_df["zone"] == row["zone"]) & (train_df["target_hour_utc"] == row["target_hour_utc"])]
    err = np.quantile(g["cf"] - g["last_cf"], QUANTILE_LEVELS)
    expected = np.clip(row["last_cf"] + err, 0, 1) * row["capacity_mw"]
    np.testing.assert_allclose(forecast.quantiles[0], expected)


# A row without any earlier observation still gets a finite forecast (the zone's training mean CF)
def test_persistence_without_observation(train_df, test_df):
    rows = test_df.head(3).assign(last_cf=np.nan, last_power=np.nan)
    forecast = fit_predict(Persistence, train_df, rows)
    assert np.isfinite(forecast.point).all() and np.isfinite(forecast.quantiles).all()


# Climatology pools hours h-1, h, h+1 within the month: predicting a training row reproduces that cell
def test_climatology_cell_is_pooled_month_hour(train_df):
    forecast = fit_predict(Climatology, train_df, train_df.head(1))
    row = train_df.iloc[0]
    h = row["target_hour_utc"]
    cell = train_df[(train_df["zone"] == row["zone"]) & (train_df["target_time"].dt.month == row["target_time"].month)
                    & train_df["target_hour_utc"].isin([(h - 1) % 24, h, (h + 1) % 24])]
    np.testing.assert_allclose(forecast.quantiles[0], np.quantile(cell["cf"], QUANTILE_LEVELS) * row["capacity_mw"])
    assert forecast.point[0] == pytest.approx(cell["cf"].mean() * row["capacity_mw"])


# Climatology works on capacity factor, so doubling the installed capacity doubles the forecast
def test_climatology_scales_with_capacity(train_df, test_df):
    base = fit_predict(Climatology, train_df, test_df)
    doubled = fit_predict(Climatology, train_df, test_df.assign(capacity_mw=2 * test_df["capacity_mw"]))
    np.testing.assert_allclose(doubled.quantiles, 2 * base.quantiles)


# Climatology must not depend on the test-period ensemble: changing the members leaves it unchanged
def test_climatology_ignores_ensemble(train_df, test_df):
    base = fit_predict(Climatology, train_df, test_df)
    shuffled = test_df.copy()
    shuffled[MEMBERS] = 0.0
    other = fit_predict(Climatology, train_df, shuffled)
    np.testing.assert_allclose(other.quantiles, base.quantiles)


# The members p_XX are capacity factors; the Forecast is in MW, so the samples are members x capacity
def test_raw_ensemble_returns_members(train_df, test_df):
    forecast = fit_predict(RawEnsemble, train_df, test_df)
    assert forecast.samples is not None
    members_mw = test_df[MEMBERS].to_numpy() * test_df["capacity_mw"].to_numpy()[:, None]
    np.testing.assert_allclose(np.sort(forecast.samples, axis=1), np.sort(members_mw, axis=1))
