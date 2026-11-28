import numpy as np
import pytest
from windai.evaluation.evaluate import score, score_rows
from windai.evaluation.forecast import Forecast, quantiles_from_samples
from windai.evaluation.scoring import rmse

MEMBERS = [f"p_{i:02d}" for i in range(15)]


# A forecast built directly from the synthetic members (CF, so x capacity for MW), standing in for a real model
@pytest.fixture
def ensemble_forecast(test_df):
    samples = test_df[MEMBERS].to_numpy() * test_df["capacity_mw"].to_numpy()[:, None]
    return Forecast(keys=test_df, point=samples.mean(axis=1), quantiles=quantiles_from_samples(samples),
                    samples=samples)


def test_score_table_layout(ensemble_forecast, test_df):
    table = score(ensemble_forecast, test_df["power"].to_numpy())
    assert list(table.index) == sorted(test_df["zone"].unique()) + ["all"]
    assert list(table.columns[:6]) == ["n", "rmse", "rmse_daily", "mae", "bias", "crps_q"]
    for col in ["pinball_0.1", "pinball_0.5", "pinball_0.9", "cov_50", "cov_80", "cov_90", "cov_95", "crps_sample"]:
        assert col in table.columns
    assert "crps_exact" not in table.columns
    assert table.loc["all", "n"] == len(test_df)


def test_rmse_is_pooled_within_group(ensemble_forecast, test_df):
    table = score(ensemble_forecast, test_df["power"].to_numpy())
    assert table.loc["all", "rmse"] == pytest.approx(rmse(ensemble_forecast.point, test_df["power"]))


def test_group_by_two_keys(ensemble_forecast, test_df):
    table = score(ensemble_forecast, test_df["power"].to_numpy(), by=("zone", "issue_time"))
    assert table.index.names == ["zone", "issue_time"]
    assert ("all", "all") in table.index


def test_optional_columns(test_df):
    n = len(test_df)
    plain = Forecast(keys=test_df, point=np.zeros(n), quantiles=np.zeros((n, 99)))
    with_fn = Forecast(keys=test_df, point=np.zeros(n), quantiles=np.zeros((n, 99)), crps_fn=lambda y: np.abs(y))
    y = test_df["power"].to_numpy()
    assert "crps_sample" not in score_rows(plain, y).columns
    rows = score_rows(with_fn, y)
    np.testing.assert_allclose(rows["crps_exact"], np.abs(y))


def test_missing_observations_are_dropped(ensemble_forecast, test_df):
    y = test_df["power"].to_numpy().copy()
    y[:10] = np.nan
    assert len(score_rows(ensemble_forecast, y)) == len(test_df) - 10


def test_wrong_length(ensemble_forecast):
    with pytest.raises(ValueError, match="y has shape"):
        score_rows(ensemble_forecast, np.zeros(3))


# Bias is the mean of forecast minus observation; rmse_daily is the RMSE per issue, averaged over issues
def test_bias_and_daily_rmse(ensemble_forecast, test_df):
    y = test_df["power"].to_numpy()
    table = score(ensemble_forecast, y)
    err = ensemble_forecast.point - y
    assert table.loc["all", "bias"] == pytest.approx(np.nanmean(err))
    d = test_df.assign(sq=err ** 2).dropna(subset=["sq"])
    assert table.loc["all", "rmse_daily"] == pytest.approx(np.sqrt(d.groupby("issue_time")["sq"].mean()).mean())
    one = d.loc[d["zone"] == "ELSPOT NO2"]
    assert table.loc["ELSPOT NO2", "rmse_daily"] == pytest.approx(np.sqrt(one.groupby("issue_time")["sq"].mean()).mean())


# Interval widths grow with the nominal width; a Brier column appears per event
def test_widths_and_events(ensemble_forecast, test_df):
    y = test_df["power"].to_numpy()
    threshold = 0.6 * test_df["capacity_mw"].to_numpy()
    table = score(ensemble_forecast, y, events={"high": threshold})
    w = table.loc["all", ["width_50", "width_80", "width_90", "width_95"]].to_numpy(dtype=float)
    assert (np.diff(w) > 0).all() and (w > 0).all()
    assert 0 <= table.loc["all", "bs_high"] <= 1
