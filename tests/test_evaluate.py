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
    assert list(table.columns[:4]) == ["n", "rmse", "mae", "crps_q"]
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
