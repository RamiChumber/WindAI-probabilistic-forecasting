import numpy as np
import pandas as pd
import pytest
from windai.evaluation.forecast import Forecast, QUANTILE_LEVELS, quantiles_from_samples


def _keys(n):
    return pd.DataFrame({
        "issue_time": pd.Timestamp("2025-01-01 09:00", tz="UTC"),
        "zone": "ELSPOT NO1",
        "target_time": pd.date_range("2025-01-02 23:00", periods=n, freq="h", tz="UTC"),
    })


def test_valid_forecast():
    q = np.tile(np.linspace(0, 10, 99), (3, 1))
    f = Forecast(keys=_keys(3), point=np.full(3, 5.0), quantiles=q)
    assert len(f) == 3
    assert list(f.keys.columns) == ["issue_time", "zone", "target_time"]
    assert f.samples is None and f.crps_fn is None


def test_point_mass_is_allowed():
    f = Forecast(keys=_keys(2), point=np.zeros(2), quantiles=np.zeros((2, 99)))
    assert np.all(f.quantiles == 0)


def test_extra_key_columns_are_dropped():
    keys = _keys(2).assign(power=[1.0, 2.0])
    f = Forecast(keys=keys, point=np.zeros(2), quantiles=np.zeros((2, 99)))
    assert "power" not in f.keys.columns


def test_wrong_point_shape():
    with pytest.raises(ValueError, match="point has shape"):
        Forecast(keys=_keys(3), point=np.zeros(2), quantiles=np.zeros((3, 99)))


def test_wrong_quantile_shape():
    with pytest.raises(ValueError, match="quantiles has shape"):
        Forecast(keys=_keys(3), point=np.zeros(3), quantiles=np.zeros((3, 50)))


def test_quantile_crossing():
    q = np.tile(np.linspace(0, 10, 99), (2, 1))
    q[1, 40] = 100.0
    with pytest.raises(ValueError, match="quantile crossing"):
        Forecast(keys=_keys(2), point=np.zeros(2), quantiles=q)


def test_non_finite():
    q = np.zeros((2, 99))
    q[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Forecast(keys=_keys(2), point=np.zeros(2), quantiles=q)


def test_wrong_samples_shape():
    with pytest.raises(ValueError, match="samples has shape"):
        Forecast(keys=_keys(3), point=np.zeros(3), quantiles=np.zeros((3, 99)), samples=np.zeros((2, 15)))


def test_quantiles_from_samples():
    samples = np.tile(np.arange(101, dtype=float), (2, 1))   # 0, 1, ..., 100
    q = quantiles_from_samples(samples)
    assert q.shape == (2, 99)
    np.testing.assert_allclose(q[0], QUANTILE_LEVELS * 100)
