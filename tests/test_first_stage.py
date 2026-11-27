import numpy as np
import pandas as pd
import pytest
from windai.data.first_stage import (PowerCurve, fit_power_curve, add_power_ensemble, alignment_weights,
                                     power_curve, park_cf)

ZONE = "ELSPOT NO1"
TRUE = {"a": 0.95, "c": 8.0, "s": 1.4, "d": 0.02, "c_out": 13.0, "s_out": 0.8, "w": np.array([0.05, 0.5, 0.45])}


def logits(w):
    return np.log(w[0] / w[1]), np.log(w[2] / w[1])


def synthetic_parks():
    return pd.DataFrame({
        "park": ["p1", "p2", "p3"], "zone": ZONE, "capacity_mw": [100.0, 50.0, 30.0],
        "prod_start": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-03-01"], utc=True),
        "has_weather": True,
    })


# Wind for 3 parks and zone power generated from a known curve and known alignment weights
def synthetic_fit_data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    base = np.cumsum(rng.normal(0, 0.8, n))
    base = 2 + 14 * (base - base.min()) / (base.max() - base.min())
    wind = pd.DataFrame({p: np.clip(base + rng.normal(0, 1.5, n), 0, None) for p in ["p1", "p2", "p3"]}, index=times)
    parks = synthetic_parks()
    theta = np.array([TRUE["a"], TRUE["c"], TRUE["s"], TRUE["d"], TRUE["c_out"], TRUE["s_out"], *logits(TRUE["w"])])
    cf_park = park_cf(theta, wind.shift(1).to_numpy(), wind.to_numpy(), wind.shift(-1).to_numpy())
    started = (parks["prod_start"].to_numpy()[None, :] <= times.to_numpy()[:, None]) * parks["capacity_mw"].to_numpy()
    zone_mw = (started * cf_park).sum(axis=1) + rng.normal(0, 1.0, n)
    power = pd.DataFrame({ZONE: zone_mw}, index=times)
    return wind, power, parks


def test_alignment_weights_sum_to_one():
    w = alignment_weights(0.3, -1.2)
    assert w.sum() == pytest.approx(1) and (w > 0).all()


# The fit recovers the curve, the cut-out and the alignment weights it was generated with
def test_fit_recovers_known_curve():
    wind, power, parks = synthetic_fit_data()
    curve = fit_power_curve("2030-01-01", zones=[ZONE], wind=wind, power=power, parks=parks)
    a, c, s, d, c_out, s_out = curve.params[ZONE][:6]
    assert a == pytest.approx(TRUE["a"], abs=0.03)
    assert c == pytest.approx(TRUE["c"], abs=0.2)
    assert s == pytest.approx(TRUE["s"], abs=0.1)
    assert d == pytest.approx(TRUE["d"], abs=0.02)
    assert c_out == pytest.approx(TRUE["c_out"], abs=0.5)
    assert s_out == pytest.approx(TRUE["s_out"], abs=0.3)
    np.testing.assert_allclose(curve.weights(ZONE), TRUE["w"], atol=0.03)
    assert curve.diagnostics[ZONE]["rmse_cf"] < 0.02
    assert curve.diagnostics[ZONE]["cutout_fitted"]


# Without enough high-wind hours the cut-out is held out of range instead of fitted beyond the data
def test_no_cutout_without_high_wind():
    wind, power, parks = synthetic_fit_data()
    low = wind.clip(upper=11.0)                     # zone wind never above 12 m/s
    curve = fit_power_curve("2030-01-01", zones=[ZONE], wind=low, power=power, parks=parks)
    assert not curve.diagnostics[ZONE]["cutout_fitted"]
    assert tuple(curve.params[ZONE][4:6]) == (40.0, 1.0)
    assert curve.diagnostics[ZONE]["n_high_wind"] == 0


# `until` excludes later hours from the fit
def test_fit_respects_until():
    wind, power, parks = synthetic_fit_data()
    curve = fit_power_curve("2020-02-01", zones=[ZONE], wind=wind, power=power, parks=parks)
    assert curve.diagnostics[ZONE]["n_hours"] < 31 * 24


# By hand: one issue, two parks, alignment weight only on t, one park not yet in operation at issue time.
# p must be the capacity-weighted mean of f(ws) over the parks in operation.
def test_add_power_ensemble_by_hand():
    lts = np.arange(33, 63)
    issue = pd.DatetimeIndex([pd.Timestamp("2020-02-01 12:00", tz="UTC")]).as_unit("ns")
    ws = np.zeros((1, 3, len(lts), 2), dtype=np.float32)
    ws[0, 0], ws[0, 1], ws[0, 2] = 6.0, 10.0, 12.0          # p1, p2, p3 (p3 starts 1 Mar: excluded)
    cube = {"ws": ws, "issues": issue, "parks": np.array(["p1", "p2", "p3"], dtype=object), "lts": lts}
    theta = np.array([1.0, 8.0, 1.5, 0.0, 40.0, 1.0, -50.0, -50.0])  # no cut-out in range, weights ~ (0, 1, 0)
    curve = PowerCurve(params={ZONE: theta}, until=pd.Timestamp("2030-01-01", tz="UTC"))
    table = pd.DataFrame({"issue_time": issue.repeat(2), "zone": ZONE, "lt": [40, 50]})
    out = add_power_ensemble(table, cube, curve, parks=synthetic_parks())
    f = lambda x: power_curve(x, 1.0, 8.0, 1.5, 0.0, 40.0, 1.0)
    expected = (100 * f(6.0) + 50 * f(10.0)) / 150
    np.testing.assert_allclose(out[["p_00", "p_01"]].to_numpy(), expected, rtol=1e-5)


# The cut-out factor: no effect well below c_out, half the power at c_out, almost none well above it.
# The rise (c = 5) is saturated well before 12 m/s, so only the cut-out factor varies here.
def test_cutout_shape():
    f = lambda ws: power_curve(ws, 1.0, 5.0, 1.0, 0.0, 15.0, 0.5)
    assert f(12.0) == pytest.approx(1.0, abs=0.01)
    assert f(15.0) == pytest.approx(0.5, abs=0.01)
    assert f(18.0) < 0.01
