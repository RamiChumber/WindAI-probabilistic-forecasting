from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import least_squares
from windai.paths import get_data_root
from windai.data.dataset_loader import load_power, MEPS_NOWCAST, STORED_TO_CANONICAL_NAMES, VALID_ZONES
from windai.data.parks import parks_table, capacity_matrix
from windai.data.timing import TZ

# First stage: weather -> power (D7, D11).
# One power curve per zone, applied per park, aggregated with capacity weights:
#   park CF(t) = w_prev f(ws(t-1)) + w_0 f(ws(t)) + w_next f(ws(t+1)),   f(ws) = a / (1 + exp(-(ws - c) / s)) + d
#   zone CF(t) = sum_p cap_p park CF_p(t) / sum_p cap_p      over parks with weather data in operation
# Park power isn't observed, so the curve is fitted through the zone total, on Nowcast (observed weather).
# The weights (w >= 0, sum 1) are a softmax of (l_prev, 0, l_next); fitting them covers both power timestamp
# conventions, and the fitted values are a second test of C9.

PARAMS = ["a", "c", "s", "d", "l_prev", "l_next"]
X0 = np.array([0.9, 8.0, 1.5, 0.0, 0.0, 0.0])
LOWER = np.array([0.0, 0.0, 0.1, 0.0, -10.0, -10.0])
UPPER = np.array([1.2, 25.0, 10.0, 0.3, 10.0, 10.0])


def logistic(ws, a, c, s, d):
    return a / (1 + np.exp(-(ws - c) / s)) + d


def alignment_weights(l_prev, l_next):
    z = np.array([l_prev, 0.0, l_next])
    e = np.exp(z - z.max())
    return e / e.sum()


def park_cf(theta, ws_prev, ws, ws_next):
    a, c, s, d, l_prev, l_next = theta
    w = alignment_weights(l_prev, l_next)
    return (w[0] * logistic(ws_prev, a, c, s, d) + w[1] * logistic(ws, a, c, s, d)
            + w[2] * logistic(ws_next, a, c, s, d))


@dataclass
class PowerCurve:
    params: dict                                   # zone -> array of PARAMS
    until: pd.Timestamp                            # fitted on power labels < until (UTC)
    diagnostics: dict = field(default_factory=dict)  # zone -> {"rmse_cf", "n_hours"}

    def weights(self, zone):
        return alignment_weights(*self.params[zone][4:])

    def summary(self):
        rows = []
        for zone, theta in self.params.items():
            w = self.weights(zone)
            rows.append({"zone": zone, **dict(zip(PARAMS[:4], theta[:4])), "w_prev": w[0], "w_0": w[1],
                         "w_next": w[2], **self.diagnostics.get(zone, {})})
        return pd.DataFrame(rows).set_index("zone")


# Nowcast wind speed as a wide matrix: one row per hour (contiguous, UTC; missing hours are NaN), one column
# per park. Read with the park column dictionary-encoded to keep memory low.
def load_nowcast_wind():
    table = pq.read_table(get_data_root() / MEPS_NOWCAST, columns=["windpark", "wind_speed_10m", "time"],
                          read_dictionary=["windpark"])
    df = table.to_pandas(ignore_metadata=True)
    df["windpark"] = df["windpark"].cat.rename_categories(lambda n: STORED_TO_CANONICAL_NAMES.get(n, n))
    wide = df.pivot(index="time", columns="windpark", values="wind_speed_10m").astype("float32")
    wide.columns = [str(c) for c in wide.columns]
    wide.index = pd.DatetimeIndex(wide.index).tz_localize("UTC")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(full)


# Fits one curve per zone on Nowcast and zone CF for the power labels before `until` (a local date, the
# boundary of the fit, D2/D7). Capacity here is the capacity in operation at hour t: this is a fit on history,
# so the issue-time rule (D1) doesn't apply. Hours with missing power or nowcast are dropped (D10).
def fit_power_curve(until, zones=VALID_ZONES, wind=None, power=None, parks=None):
    until = pd.Timestamp(until, tz=TZ).tz_convert("UTC")
    wind = load_nowcast_wind() if wind is None else wind
    power = load_power() if power is None else power
    parks = parks_table() if parks is None else parks
    wind = wind.loc[wind.index < until]
    times = wind.index
    ws_prev, ws_next = wind.shift(1), wind.shift(-1)

    params, diagnostics = {}, {}
    for zone in zones:
        zone_parks = parks.loc[parks["zone"] == zone]
        covered = zone_parks.loc[zone_parks["has_weather"]]
        cap = capacity_matrix(covered, times)                         # (hours, covered parks)
        cap_all = capacity_matrix(zone_parks, times).sum(axis=1)
        cf = power[zone].reindex(times).to_numpy() / np.where(cap_all > 0, cap_all, np.nan)
        cols = covered["park"].tolist()
        w_p, w_0, w_n = (x[cols].to_numpy(dtype=float) for x in (ws_prev, wind, ws_next))
        ok = (np.isfinite(cf) & (cap.sum(axis=1) > 0) & np.isfinite(w_p).all(axis=1)
              & np.isfinite(w_0).all(axis=1) & np.isfinite(w_n).all(axis=1))
        cap, cf, w_p, w_0, w_n = cap[ok], cf[ok], w_p[ok], w_0[ok], w_n[ok]
        weights = cap / cap.sum(axis=1, keepdims=True)

        def residual(theta):
            return (weights * park_cf(theta, w_p, w_0, w_n)).sum(axis=1) - cf

        fit = least_squares(residual, X0, bounds=(LOWER, UPPER))
        params[zone] = fit.x
        diagnostics[zone] = {"rmse_cf": float(np.sqrt(np.mean(fit.fun ** 2))), "n_hours": int(ok.sum())}
    return PowerCurve(params=params, until=until, diagnostics=diagnostics)


# Capacity-weighted mean over a zone's covered parks for every (issue, lt, member):
#   values (issues, parks, lts, members), capw (issues, parks) -> (issues, lts, members); NaN where capw sums to 0
def zone_weighted(values, capw):
    total = capw.sum(axis=1)
    out = np.einsum("ip,iplm->ilm", capw.astype(np.float32), values, optimize=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / np.where(total > 0, total, np.nan)[:, None, None]


# Adds p_00..p_14 (zone CF per member, D5) to a base table, from the wind cube and a fitted curve. Parks are
# weighted by their capacity in operation at issue time (D1/D5). Computed zone by zone to bound memory.
def add_power_ensemble(table, cube, curve, parks=None):
    parks = parks_table() if parks is None else parks
    table = table.copy()
    ws, cube_parks, lts = cube["ws"], list(cube["parks"]), cube["lts"]
    issue_index = pd.Index(cube["issues"])
    members = ws.shape[3]
    p_cols = [f"p_{m:02d}" for m in range(members)]
    out = np.full((len(table), members), np.nan)

    for zone in table["zone"].unique():
        covered = parks.loc[(parks["zone"] == zone) & parks["has_weather"] & parks["park"].isin(cube_parks)]
        pidx = [cube_parks.index(p) for p in covered["park"]]
        capw = capacity_matrix(covered, cube["issues"])                  # (issues, parks) at issue time
        a, c, s, d, l_prev, l_next = curve.params[zone]
        w = curve.weights(zone).astype(np.float32)
        f = logistic(ws[:, pidx], np.float32(a), np.float32(c), np.float32(s), np.float32(d))
        aligned = w[0] * f[:, :, :-2] + w[1] * f[:, :, 1:-1] + w[2] * f[:, :, 2:]   # lts[1:-1]
        del f
        zone_cf = zone_weighted(aligned, capw)
        del aligned

        rows = np.flatnonzero((table["zone"] == zone).to_numpy())
        ii = issue_index.get_indexer(table["issue_time"].iloc[rows])
        li = table["lt"].iloc[rows].to_numpy() - lts[1]
        assert (ii >= 0).all(), "issue missing from the wind cube"
        assert ((li >= 0) & (li < zone_cf.shape[1])).all(), "lead time outside the cube's aligned range"
        out[rows] = zone_cf[ii, li]

    table[p_cols] = out
    return table
