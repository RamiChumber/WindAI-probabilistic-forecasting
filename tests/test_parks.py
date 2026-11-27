import numpy as np
import pandas as pd
import pytest
from windai.data.parks import parks_table, capacity_at, capacity_matrix


@pytest.fixture(scope="module")
def parks():
    return parks_table()


# Deduplicated by name: 76 metadata rows, 65 parks
def test_one_row_per_park(parks):
    assert len(parks) == 65
    assert parks["park"].is_unique


# Øyfjell1 and Øyfjell2 share an EIC code but are two phases: both are kept
def test_oyfjell_phases_kept(parks):
    assert {"Øyfjell1", "Øyfjell2"} <= set(parks["park"])


def test_parks_without_weather(parks):
    no_weather = set(parks.loc[~parks["has_weather"], "park"])
    assert no_weather == {"Hinnøy", "Håvik", "Orkdal", "Ramslandsvågen"}


# Final installed capacity per zone
@pytest.mark.parametrize("zone, mw", [("ELSPOT NO1", 405.7), ("ELSPOT NO2", 1451.3),
                                      ("ELSPOT NO3", 2120.4), ("ELSPOT NO4", 1160.0)])
def test_final_capacity(parks, zone, mw):
    assert capacity_at(parks, zone, [pd.Timestamp("2030-01-01", tz="UTC")])[0] == pytest.approx(mw)


# Step function: a park counts from its start time on; covered_only skips parks without weather
def test_capacity_steps():
    parks = pd.DataFrame({
        "park": ["a", "b", "c"], "zone": ["Z", "Z", "Z"], "capacity_mw": [10.0, 5.0, 1.0],
        "prod_start": pd.to_datetime(["2020-01-01", "2021-06-01", "2020-01-01"], utc=True),
        "has_weather": [True, True, False],
    })
    times = pd.to_datetime(["2019-12-31", "2020-01-01", "2021-05-31", "2021-06-01"], utc=True)
    np.testing.assert_allclose(capacity_at(parks, "Z", times), [0, 11, 11, 16])
    np.testing.assert_allclose(capacity_at(parks, "Z", times, covered_only=True), [0, 10, 10, 15])
    assert capacity_matrix(parks, times).shape == (4, 3)
