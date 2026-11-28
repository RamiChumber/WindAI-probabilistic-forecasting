# WindAI-probabilistic-forecasting

Probabilistic two-day-ahead forecasts of hourly wind power in the four Norwegian bidding zones (NO1–NO4), built
on the WindAI competition dataset: the MET Norway MEPS 15-member weather ensemble, the MEPS Nowcast, Statnett
production per zone, and park metadata. Part of an M.Sc. specialization project in Computer Science.

**The task.** At issue time on day D, forecast the zone power for every hour of day D+2 (00:00–23:00 Norwegian
local time). Forecasts are probabilistic: every model returns 99 quantiles (and optionally samples, an exact
CRPS and an exact PIT), scored with proper scoring rules.

**The approach.** Weather forecasts are turned into power per ensemble member (a first-stage power curve), then
post-processed into a calibrated predictive distribution (EMOS), and compared with reference forecasts.

---

## Repository layout

```
src/windai/
  paths.py                 data location
  data/                    from raw files to the model table
  models/                  forecasting models
  evaluation/              forecast format and scoring rules
  utils/synthetic.py       fake model table with the real schema, for tests
experiments/run_baselines.py   fits and scores all models
tests/                     pytest suite
environment.yml, pyproject.toml   environment and package definition
```

### `src/windai/paths.py`
- `get_data_root()`: the raw dataset folder. It's the env variable `WINDAI_DATA_ROOT` if set, otherwise
  `data/raw/kushtriv/Desktop/windai_dataset` inside the repo.
- `list_files(subdir)`: the files in a dataset subfolder.

### `src/windai/data/`: building the model table
| File | Contents |
|---|---|
| `dataset_loader.py` | Raw loaders, each with `split="train"` (main dataset) or `"eval"` (the competition period, daily files): `load_power`, `load_met_forecast` (MEPS members, wide format), `load_met_nowcast`, `load_parks`. Timestamps are UTC; nowcast park names are mapped to the forecast's names. |
| `timing.py` | The only place with local time. `target_hours_utc` (the hours of D+2, 23/24/25 on DST days), `target_lead_times`, `info_cutoff` / `last_power_limit` (the information set: 09Z on day D), `FOLDS` and `fold_of` (train / val / test by local issue date). |
| `parks.py` | `parks_table()` (one row per park; the metadata lists some parks under several EIC codes and is deduplicated by name), `capacity_at` / `capacity_matrix` (installed capacity over time, all parks or only parks with weather data). |
| `first_stage.py` | The power curve: a logistic in 10 m wind speed with a storm cut-out, applied per park, combined over t−1, t, t+1 and aggregated with capacity weights. `fit_power_curve(until)` fits it per zone on Nowcast data; `add_power_ensemble` turns member wind into zone capacity factor per member (`p_00…p_14`). |
| `build.py` | `build_wind_cube` (member wind per issue × park × lead time × member), `build_base_table` (one row per issue × zone × target hour, with leakage and row-count asserts), the caches in `data/processed/`, and `load_dataset(split, fit)`: the base table plus `p_00…p_14` from a curve fitted for that fit. Also a CLI (see Usage). |
| `splits.py` | `training_rows(df, fit)` (training rows, purged at the fold boundary), `evaluation_rows(df, fit)`, and `history_rows(df, fit)` for sequential models. |

**The model table** has one row per `(issue_time, zone, target_time)`, with `target_time` in UTC:
- keys and time: `issue_date` (local day D), `issue_hour` (12 or 9), `fold`, `target_hour_utc`,
  `target_hour_local`, `lt`;
- members: `ws_00…ws_14` (zone wind, m/s) and `p_00…p_14` (zone capacity factor);
- persistence: `last_power`, `last_power_time`, `last_cf`;
- capacity at issue time: `capacity_mw`, `capacity_cov_mw`;
- targets: `power` (MW) and `cf` (capacity factor).

**Folds** (by local issue date):

| Fold | Issues |
|---|---|
| train | 2020-02-15 – 2023-03-31 |
| val | 2023-04-01 – 2024-03-31 |
| test | 2024-04-01 – 2025-03-24 |
| windai | the competition period (the eval split) |

A **fit** names what is evaluated:

| Fit | Trains on | Scores |
|---|---|---|
| `"val"` | train | val |
| `"test"` | train + val | test |
| `"windai"` | everything up to 31 Mar 2025 | the competition period |

### `src/windai/models/`
| File | Contents |
|---|---|
| `base.py` | `Model`, the interface: `fit(train_df)`, then `predict(df) -> Forecast`. `predict` must not read `power` or `cf`. |
| `baselines.py` | `Persistence` (last observation plus empirical errors per zone and hour), `Climatology` (training distribution per zone, month and hour, neighbouring hours pooled), `RawEnsemble` (the 15 members as the forecast). |
| `censored_normal.py` | The normal distribution censored to [0, 1]: `cdf`, `quantile`, `mean`, closed-form `crps`, `log_likelihood` and its gradient, randomised `pit`. |
| `EMOS.py` | `EMOS`: a censored normal per zone and UTC hour, μ = a + b·mean(all 15 members), σ² = c² + d²·var(members 01–14), fitted by maximum likelihood (or CRPS). `RollingEMOS(window_days)`: the same model refitted for every issue on the observations known at that time (issues up to D−3; 30/90/365 days or expanding). `SeasonalEMOS`: static EMOS with an annual harmonic in the intercept, the slope and the log scale. |

Models work in capacity factor and return forecasts in MW (× capacity at issue time).

### `src/windai/evaluation/`
| File | Contents |
|---|---|
| `forecast.py` | `Forecast`: `keys`, `point`, 99 `quantiles` (levels 0.01–0.99), optional `samples`, `crps_fn` (exact CRPS) and `pit_fn` (exact PIT). `quantiles_from_samples`. |
| `scoring.py` | `rmse`, `mae`, `pinball`, `crps_quantiles`, `crps_sample` (ensemble CRPS), `pit_from_quantiles` (randomised PIT), `quantile_at` (interpolates between grid levels), `coverage`, `interval_width`, `exceedance_prob` (P(Y > x) from the quantile grid), `brier`. |
| `dependence.py` | Dependence in forecast errors: `normal_scores` (Φ⁻¹ of the PIT), `z_matrix` (issue day × local hour per zone), `acf` and `n_eff` on a calendar-day axis, `long_run_cov` (Newey–West), `numerical_hessian` and `sandwich_ratios` (B/A), `diebold_mariano` (daily-mean loss differences with a Newey–West variance). |
| `evaluate.py` | `score_rows` (per-row scores) and `score(forecast, y, by=...)`: a summary table per group (e.g. zone, or zone × `issue_hour`) with RMSE, daily RMSE (per forecast day, then averaged; over all zones this matches the competition's scale), MAE, bias, CRPS, pinball losses, and coverage and width at 50/80/90/95 %. Optional `events` (a threshold per row) add Brier scores. |

### `experiments/run_baselines.py`
Fits every model on the training rows of a fit, forecasts its evaluation fold, and writes a score table (per zone,
and per zone × issue regime) to `experiments/results/`. It includes Brier scores for high production (CF above the
zone's 90th and 95th training percentile) and their skill against climatology.

### `tests/`
`conftest.py` provides a synthetic model table (Jan–May 2023, split into train and val). Tests that need the real
data are `test_dataset_loader.py`, `test_paths.py`, `test_parks.py` and `test_build.py`; the others run on
synthetic data.

---

## Usage

### 1. Environment
```
conda env create -f environment.yml          # creates the env "tdt4501" and installs the package (pip -e .)
conda activate tdt4501
```
The environment pins OpenBLAS as the linear-algebra library (MKL's threaded mode crashed on Windows).

### 2. Data
Put the WindAI dataset folder `windai_dataset` at `data/raw/kushtriv/Desktop/windai_dataset`, or set
`WINDAI_DATA_ROOT` to its path. The `data/` folder is not tracked.

### 3. Build the caches (once, and after any change in `src/windai/data/`)
```
python -m windai.data.build --split train      # ~1 min, peak memory ~0.7 GB
python -m windai.data.build --split eval       # ~15 s
```
These write `data/processed/wind_cube_{split}.npz` and `data/processed/dataset_{split}.parquet`. The raw
forecast file is 4.7 GB and is never loaded whole.

### 4. Run the tests
```
python -m pytest -q
```

### 5. Score the models
```
python experiments/run_baselines.py               # fit on train, score val
python experiments/run_baselines.py --windows     # also EMOS on 30/90/365-day and expanding windows (~12 min)
python experiments/run_baselines.py --synthetic   # fake data, to check the plumbing
python experiments/run_baselines.py --test        # refit on train + val, score the test fold
python experiments/run_baselines.py --windai      # refit on everything, score the competition period
```
`--test` and `--windai` score the held-out periods. They're meant to be run once, at the end of the study.

### 6. Use the pieces in Python
```python
from windai.data.build import load_dataset
from windai.data.splits import training_rows, evaluation_rows, history_rows
from windai.models.EMOS import EMOS, RollingEMOS
from windai.evaluation.evaluate import score

df, curve = load_dataset("train", fit="val")         # model table + power curve fitted on the train fold
train, val = training_rows(df, "val"), evaluation_rows(df, "val")

model = EMOS().fit(train)
forecast = model.predict(val.drop(columns=["power", "cf"]))
print(score(forecast, val["power"].to_numpy(), by=("zone",)))

rolling = RollingEMOS(90).fit(history_rows(df, "val"))   # refits per issue on data known at that time
forecast_90 = rolling.predict(val.drop(columns=["power", "cf"]))
```

### 7. Add a model
Subclass `windai.models.base.Model`:
- `fit(train_df)` estimates the parameters from training rows;
- `predict(df)` returns a `Forecast` in MW with one row per input row, in the same order, without reading `power`
  or `cf`.

Add it to `ALL_MODELS` in `tests/test_models.py`; the contract test then checks the format, the row order and the
bounds [0, capacity]. A model that refits per issue sets `sequential = True`: it receives `history_rows` and must
use only issues up to D−3 for issue D.
