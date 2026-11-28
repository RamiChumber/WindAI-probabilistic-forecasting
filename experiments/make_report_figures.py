"""Figures and tables for the project report, written to reports/figures (PDF + PNG) and reports/tables (LaTeX +
CSV). Validation setup only: models are fitted on the train fold (or refitted on history up to D-3) and scored on
the validation fold; B/A is estimated in sample on the train fold. The test fold and the WindAI period are not used.

The main score tables read the newest experiments/results/scores_real_val_*.csv, so run
`python experiments/run_baselines.py --windows` first. Runtime about 15 minutes (three rolling-window fits).

Usage (from the repo root): python experiments/make_report_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from windai.data.build import load_dataset
from windai.data.splits import training_rows, evaluation_rows, history_rows
from windai.data.first_stage import load_nowcast_wind, park_cf
from windai.data.parks import parks_table, capacity_matrix
from windai.data.dataset_loader import load_power
from windai.models.EMOS import (EMOS, SeasonalEMOS, RollingEMOS, ALL_MEMBERS, ensemble_stats,
                                score_contributions)
from windai.models.baselines import RawEnsemble
from windai.evaluation.evaluate import score_rows
from windai.evaluation import dependence as dep
from windai.evaluation.scoring import pit_from_quantiles

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "reports" / "figures", ROOT / "reports" / "tables"
ZONES = ["ELSPOT NO1", "ELSPOT NO2", "ELSPOT NO3", "ELSPOT NO4"]
SHORT = {z: z.split()[-1] for z in ZONES}
VAL_START = pd.Timestamp("2023-04-01")

# One colour per model throughout (validated categorical palette, light mode); zones are always separate panels
COLOR = {"emos_static": "#2a78d6", "emos_seasonal": "#eb6834", "emos_rolling_90": "#1baf7a",
         "raw_ensemble": "#eda100", "emos_rolling_365": "#e87ba4"}
LABEL = {"persistence": "Persistence", "climatology": "Climatology", "raw_ensemble": "Raw ensemble",
         "emos": "EMOS static", "emos_static": "EMOS static", "emos_seasonal": "EMOS seasonal",
         "emos_rolling_30": "EMOS 30 d", "emos_rolling_90": "EMOS 90 d", "emos_rolling_365": "EMOS 365 d",
         "emos_expanding": "EMOS expanding"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.4, "legend.frameon": False, "figure.dpi": 100,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("figure", name)


def top_legend(fig, ax, ncol):
    """One legend above the panels, built from the handles of one axis, so no panel's data is covered."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, handlelength=1.6,
               columnspacing=1.5)


def write_table(df, name, caption, label, align=None):
    """A booktabs LaTeX table from a DataFrame of preformatted strings (index as first column), plus a CSV."""
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")
    esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    align = align or "l" + "r" * df.shape[1]
    header = " & ".join([esc(df.index.name or "")] + [esc(c) for c in df.columns]) + r" \\"
    body = [" & ".join([esc(i)] + [esc(v) for v in row]) + r" \\" for i, row in zip(df.index, df.to_numpy())]
    tex = [r"\begin{table}[htbp]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{align}}}", r"\toprule", header, r"\midrule", *body, r"\bottomrule",
           r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("table", name)


def fmt(x, nd=1):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------------------------------- data

def load():
    df, curve = load_dataset("train", fit="val")
    train, history, val = training_rows(df, "val"), history_rows(df, "val"), evaluation_rows(df, "val")
    return df, curve, train, history, val


def fit_forecasts(train, history, val):
    hidden = val.drop(columns=["power", "cf"])
    models = {"raw_ensemble": (RawEnsemble(), train), "emos_static": (EMOS(), train),
              "emos_seasonal": (SeasonalEMOS(), train), "emos_rolling_90": (RollingEMOS(90), history),
              "emos_rolling_365": (RollingEMOS(365), history)}
    fitted, fc = {}, {}
    for name, (model, data) in models.items():
        fitted[name] = model.fit(data)
        fc[name] = fitted[name].predict(hidden)
        print("fitted", name)
    return fitted, fc


# ---------------------------------------------------------------------------------------------------- tables

def table_folds(df):
    d = df.loc[df["fold"].isin(["train", "val", "test"])]
    issues = d.drop_duplicates("issue_time")
    rows = {}
    for fold, g in issues.groupby("fold", sort=False):
        rows[fold] = {"Issues": f"{len(g):,}", "12Z / 09Z": f"{(g['issue_hour'] == 12).sum()} / {(g['issue_hour'] == 9).sum()}",
                      "First issue": str(g["issue_date"].min()), "Last issue": str(g["issue_date"].max())}
    t = pd.DataFrame(rows).T.loc[["train", "val", "test"]]
    t.index = ["Train", "Validation", "Test"]
    t.index.name = "Fold"
    write_table(t, "tab_folds", "Folds by local issue date: usable forecast issues (10 truncated 12Z runs are "
                "dropped) and their issue times. The test fold is held out for the thesis.", "tab:folds",
                "lrrll")

    cap = d.loc[d["fold"].isin(["train", "val"])].groupby(["zone", "fold"])["capacity_mw"].agg(["min", "max"])
    t = pd.DataFrame({SHORT[z]: {"Start of train (MW)": fmt(cap.loc[(z, "train"), "min"]),
                                 "End of validation (MW)": fmt(cap.loc[(z, "val"), "max"])} for z in ZONES}).T
    t.index.name = "Zone"
    write_table(t, "tab_capacity", "Installed capacity per zone at the start of the train fold and at the end of the "
                "validation fold (all parks, deduplicated by name).", "tab:capacity")


def latest_scores():
    files = sorted((ROOT / "experiments" / "results").glob("scores_real_val_*.csv"))
    if not files:
        raise FileNotFoundError("run experiments/run_baselines.py --windows first")
    print("scores from", files[-1].name)
    return pd.read_csv(files[-1])


def table_scores(scores):
    order = ["persistence", "climatology", "raw_ensemble", "emos", "emos_seasonal", "emos_rolling_30",
             "emos_rolling_90", "emos_rolling_365", "emos_expanding"]
    a = scores.loc[(scores["zone"] == "all") & (scores["issue_hour"] == "all")].set_index("model").loc[order]
    clim = a.loc["climatology", "crps_q"]
    t = pd.DataFrame({
        "CRPS": a["crps_q"].map(fmt), "CRPSS": (1 - a["crps_q"] / clim).map(lambda x: fmt(x, 3)),
        "RMSE": a["rmse"].map(fmt), "MAE": a["mae"].map(fmt), "Bias": a["bias"].map(fmt),
        "Cov. 50": (100 * a["cov_50"]).map(lambda x: fmt(x, 0)), "Cov. 80": (100 * a["cov_80"]).map(lambda x: fmt(x, 0)),
        "Cov. 90": (100 * a["cov_90"]).map(lambda x: fmt(x, 0)), "Cov. 95": (100 * a["cov_95"]).map(lambda x: fmt(x, 0)),
        "Width 95": a["width_95"].map(lambda x: fmt(x, 0)),
        "BSS q90": a["bss_q90"].map(lambda x: fmt(x, 2)), "BSS q95": a["bss_q95"].map(lambda x: fmt(x, 2)),
    })
    t.index = [LABEL[m] for m in order]
    t.index.name = "Model"
    write_table(t, "tab_scores", "Scores on the validation fold, all zones. CRPS (from the 99 quantiles), RMSE, MAE, "
                "bias (forecast minus observed) and the width of the 95\\,\\% interval in MW; coverage of the central "
                "intervals in \\%; CRPSS and Brier skill scores (BSS) against climatology. The Brier events are CF above "
                "the zone's 90th and 95th percentile of the training CF. RMSE pools all zones, hours and days, as in the "
                "competition.", "tab:scores")

    z = scores.loc[(scores["zone"] != "all") & (scores["issue_hour"] == "all")]
    crps = z.pivot(index="model", columns="zone", values="crps_q")
    skill = 1 - crps.div(crps.loc["climatology"], axis=1)
    keep = [m for m in order if m not in ("climatology",)]
    t = skill.loc[keep, ZONES].map(lambda x: fmt(x, 3))
    t.columns = [SHORT[c] for c in t.columns]
    t.index = [LABEL[m] for m in keep]
    t.index.name = "Model"
    write_table(t, "tab_crpss_zone", "CRPSS against climatology per zone, validation fold.", "tab:crpss-zone")


PAIRS = [("emos_static", "raw_ensemble"), ("emos_rolling_90", "emos_static"), ("emos_seasonal", "emos_static"),
         ("emos_rolling_90", "emos_seasonal"), ("emos_rolling_90", "emos_rolling_365")]


def losses(fc, y):
    out, keys = {}, None
    for name, f in fc.items():
        r = score_rows(f, y)
        out[name] = (r["crps_exact"] if "crps_exact" in r else r["crps_sample"]).to_numpy()
        keys = r[["zone", "issue_time", "target_time"]]
    return out, keys


def table_dm(loss, keys):
    dates = keys["issue_time"].dt.tz_convert(None).dt.normalize()
    rows, per_zone_p = {}, []
    for a, b in PAIRS:
        name = f"{LABEL[a]} vs {LABEL[b]}"
        row = {}
        for L in (3, 7, 14):
            res = dep.diebold_mariano(loss[a], loss[b], dates, lag=L)
            if L == 7:
                row["Diff. (%)"] = fmt(100 * res["rel_diff"], 1)
            row[f"p (L={L})"] = f"{res['p']:.3f}" if res["p"] >= 0.001 else "<0.001"
        for z in ZONES:
            m = (keys["zone"] == z).to_numpy()
            res = dep.diebold_mariano(loss[a][m], loss[b][m], dates[m], lag=7)
            per_zone_p.append((name, z, res["p"], res["rel_diff"]))
        rows[name] = row
    adj = dep.benjamini_hochberg([p for *_, p, _ in per_zone_p])
    for (name, z, _, rel), q in zip(per_zone_p, adj):
        rows[name][f"{SHORT[z]} diff. / p_BH"] = f"{100 * rel:+.1f} / " + (f"{q:.3f}" if q >= 0.001 else "<0.001")
    t = pd.DataFrame(rows).T
    t = t[["Diff. (%)", "p (L=3)", "p (L=7)", "p (L=14)"] + [f"{SHORT[z]} diff. / p_BH" for z in ZONES]]
    t.index.name = "Comparison"
    write_table(t, "tab_dm", "Diebold--Mariano tests of equal CRPS on the validation fold. Loss differences are "
                "averaged per issue day; the variance is Newey--West with L lags of days. Diff.: relative CRPS "
                "difference of the first model (negative favours it). Per zone: lag 7, p-values adjusted by "
                "Benjamini--Hochberg over the 20 per-zone tests.", "tab:dm", "l" + "r" * t.shape[1])


def residual_diagnostics(fc, val, y, train, static):
    obs = ~np.isnan(y)
    rows = val.loc[obs].reset_index(drop=True)
    Z = {n: dep.z_matrix(rows, dep.normal_scores(fc[n].pit_fn(y[obs], rng=0)))
         for n in ("emos_rolling_90", "emos_seasonal")}

    # B/A on the train fold for static EMOS, median over the 24 hourly models
    d = train.dropna(subset=["cf"])
    m_all, s2_all = ensemble_stats(d)
    y_all = np.clip(d["cf"].to_numpy(), 0, 1)
    ba = []
    for (zone, hour), idx in d.groupby(["zone", "target_hour_utc"]).indices.items():
        theta = static.params.loc[(zone, hour), ["a", "b", "c", "d"]].to_numpy(dtype=float)
        yy, mm, ss = y_all[idx], m_all[idx], s2_all[idx]
        S = score_contributions(theta, yy, mm, ss)
        A = dep.numerical_hessian(lambda t: -score_contributions(t, yy, mm, ss).mean(axis=0), theta)
        for lag in (0, 7):
            ratio, _ = dep.sandwich_ratios(A, dep.long_run_cov(S, lag, dates=d["issue_date"].to_numpy()[idx]))
            ba.append({"zone": zone, "lag": lag, **dict(zip("abcd", ratio))})
    ba = pd.DataFrame(ba).groupby(["zone", "lag"]).median()

    table = {}
    for z in ZONES:
        M = Z["emos_rolling_90"][z]
        C = M.corr().to_numpy()
        n = M.notna().any(axis=1).sum()
        neff = np.nanmedian([dep.n_eff(dep.acf(M[h]), M[h].notna().sum()) for h in range(24)])
        table[SHORT[z]] = {
            "Corr. 1 h": fmt(np.nanmean(np.diag(C, 1)), 2), "Corr. 6 h": fmt(np.nanmean(np.diag(C, 6)), 2),
            "Corr. 12 h": fmt(np.nanmean(np.diag(C, 12)), 2),
            "ACF 1 d (daily mean)": fmt(dep.acf(M.mean(axis=1))[0], 2),
            "ACF 1 d (per hour)": fmt(np.nanmedian([dep.acf(M[h])[0] for h in range(24)]), 2),
            "n / n_eff": fmt(n / neff, 1),
            "B/A a (indep.)": fmt(ba.loc[(z, 0), "a"], 2), "B/A a (L=7)": fmt(ba.loc[(z, 7), "a"], 2),
            "B/A b (L=7)": fmt(ba.loc[(z, 7), "b"], 2), "B/A c (indep.)": fmt(ba.loc[(z, 0), "c"], 2),
        }
    t = pd.DataFrame(table).T
    t.index.name = "Zone"
    write_table(t, "tab_dependence", "Dependence of the 90-day EMOS errors on the validation fold (normal scores of "
                "the PIT): mean correlation between hours 1, 6 and 12 h apart within the forecast day; lag-1-day "
                "autocorrelation of the daily mean and per hour (median over hours); n / n_eff per hour (median). "
                "B/A: ratio of the sandwich variance to the inverse Fisher information for static EMOS on the train "
                "fold (median over hours), assuming independence or with Newey--West at 7 days.", "tab:dependence")
    return Z


# ---------------------------------------------------------------------------------------------------- figures

def fig_power_curves(curve):
    wind = load_nowcast_wind()
    wind = wind.loc[wind.index < curve.until]
    power, parks = load_power(), parks_table()
    grid = np.linspace(0, 20, 200)
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.9), sharey=True)
    for ax, zone in zip(axes, ZONES):
        zp = parks.loc[parks["zone"] == zone]
        cov = zp.loc[zp["has_weather"]]
        cap = capacity_matrix(cov, wind.index)
        ws = (cap * wind[cov["park"]].to_numpy()).sum(1) / np.where(cap.sum(1) > 0, cap.sum(1), np.nan)
        cf = power[zone].reindex(wind.index).to_numpy() / capacity_matrix(zp, wind.index).sum(1)
        ok = np.isfinite(ws) & np.isfinite(cf)
        ax.scatter(ws[ok][::15], cf[ok][::15], s=0.6, color="#b8b7b1", alpha=0.35, linewidths=0, rasterized=True)
        med = pd.Series(cf[ok]).groupby(pd.cut(ws[ok], np.arange(0, 21, 1)), observed=True).median()
        ax.plot([b.mid for b in med.index], med.to_numpy(), "o", ms=2.5, color=INK, label="Binned median")
        ax.plot(grid, curve.curve(zone, grid), color=COLOR["emos_static"], label="Fitted curve")
        ax.set_title(SHORT[zone]); ax.set_xlim(0, 20); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Capacity factor")
    fig.supxlabel("Capacity-weighted zone wind at 10 m, Nowcast (m/s)", fontsize=8, y=-0.08)
    top_legend(fig, axes[0], 2)
    save(fig, "fig_power_curves")


def fig_pit(fc, val, y):
    obs = ~np.isnan(y)
    zones = val.loc[obs, "zone"].to_numpy()
    names = ["raw_ensemble", "emos_static", "emos_rolling_90"]
    fig, axes = plt.subplots(3, 4, figsize=(6.5, 4.0), sharex=True, sharey="row")
    for i, n in enumerate(names):
        f = fc[n]
        pit = f.pit_fn(y[obs], rng=0) if f.pit_fn is not None else \
            pit_from_quantiles(f.quantiles[obs], f.levels, y[obs], rng=0)
        for j, z in enumerate(ZONES):
            ax = axes[i, j]
            ax.hist(pit[zones == z], bins=10, range=(0, 1), density=True, color=COLOR[n], rwidth=0.9)
            ax.axhline(1, color=INK, lw=0.8, ls="--")
            ax.set_xticks([0, 0.5, 1])
            if i == 0:
                ax.set_title(SHORT[z])
        axes[i, 0].set_ylabel(LABEL[n].replace("EMOS ", "EMOS\n"))
    fig.supxlabel("PIT (bars: density; dashed line: uniform)", fontsize=8, y=-0.02)
    save(fig, "fig_pit")


def fig_coefficients(df, history):
    issue_days = np.sort(df.loc[df["fold"].isin(["train", "val"]), "issue_date"].unique())
    weekly = issue_days[::7][7:]
    rows = df.loc[df["issue_date"].isin(weekly)].drop(columns=["power", "cf"])
    med = {}
    for w in (90, 365):
        m = RollingEMOS(w).fit(history)
        m.predict(rows)
        med[w] = m.params_by_issue.groupby(["issue_date", "zone"])[["b", "c"]].median().reset_index()
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.2), sharex=True, sharey="row")
    for j, z in enumerate(ZONES):
        for w, key in ((90, "emos_rolling_90"), (365, "emos_rolling_365")):
            g = med[w].loc[med[w]["zone"] == z]
            t = pd.to_datetime(g["issue_date"])
            for i, col in enumerate(["b", "c"]):
                axes[i, j].plot(t, g[col], color=COLOR[key], lw=1.0, label=f"{w}-day window")
        for i in range(2):
            axes[i, j].axvline(VAL_START, color=MUTED, lw=0.8, ls="--")
            axes[i, j].xaxis.set_major_locator(mdates.YearLocator())
            axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter("%y"))
        axes[0, j].set_title(SHORT[z])
    axes[0, 0].set_ylabel("Slope b"); axes[1, 0].set_ylabel("Constant SD c (CF)")
    axes[1, 0].set_ylim(bottom=0)
    top_legend(fig, axes[0, 0], 2)
    fig.supxlabel("Issue date (year 20..); dashed line: start of the validation fold", fontsize=8, y=-0.04)
    save(fig, "fig_coefficients")


def fig_monthly_bias(fc, val, y):
    obs = ~np.isnan(y)
    v = val.loc[obs, ["zone", "target_time", "capacity_mw"]].copy()
    v["month"] = pd.to_datetime(v["target_time"]).dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    names = ["emos_static", "emos_seasonal", "emos_rolling_90"]
    for n in names:
        v[n] = (fc[n].point[obs] - y[obs]) / v["capacity_mw"]
    v = v.loc[v["month"] < pd.Timestamp("2024-04-01")]           # April 2024 holds only two target days
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), sharey=True)
    for ax, z in zip(axes, ZONES):
        mon = v.loc[v["zone"] == z].groupby("month")[names].mean()
        for n in names:
            ax.plot(mon.index, mon[n], marker="o", ms=2.5, color=COLOR[n], label=LABEL[n])
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_title(SHORT[z])
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(4, 7, 10, 1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].set_ylabel("Forecast − observed\n(share of capacity)")
    fig.supxlabel("Target month (April 2023 – March 2024)", fontsize=8, y=-0.06)
    top_legend(fig, axes[0], 3)
    save(fig, "fig_monthly_bias")


def fig_drift(df, curve):
    wind, parks = load_nowcast_wind(), parks_table()
    d = df.loc[df["fold"].isin(["train", "val"]) & df["cf"].notna()].reset_index(drop=True)
    d["cf_now"] = np.nan
    d["ws_now"] = np.nan
    for zone in ZONES:
        rows = np.flatnonzero((d["zone"] == zone).to_numpy())
        cov = parks.loc[(parks["zone"] == zone) & parks["has_weather"] & parks["park"].isin(wind.columns)]
        W = wind[cov["park"].tolist()].to_numpy(dtype=float)
        idx = wind.index.get_indexer(d["target_time"].iloc[rows])
        ok = (idx >= 1) & (idx < len(wind) - 1)
        rows, idx = rows[ok], idx[ok]
        capw = capacity_matrix(cov, d["issue_time"].iloc[rows])
        pc = park_cf(curve.params[zone], W[idx - 1], W[idx], W[idx + 1])
        good = np.isfinite(pc).all(axis=1)
        d.loc[rows[good], "cf_now"] = ((capw * pc).sum(1) / capw.sum(1))[good]
        d.loc[rows[good], "ws_now"] = ((capw * W[idx]).sum(1) / capw.sum(1))[good]
    del wind
    d = d.dropna(subset=["cf_now"])
    p_mean = d[ALL_MEMBERS].mean(axis=1)
    d["Total"] = p_mean - d["cf"]                       # forecast minus observed, as everywhere in the report
    d["Power mapping"] = d["cf_now"] - d["cf"]          # power curve on Nowcast wind against observed power
    d["Weather forecast"] = p_mean - d["cf_now"]        # the same curve on the MEPS members against on Nowcast
    d["wind_bias"] = d[[f"ws_{m:02d}" for m in range(15)]].mean(axis=1) - d["ws_now"]
    d["month"] = pd.to_datetime(d["target_time"]).dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    d = d.loc[d["month"] < pd.Timestamp("2024-04-01")]
    mon = d.groupby(["zone", "month"])[["Total", "Power mapping", "Weather forecast", "wind_bias"]].mean()

    parts = {"Total": INK, "Power mapping": "#6250d6", "Weather forecast": "#008300"}
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.4), sharex=True, sharey="row")
    for j, z in enumerate(ZONES):
        g = mon.loc[z]
        for col, c in parts.items():
            axes[0, j].plot(g.index, g[col], color=c, lw=1.4 if col == "Total" else 1.0, label=col)
        axes[1, j].plot(g.index, g["wind_bias"], color=MUTED, lw=1.0)
        for i in range(2):
            axes[i, j].axhline(0, color=INK, lw=0.6)
            axes[i, j].axvline(VAL_START, color=MUTED, lw=0.8, ls="--")
            axes[i, j].xaxis.set_major_locator(mdates.YearLocator())
            axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter("%y"))
        axes[0, j].set_title(SHORT[z])
    axes[0, 0].set_ylabel("Ensemble mean −\nobserved (CF)")
    axes[1, 0].set_ylabel("MEPS − Nowcast\nwind (m/s)")
    top_legend(fig, axes[0, 0], 3)
    fig.supxlabel("Target date (year 20..); dashed line: start of the validation fold", fontsize=8, y=-0.04)
    save(fig, "fig_drift")


def fig_dependence(Z):
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.6), gridspec_kw={"height_ratios": [1.1, 1]})
    for j, z in enumerate(ZONES):
        C = Z["emos_rolling_90"][z].corr().to_numpy()
        im = axes[0, j].imshow(C, vmin=0, vmax=1, cmap="Blues", origin="lower")
        axes[0, j].set_title(SHORT[z]); axes[0, j].grid(False)
        axes[0, j].set_xticks([0, 12, 23]); axes[0, j].set_yticks([0, 12, 23])
        if j:
            axes[0, j].set_yticklabels([])
        ax = axes[1, j]
        n = len(Z["emos_rolling_90"][z])
        ax.axhspan(-2 / np.sqrt(n), 2 / np.sqrt(n), color=GRID, lw=0)
        lags = np.arange(1, dep.MAX_LAG + 1)
        for name in ("emos_rolling_90", "emos_seasonal"):
            ax.plot(lags, dep.acf(Z[name][z].mean(axis=1)), marker="o", ms=2.5, color=COLOR[name], label=LABEL[name])
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_ylim(-0.2, 0.65); ax.set_xticks([1, 7, 14])
        if j:
            ax.set_yticklabels([])
    axes[0, 0].set_ylabel("Local hour of D+2")
    axes[1, 0].set_ylabel("ACF of daily mean z")
    axes[0, 0].set_xlabel("Local hour of D+2")
    axes[1, 0].set_xlabel("Lag (days)")
    top_legend(fig, axes[1, 0], 2)
    fig.colorbar(im, ax=axes[0, :], shrink=0.9, label="Correlation", pad=0.02)
    save(fig, "fig_dependence")


def main():
    style()
    df, curve, train, history, val = load()
    y = val["power"].to_numpy()
    table_folds(df)
    scores = latest_scores()
    table_scores(scores)
    fitted, fc = fit_forecasts(train, history, val)
    loss, keys = losses(fc, y)
    table_dm(loss, keys)
    Z = residual_diagnostics(fc, val, y, train, fitted["emos_static"])
    fig_pit(fc, val, y)
    fig_monthly_bias(fc, val, y)
    fig_dependence(Z)
    fig_power_curves(curve)
    fig_coefficients(df, history)
    fig_drift(df, curve)


if __name__ == "__main__":
    main()
