"""Fit every model on the training rows of a fit, forecast its evaluation fold and write a score table.

A fit names what is evaluated:
    val     fit on train (purged at 1 Apr 2023), score the validation fold      (default)
    test    refit on train + val (purged at 1 Apr 2024), score the test fold    (only with --test)
    windai  refit on everything up to 31 Mar 2025, score the competition period (only with --windai)
The test fold and the WindAI period are each meant to be scored once, so they need an explicit flag.

Usage (from the repo root, tdt4501 env; build the caches first with `python -m windai.data.build`):
    python experiments/run_baselines.py               # val, real data
    python experiments/run_baselines.py --test        # test, real data
    python experiments/run_baselines.py --synthetic   # fake data, to check the plumbing
"""
import argparse
from pathlib import Path
import pandas as pd
from windai.evaluation.evaluate import score
from windai.models.baselines import Persistence, Climatology, RawEnsemble
from windai.models.EMOS import EMOS
from windai.data.splits import training_rows, evaluation_rows
from windai.utils.synthetic import make_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
MODELS = [Persistence, Climatology, RawEnsemble, EMOS]


def run(df, fit="val", models=MODELS, by=("zone",)):
    train = training_rows(df, fit)
    target = evaluation_rows(df, fit)
    print(f"fit '{fit}': train {train['issue_time'].nunique()} issues, evaluate {target['issue_time'].nunique()} issues")

    tables = []
    for model_cls in models:
        model = model_cls()
        try:
            model.fit(train)
            forecast = model.predict(target.drop(columns=["power", "cf"]))
        except NotImplementedError:
            print(f"{model.name}: skipped, not implemented")
            continue
        y = target["power"].to_numpy()
        # Overall per zone, and per issue regime (12Z / 09Z), since val contains both
        for group in (by, (*by, "issue_hour")):
            table = score(forecast, y, by=group).reset_index()
            table.insert(0, "model", model.name)
            if "issue_hour" not in table.columns:
                table.insert(2, "issue_hour", "all")
            else:
                table = table.loc[table[by[0]] != "all"]      # the overall row is already in the first table
            tables.append(table)
        print(f"{model.name}: done")

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True).set_index(["model", *by, "issue_hour"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use fake data instead of the real dataset")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--test", action="store_true", help="score the test fold (touch once)")
    which.add_argument("--windai", action="store_true", help="score the WindAI period (touch once)")
    args = parser.parse_args()
    fit = "test" if args.test else "windai" if args.windai else "val"

    if args.synthetic:
        df = make_synthetic_dataset()
    else:
        from windai.data.build import load_dataset
        split = "eval" if fit == "windai" else "train"
        df, curve = load_dataset("train", fit=fit)
        if split == "eval":
            eval_df, _ = load_dataset("eval", fit=fit, curve=curve)
            df = pd.concat([df, eval_df], ignore_index=True)
        print(curve.summary().round(3))

    results = run(df, fit=fit)
    if results.empty:
        print("No models implemented yet.")
    else:
        pd.set_option("display.width", 200)
        print(results.round(3))
        RESULTS.mkdir(exist_ok=True)
        tag = ("synthetic" if args.synthetic else "real") + f"_{fit}"
        out = RESULTS / f"scores_{tag}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv"
        results.to_csv(out)
        print(f"written to {out}")
