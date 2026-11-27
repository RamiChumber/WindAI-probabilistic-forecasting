from abc import ABC, abstractmethod

# Interface shared by every model (baselines, EMOS, later the BNN).
# Input to both methods is the model table (windai.data.build.load_dataset): one row per
# (issue_time, zone, target_time). Members p_XX and the target cf are capacity factors; power and capacity_mw
# are MW. Models work in CF and return the Forecast in MW (x capacity_mw, the capacity at issue time; D9).
#   fit(train_df):  estimate parameters from training rows only (windai.data.splits.training_rows). May use the
#                   target columns `power` and `cf`.
#   predict(df):    return a windai.evaluation.forecast.Forecast with one row per row of df, in the same order.
#                   Must NOT read `power` or `cf`: at forecast time the target is unknown. The tests and the
#                   runner call predict() with both dropped to enforce this.


class Model(ABC):
    name = "model"

    @abstractmethod
    def fit(self, train_df):
        ...

    @abstractmethod
    def predict(self, df):
        ...
