import numpy as np
from windai.models.base import Model
from windai.evaluation.forecast import Forecast, QUANTILE_LEVELS

# EMOS (Gneiting et al., 2005): a parametric predictive distribution whose location and scale are linear in
# ensemble statistics, e.g. mu = a + b * mean(p), sigma^2 = c + d * var(p), fitted by minimising mean CRPS
# over a training window.
# Here: a censored normal on [0, 1] in capacity factor (power is bounded by 0 and capacity), fitted per zone
# and UTC target hour. Location from the mean of all 15 members; scale from the spread of members 01-14, the
# exchangeable ones (member 00, the unperturbed control run, sits in the middle of the ensemble and would
# understate the spread). Still open: the training window (rolling or all training data).
# Fill Forecast.quantiles from the fitted distribution's quantile function at QUANTILE_LEVELS. If your
# distribution has a closed-form CRPS, also pass crps_fn=lambda y: <CRPS of each row's distribution at y>.
# You need that formula anyway as the training objective.


class EMOS(Model):
    name = "emos"

    def fit(self, train_df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError
