import numpy as np
from windai.models.base import Model
from windai.evaluation.forecast import Forecast, QUANTILE_LEVELS

# EMOS (Gneiting et al., 2005): a parametric predictive distribution whose location and scale are linear in
# ensemble statistics, e.g. mu = a + b * mean(p), sigma^2 = c + d * var(p), fitted by minimising mean CRPS
# over a training window.
# Decisions to make first (notes/decisions.md):
#   - distribution and support (Set 1, 4.1): power is bounded by 0 and capacity
#   - scale: MW or capacity factor (Set 1, 2.6)
#   - how member 00 enters the predictors (Set 1, 6.4 and your supervisor's note)
#   - training window: rolling length or expanding (Set 1, Part 8)
# Fill Forecast.quantiles from the fitted distribution's quantile function at QUANTILE_LEVELS. If your
# distribution has a closed-form CRPS, also pass crps_fn=lambda y: <CRPS of each row's distribution at y>.
# You need that formula anyway as the training objective.


class EMOS(Model):
    name = "emos"

    def fit(self, train_df):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError
