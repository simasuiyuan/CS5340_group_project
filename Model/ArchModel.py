from Model.ModelInterface import ModelInterface
import pandas as pd
from arch import arch_model
import statsmodels.tsa.api as smt
import numpy as np


def _get_best_model(training_data):
    """
    Uses the Akaike Information Criteria (AIC) to select best order of GARCH model
    :param training_data: Training data
    :return: Params for the order
    """
    best_aic = np.inf
    best_order = None

    pq_rng = range(10)
    for i in pq_rng:
        for j in pq_rng:
            try:
                tmp_mdl = smt.ARIMA(training_data, order=(i, 0, j)).fit(
                    method='mle', trend='nc'
                )
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic, best_order = tmp_aic, (i, 0, j)
            except:
                continue

    return best_order


class ArchModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model = None
        self.p, self.q = None, None
        self.training_data = None

    def fit(self, training_data: pd.DataFrame, **kwargs):
        self.training_data = training_data
        self.p, _, self.q = _get_best_model(training_data)

    def project(self, projection_data: pd.DataFrame, **kwargs):
        horizon = kwargs[0]
        final_data = self.training_data.copy().append(projection_data)
        start_date = projection_data.index.min()
        model = arch_model(final_data, p=self.p, q=self.q)
        result = model.fit(last_obs=start_date)
        forecast = result.forecast(horizon=horizon, start=start_date)
        return forecast.mean

