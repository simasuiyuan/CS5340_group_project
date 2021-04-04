from Model.ModelInterface import ModelInterface
import pandas as pd
import pmdarima as pmd


class ArmaModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model = pmd.arima.AutoARIMA(d=0)
        self.training_data = None

    def fit(self, training_data: pd.DataFrame, **kwargs):
        self.model.fit(training_data)
        self.training_data = training_data

    def project(self, projection_data: pd.DataFrame, **kwargs):
        horizon = len(projection_data.index)
        return self.model.fit_predict(self.training_data, n_periods=horizon)

    def summary(self):
        return self.model.summary()
