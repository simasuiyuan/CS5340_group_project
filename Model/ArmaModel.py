from Model.ModelInterface import ModelInterface
import pandas as pd
import pmdarima as pmd


class ArmaModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model = pmd.arima.AutoARIMA(d=0)

    def fit(self, training_data: pd.DataFrame, **kwargs):
        self.model.fit(training_data)

    def project(self, projection_data: pd.DataFrame, **kwargs):
        return self.model.fit_predict(projection_data)

    def summary(self):
        return self.model.summary()
