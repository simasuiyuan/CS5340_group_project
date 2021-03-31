import pandas as pd


class ModelInterface:
    def __init__(self):
        pass

    def fit(self, training_data: pd.DataFrame, **kwargs):
        print("Not Implemented")
        pass

    def project(self, projection_data: pd.DataFrame, **kwargs):
        print("Not Implemented")
        pass

    def summary(self):
        print("Not Implemented")
        pass
