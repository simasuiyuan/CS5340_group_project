from Model.ModelInterface import  ModelInterface
from TM_members_study.Esmond import financial_data as fd
from Model.ArmaModel import ArmaModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def train_model(model: ModelInterface, data: pd.DataFrame):
    model.fit(data)


def test_model(model: ModelInterface, data: pd.DataFrame):
    data_list = data.values.flatten().tolist()
    projection = model.project(data)
    return r2_score(data_list, projection)


if __name__ == "__main__":
    GME_TRAIN = fd.get_financial_time_series("GME", "2020-01-01", "2021-01-01", ["Close"]).pct_change().dropna()
    GME_TEST = fd.get_financial_time_series("GME", "2021-01-01", "2021-01-07", ["Close"]).pct_change().dropna()
    arma_model = ArmaModel()
    train_model(arma_model, GME_TRAIN)
    print(test_model(arma_model, GME_TEST))
