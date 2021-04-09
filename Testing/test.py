from Model.ModelInterface import ModelInterface
from TM_members_study.Esmond import financial_data as fd
from Model.ArmaModel import ArmaModel
import pandas as pd
import sklearn.metrics as metrics
import numpy as np


def train_model(model: ModelInterface, data: pd.DataFrame):
    model.fit(data)


def test_model(model: ModelInterface, data: pd.DataFrame, score: metrics = metrics.r2_score):
    data_list = data.values.flatten().tolist()
    projection = model.project(data)
    return np.round(score(data_list, projection), 3)


def rolling_window_validation(model: ModelInterface, data: pd.DataFrame,
                              window_size: int = 253, horizon: int = 5, stride: int = 21):

    output_array = []
    current_index, row_count = 0, 0
    while current_index + window_size + horizon < len(data.index):
        start, mid, end = current_index, current_index + window_size, current_index + window_size + horizon
        training_data, test_data = data[start:mid], data[mid:end]["Close"]
        train_model(model, training_data)
        testing_variance, score = test_data.var(), test_model(model, test_data)
        output_array.append([data.index[start], data.index[mid], data.index[end], testing_variance, score])
        current_index += stride
    return pd.DataFrame(data=output_array, columns=["start", "mid", "end", "test_var", "score"])


def log_returns(data: pd.DataFrame):
    return np.log1p(data.pct_change()).dropna()


if __name__ == "__main__":
    price_data = fd.get_financial_time_series("^GSPC", "2018-01-01", "2021-03-01", ["Open", "Close"])
    returns_data = log_returns(price_data)
    print(returns_data)
    print(rolling_window_validation(ArmaModel(), returns_data))
