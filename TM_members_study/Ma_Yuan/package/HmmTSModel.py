import pandas as pd
import numpy as np
import itertools
import logging
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from functools import wraps
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List
import warnings
warnings.filterwarnings("ignore")


class HmmTimeSeriesModel(object):
    def __init__(self,
                 n_hidden_states: int=4, 
                 n_iter: int=1000,
                 covariance_type: str='full',
                 verbose: bool=False,n_latency=10):
        
        self.hmm = GaussianHMM(n_components=n_hidden_states,
                               covariance_type=covariance_type, 
                               n_iter=n_iter,
                               verbose=verbose)
        self.n_latency = n_latency
        self.features = None
        
    def extract_stock_features(self, data: pd.DataFrame, feature_used: List):
        feature_used = [each_string.lower() for each_string in feature_used]
        data.columns= data.columns.str.strip().str.lower()
        if not all(item in data.columns for item in feature_used):
            raise ValueError(f"lack of features:{[item not in data.columns for item in feature_used]}")
            
        data['frac_change'] = (np.array(data['close']) - np.array(data['open'])) / np.array(data['open'])
        data['frac_high'] = (np.array(data['high']) - np.array(data['open'])) / np.array(data['open'])
        data['frac_low'] = (np.array(data['open']) - np.array(data['low'])) / np.array(data['open'])
        
        forecast_var = np.array(data['open'].shift(1))- np.array(data['open'])
        vol_gap = np.array(data['volume'].pct_change())
        data['forecast_var'] = forecast_var
        data['vol_gap'] = vol_gap
        self.features = ['frac_change','frac_high','frac_low','vol_gap']#,'forecast_var'
        return data.dropna()
        
    def _split_train_test_data(self, data: pd.DataFrame, test_size: float):
        _train_data, _test_data = train_test_split(data, test_size=test_size, shuffle=False)
        self._train_data = _train_data
        self.projection_data = _test_data    
        
    def fit(self, training_data: pd.DataFrame,
            split_data: bool=False, test_size: float=0.2,
            is_printed: bool=False, step: int=None):
        if self.features is None:
            self.features = list(training_data.columns)
            
        if split_data:
            self._split_train_test_data(training_data, test_size)
        else:
            self._train_data = training_data.loc[:, self.features]
        
        self.outcomes_sampling(step=step)
        
        #display(self._train_data)
        self.hmm.fit(np.array(self._train_data))
        if is_printed:
            for i in range(self.hmm.n_components):
                print("{0}th hidden state".format(i))
                print("mean = ", self.hmm.means_[i])
                print("var = ", np.diag(self.hmm.covars_[i]))
                print()
        return self.hmm.means_, self.hmm.covars_, self.hmm.transmat_
    
    def outcomes_sampling(self,step=None):
        if step is None:
            feature_step=[10]*len(self.features)
        else:
            feature_step = step
        temp_single_out = []
        for feature in self.features:
            temp_single_out.append(np.linspace(np.array(self._train_data[feature]).min(),
                                               np.array(self._train_data[feature]).max(),
                                               feature_step[self.features.index(feature)]))
        self._possible_outcomes = list(itertools.product(*temp_single_out))
        
    def _get_probable_outcome(self, index, latency=None):
        if latency == None: latency = self.n_latency
        previous_start_index = max(0, index - latency)
        previous_end_index = max(0, index - 1)
        previous_data = self.projection_data.iloc[previous_start_index : previous_end_index]
        previous_data_features = np.array(previous_data)
        #log probability under the current model (Sum of prodcut)
        log_likihood_outcome = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
#             log_likihood, predicted_sq  = self.hmm.decode(total_data,algorithm='viterbi')
#             log_likihood_outcome.append(log_likihood)
            log_likihood_outcome.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(log_likihood_outcome)]
        return log_likihood_outcome, most_probable_outcome
    
    def project(self, projection_data: pd.DataFrame=None, with_plot=False):
        forcast_res = []
        if projection_data is not None:
            projection_data = projection_data.loc[:, self.features]
            projection_data.columns= projection_data.columns.str.strip().str.lower()
            self.projection_data = projection_data
        for index in tqdm(range(len(self.projection_data))):
            forcast_res.append(self._get_probable_outcome(index, latency=index)[-1])
        self.forcast_res = forcast_res
        return forcast_res
    
    def get_close(self, data, forcast_res, key_word: str="open"):
        predicted_frac_change, _, _, _ = np.array(list(zip(*self.forcast_res)))
        open_price = np.array(data[key_word])
        return open_price * (1 + predicted_frac_change)
    
    
#     def stock_return(self,close_price:np.ndarray):
#         Pt_prev = np.array(close_price)[:-1]
#         Pt = np.array(close_price)[1:]
#         rt = 100*(np.log(Pt) - np.log(Pt_prev))
#         rt = np.insert(rt, 0, np.nan, axis=0)
#         return rt
    def get_return(self, data, forcast_res, key_word: str="close_pred"):
        if key_word != "close":
            data[key_word] = self.get_close(data, forcast_res)
        return data[key_word].pct_change().dropna()