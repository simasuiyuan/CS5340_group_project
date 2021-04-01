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
        
    def _split_train_test_data(self, data: pd.DataFrame, test_size: float):
        _train_data, _test_data = train_test_split(data, test_size=test_size, shuffle=False)
        self._train_data = _train_data
        self.projection_data = _test_data    
        
    def fit(self, training_data: pd.DataFrame,
            split_data: bool=False, test_size: float=0.2,
            is_printed: bool=False, step: int=None):
        self.features = list(training_data.columns)
        if split_data:
            self._split_train_test_data(training_data, test_size)
        else:
            self._train_data = training_data
        
        self.outcomes_sampling(step=step)
        
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
            self.projection_data = projection_data
        for index in tqdm(range(len(self.projection_data))):
            forcast_res.append(self._get_probable_outcome(index, latency=index)[-1])
        return forcast_res