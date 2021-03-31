import pandas as pd
import numpy as np
from typing import List,Union,Callable
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
                data: pd.DataFrame, 
                features:Union[List,np.ndarray,str],
                test_size: float=0.2,
                n_latency_days=10,
                n_hidden_states: int=4, 
                startprob: np.ndarray=None,
                n_iter: int=1000,
                covariance_type: str='full',
                is_inferring: bool=False,
                step: np.ndarray=None,
                ):

                self._init_logger()
                self.data = data
                self.n_latency_days = n_latency_days
                self.features = list(features),
                self.startprob = startprob,
                self.hmm = GaussianHMM(n_components=n_hidden_states,covariance_type=covariance_type, n_iter=n_iter,verbose=False)
                self._split_train_test_data(test_size)
                self._compute_all_possible_outcomes(step=step)

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

       def _split_train_test_data(self, test_size: float):
        _train_data, _test_data = train_test_split(
            self.data, test_size=test_size, shuffle=False)
 
        self._train_data = _train_data
        self._test_data = _test_data

    def _extract_features(
        self,
        data: pd.DataFrame, 
        features: Union[List,np.ndarray,str]
        )->np.ndarray:
        return np.array(data[self.features[0]])

    def fit(self,is_printed=False):
        feature_vector = self._extract_features(self._train_data,self.features)
        if not self.startprob:
            self.hmm.startprob_ = self.startprob
        self.hmm.fit(feature_vector)
        if is_printed:
            for i in range(self.hmm.n_components):
                print("{0}th hidden state".format(i))
                print("mean = ", self.hmm.means_[i])
                print("var = ", np.diag(self.hmm.covars_[i]))
                print()
        return self.hmm.means_, self.hmm.covars_, self.hmm.transmat_
    
    def _compute_all_possible_outcomes(self,step=None):
        if step is None:
            feature_step=[10]*len(self.features[0])
        else:
            feature_step = step
        temp_single_out = []
        for feature in self.features[0]:
            temp_single_out.append(np.linspace(np.array(self.data[feature]).min(),
                                               np.array(self.data[feature]).max(),
                                               feature_step[self.features[0].index(feature)]))
        self._possible_outcomes = list(itertools.product(*temp_single_out))
    
    
    def _get_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data.iloc[previous_data_start_index : previous_data_end_index]
        previous_data_features = self._extract_features(previous_data,self.features)
        
        #log probability under the current model (MAP)
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
            
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]
        return outcome_score, most_probable_outcome
    
    def predict_close_price(self, day_index):
        open_price = self._test_data.iloc[day_index]['Open']
        predicted_frac_change, _, _ = self._get_probable_outcome(day_index)[-1]
        return open_price * (1 + predicted_frac_change)
    
    def predict_states_for_days(self, days, with_plot=False, is_state=True):
        predicted_state = []
        for day_index in tqdm(range(days)):
            #means, covars, transmat = self.fit()
            if is_state:
                predicted_state.append(self._get_probable_outcome(day_index)[-1])
            else:
                predicted_state.append(self.predict_close_price(day_index))
            
        if with_plot:
            test_data = self._test_data[0: days]

            if is_state:
                actual_state = test_data['state']
                cm=confusion_matrix(actual_state,predicted_state)
                df_cm = pd.DataFrame(cm, index = [i for i in actual_state.unique()],
                                     columns = [i for i in actual_state.unique()])
                plt.figure(figsize = (10,7))
                sns.heatmap(df_cm, annot=True)
            else:
                actual_state = test_data['Close']
            
        return predicted_state, actual_state, test_data