from typing import List

import bnlearn as bn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from sklearn.mixture import GaussianMixture as GMM

from bntrain import bn_struct_training
from model import BNModel
from projection import project


def get_financial_time_series(symbol: str, start_date: str, end_date: str,
                              type: List[str] = ('Date','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','LIBOR')):
    return pdr.get_data_yahoo(symbol, start=start_date, end=end_date)[type]

# Generate an additional column of log returns to the input time-series dataframe `ts`
def _to_log_returns(ts):
    ts['Date'] = pd.to_datetime(ts['Date'])
    pt = ts['Close']
    pt_cur = pt[1:].reset_index(drop=True)
    pt_prev = pt[:-1].reset_index(drop=True)
    rt = 100*(np.log(pt_cur) - np.log(pt_prev))
    rt = pd.concat([pd.Series([np.nan]), rt]).reset_index(drop=True)
    ts['rt'] = rt
    return ts.dropna().reset_index(drop=True)

# Discretization using clustering method
def _discretize(df_rt, num_of_cls):
    Y = df_rt['rt'].values.reshape(-1,1)
    GMM_cluster = GMM(num_of_cls, covariance_type='full').fit(Y)
    df_rt['cls'] = GMM_cluster.predict(Y)
    metadata = df_rt.groupby('cls')['rt'].agg([np.min,np.max,np.mean]).to_dict()
    return {'df_rt': df_rt, 'metadata': metadata}

# Reformat the input time series dataframe with log returns `df_rt` to `df` with columns of feature(s),
# e.g. log return, in descending order w.r.t time, e.g, t-1 > t-2, from left (first) to right (last)
def _format(df_rt, period):
    df = df_rt[['cls']]
    for i in range(1, period):
        df[f'cls_{i}'] = pd.concat([pd.Series([None]*i), df_rt['cls']]).reset_index(drop=True)
    df.rename(columns={'cls': 'cls_0'}, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)
    return df

# Composition of preprocessing steps
def preprocess(ts, **kwargs):
    df_rt = _to_log_returns(ts)
    dict_res = _discretize(df_rt, kwargs['num_of_cls'])
    df_rt, metadata = dict_res['df_rt'], dict_res['metadata']
    df = _format(df_rt, kwargs['period'])
    metadata['period'] =  kwargs['period']
    return {'df': df, 'metadata': metadata}

if __name__ == "__main__":
        
    ########## Generate input data ##########
    SYMBOL = "GME"
    START_TRAIN, END_TRAIN = "2002-01-01", "2021-01-01"
    START_TEST, END_TEST = "2021-01-01", "2021-01-07"
    TYPE = ['Open', 'High', 'Low', 'Close']
    PERIOD = 7   # Number of training features e.g., r_{t}, r_{t-1}, r_{t-2}, r_{t-3}, r_{t-4}, r_{t-5}, r_{t-6}
    NUM_OF_CLS = 4   # Number of clusters for data discretization

    train_ts = get_financial_time_series(SYMBOL, START_TRAIN, END_TRAIN, TYPE).reset_index()
    test_ts = get_financial_time_series(SYMBOL, START_TEST, END_TEST, TYPE).reset_index()

    cfg_preprocess = {'num_of_cls': NUM_OF_CLS, 'period': PERIOD}
    dict_train = preprocess(train_ts, **cfg_preprocess)
    

    ########## Testing ##########
    SL_TRAIN = dict_train['df'] 
    PL_TRAIN = SL_TRAIN   # We use the same training data for both structure and parameter learning
    PROJ_TEST = _to_log_returns(test_ts)[['Date', 'rt']]
    METADATA = dict_train['metadata']
    
    # Do not change the keys 'func' and 'params' of all configurations
    cfg_sl = {'func': bn_struct_training, 
              'params': {'sl_data': SL_TRAIN, 'method': 'hillclimbsearch-modified', 'scoring': 'bic'}}
    bayesnet = BNModel(cfg_sl)

    cfg_pl = {'func': bn.parameter_learning.fit, 'params': {'model': bayesnet.struct, 'df': PL_TRAIN,}}
    bayesnet.fit(cfg_pl)
    
    cfg_proj =  {'func': project, 'params': {'data': PL_TRAIN.iloc[[-1]], 'metadata': METADATA, 'model': bayesnet.model['model'], 'mode': 'weighted'}}
    projection = bayesnet.project(cfg_proj)
    print(projection)


    ########## Visualizations ##########
    max_len = min(len(projection), len(PROJ_TEST))
    plt.plot(PROJ_TEST['Date'][:max_len], PROJ_TEST['rt'][:max_len])
    plt.plot(PROJ_TEST['Date'][:max_len], projection[:max_len])
    plt.show()
