import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM


# Generate an additional column of log returns to the input time-series dataframe `ts`  
# *UNUSED*
def _to_log_returns(ts):
    # ts['Date'] = pd.to_datetime(ts['Date'])
    pt = ts['Close']
    pt_cur = pt[1:].reset_index(drop=True)
    pt_prev = pt[:-1].reset_index(drop=True)
    rt = 100*(np.log(pt_cur) - np.log(pt_prev))
    rt = pd.concat([pd.Series([np.nan]), rt]).reset_index(drop=True)
    ts.reset_index(drop=True, inplace=True)
    ts['rt'] = rt
    return ts.dropna().reset_index(drop=True)

# Discretization using clustering method 
# *REVISED column name 'rt' to 'Close'*
def _discretize(df_rt, num_of_cls):
    Y = df_rt['Close'].values.reshape(-1,1)
    GMM_cluster = GMM(num_of_cls, covariance_type='full').fit(Y)
    df_rt.reset_index(drop=True, inplace=True)   # ADDED
    df_rt['cls'] = GMM_cluster.predict(Y)
    metadata = df_rt.groupby('cls')['Close'].agg([np.min,np.max,np.mean]).to_dict()
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
    # df_rt = _to_log_returns(ts)
    df_rt = ts
    dict_res = _discretize(df_rt, kwargs['num_of_cls'])
    df_rt, metadata = dict_res['df_rt'], dict_res['metadata']
    df = _format(df_rt, kwargs['period'])
    metadata['period'] =  kwargs['period']
    return {'df': df, 'metadata': metadata}
