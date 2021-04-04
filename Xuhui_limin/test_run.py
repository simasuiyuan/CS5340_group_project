from typing import List

import bnlearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from sklearn.mixture import GaussianMixture as GMM

def get_financial_time_series(symbol: str, start_date: str, end_date: str,
                              type: List[str] = ('Date','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','LIBOR')):
    return pdr.get_data_yahoo(symbol, start=start_date, end=end_date)[type]

def _infer(model, evidences, variables_to_predict):
    if isinstance(model, BayesianModel):
        return VariableElimination(model).map_query(variables_to_predict, evidences)
    else:
        print('Unknown')

def _project(data, metadata, model, mode='simple'):
    res = []
    p = metadata['period']
    data_keys = data.keys()
    data_values = data.to_numpy()[0]
    if mode == 'simple':
        for _ in range(p):
            data_values[0] = None
            evidences ={}
            for k, v in zip(data_keys, data_values):
                if not (v is None):
                    evidences[k] = v
            data_values = np.roll(data_values, 1)
            pred = _infer(model, evidences, [data_keys[0]])[data_keys[0]]
            res.append(metadata['mean'][pred])
    return res

if __name__ == "__main__":
    
    ### Generate input data ###
    SYMBOL = "GME"
    START, END = "2002-01-01", "2019-01-01"
    TYPE = ['Open', 'High', 'Low', 'Close']
    ts = get_financial_time_series(SYMBOL, START, END, TYPE).reset_index()
    ts['Date'] = pd.to_datetime(ts['Date'])

    pt = ts['Close']
    pt_cur = pt[1:].reset_index(drop=True)
    pt_prev = pt[:-1].reset_index(drop=True)
    
    rt = 100*(np.log(pt_cur) - np.log(pt_prev))
    rt = pd.concat([pd.Series([np.nan]), rt]).reset_index(drop=True)
    ts['rt'] = rt
    df_rt = ts.dropna().reset_index(drop=True)
    
    L = 4   # number of cluster
    Y = df_rt['rt'].values.reshape(-1,1)
    GMM_cluster = GMM(L,covariance_type='full').fit(Y)
    df_rt['cls'] = GMM_cluster.predict(Y)
    
    metadata = df_rt.groupby('cls')['rt'].agg([np.min,np.max,np.mean]).to_dict()
    
    N = 7   # period
    df = df_rt[['cls']]
    for i in range(1, N+1):
        df[f'cls_{i}'] = pd.concat([pd.Series([None]*i), df_rt['cls']]).reset_index(drop=True)
    df.dropna(inplace=True)
    print(df)
    
    ### Build & train model ###
    model_hc_bic  = bnlearn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    G = bnlearn.plot(model_hc_bic)
    plt.show()
    model = bnlearn.parameter_learning.fit(model_hc_bic, df)['model']
    
    
    ### Inference & projection ###
    data = df.iloc[[-1]]
    metadata['period'] = N
    result = _project(data, metadata, model)
    print(result)