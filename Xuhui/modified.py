import yfinance as yf # yahfinace api download dataset
from typing import List
from pandas_datareader import data as pdr
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import plotly.graph_objects as go
import bnlearn
import networkx as nx

def get_financial_time_series(symbol: str, start_date: str, end_date: str,
                              type: List[str] = ('Date','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','LIBOR')):
    return pdr.get_data_yahoo(symbol, start=start_date, end=end_date)[type]

def data_preparing():
    start, end = "2002-01-01", "2019-01-01"
    GME_stock = get_financial_time_series("GME", start, end, ['Open', 'High', 'Low', 'Close']).reset_index()
    GME_stock['Date'] = pd.to_datetime(GME_stock['Date'])

    Pt = GME_stock.loc[:,'Close']
    Pt_prev = GME_stock.loc[1:,'Close'].reset_index(drop=True)
    rt = 100*(np.log(Pt[:-1]) - np.log(Pt_prev))
    rt = pd.concat([pd.Series([np.nan]), rt]).reset_index(drop=True)
    GME_stock['rt'] = rt
    df_rt = GME_stock.dropna().reset_index(drop=True)
    return df_rt

def GMM_clustering(L,df_rt):
    def test(x):
        ls.append(x)
        print(x)
        return x[-1]
    L = 5#4
    Y = df_rt['rt'].values.reshape(-1,1)
    GMM_cluster = GMM(L,covariance_type='full').fit(Y)
    df_rt['cls'] = GMM_cluster.predict(Y)

    result_dict = df_rt.groupby('cls').agg({'rt': [np.min,np.max,np.mean]}).to_dict()
    df_rt.loc[:, 'rt_cls_mean'] = df_rt['cls'].map(result_dict[('rt', 'mean')])
    ls = []
    windows = 10
    df_rt.loc[:,['rt_cls_mean']].rolling(window=windows,
                                center=False).apply(test,raw=True)

    columns = [f'r_{i}' for i in range(windows)]
    df = pd.DataFrame(np.array(ls), columns=columns)
    return df

def bn_model_training(df,method,scoring):
    model  = bnlearn.structure_learning.fit(df, methodtype = method, scoretype = scoring)
    model_update = bnlearn.parameter_learning.fit(model, df)
    G = bnlearn.plot(model_update)
    plt.show()
    return model_update




if __name__ == '__main__':
    df = data_preparing()
    df = GMM_clustering(4,df)
    # score choices: "bic", "k2" ,"bdeu"
    # method types: 'chow-liu','cl','hc','ex','cs','exhaustivesearch','hillclimbsearch','constraintsearch'
    model = bn_model_training(df,'hc','bic')
