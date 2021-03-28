import yfinance as yf
from typing import List
from pandas_datareader import data as pdr
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Markdown, display
import plotly.express as px
from scipy.cluster import hierarchy as hac
from sklearn.mixture import GaussianMixture as GMM
import plotly.graph_objects as go
import bnlearn
import networkx as nx

def get_financial_time_series(symbol: str, start_date: str, end_date: str,
                              type: List[str] = ('Date','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume','LIBOR')):
    return pdr.get_data_yahoo(symbol, start=start_date, end=end_date)[type]

def plot_go(dataframe):
    Y = dataframe['rt'].values.reshape(-1,1)

    L= np.arange(1,100,1)
    clfs= [GMM(n,covariance_type='full').fit(Y) for n in L]
    aics= [clf.aic(Y) for clf in clfs]
    bics= [clf.bic(Y) for clf in clfs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=L, y=aics,
                        mode='lines+markers',
                        name='lines+markers'))

    fig.add_trace(go.Scatter(x=L, y=bics,
                        mode='lines+markers',
                        name='lines+markers'))
    fig.show()

def plot_clusters(dataframe):
    L=5
    Y = dataframe.loc[:,'rt'].values
    Z = hac.linkage(np.reshape(Y, (len(Y), 1)), method='ward', metric='euclidean')

    # Plot dendogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hac.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        # truncate_mode='level', p=L
    )
    plt.show()

def plot_clusters_covariance(dataframe):
    L = 5#4
    Y = dataframe['rt'].values.reshape(-1,1)
    GMM_cluster = GMM(L,covariance_type='full').fit(Y)
    df_rt['cls'] = GMM_cluster.predict(Y)
    df = px.data.iris()
    fig = px.scatter(dataframe, x="Date", y="rt", color="cls", hover_data=df_rt.columns.to_list())
    fig.show()

def test(x):
    ls.append(x)
    print(x)
    return x[-1]

if __name__ == "__main__":
    start, end = "2002-01-01", "2019-01-01"
    # print(get_financial_time_series("GME", start, end, ['Open', 'High', 'Low', 'Close']))
    GME_stock = get_financial_time_series("GME", start, end, ['Open', 'High', 'Low', 'Close']).reset_index()
    GME_stock['Date'] = pd.to_datetime(GME_stock['Date'])
    # GME_stock.head()

    Pt = GME_stock.loc[:,'Close']
    Pt_prev = GME_stock.loc[1:,'Close'].reset_index(drop=True)
    rt = 100*(np.log(Pt[:-1]) - np.log(Pt_prev))
    rt = pd.concat([pd.Series([np.nan]), rt]).reset_index(drop=True)
    GME_stock['rt'] = rt
    df_rt = GME_stock.dropna().reset_index(drop=True)
    
    L = 5#4
    Y = df_rt['rt'].values.reshape(-1,1)
    GMM_cluster = GMM(L,covariance_type='full').fit(Y)
    df_rt['cls'] = GMM_cluster.predict(Y)

    result_dict = df_rt.groupby('cls').agg({'rt': [np.min,np.max,np.mean]}).to_dict()
    df_rt.loc[:, 'rt_cls_mean'] = df_rt['cls'].map(result_dict[('rt', 'mean')])
    # print(df_rt)

    ls = []
    windows = 10
    df_rt.loc[:,['rt_cls_mean']].rolling(window=windows,
                                center=False).apply(test,raw=True)

    columns = [f'r_{i}' for i in range(windows)]
    df = pd.DataFrame(np.array(ls), columns=columns)
    print(df)