import yfinance as yf
from typing import List
from pandas_datareader import data as pdr

yf.pdr_override()


def get_financial_time_series(symbol: str, start_date: str, end_date: str,
                              type: List[str] = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume')):
    return pdr.get_data_yahoo(symbol, start=start_date, end=end_date)[type]


if __name__ == "__main__":
    start, end = "2021-01-01", "2021-03-01"
    print(get_financial_time_series("GME", start, end, ["Close"]))
