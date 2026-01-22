import pandas as pd
import numpy as np 
from statsmodels.tsa.stattools import coint
from data_miner import yfinance_data as yf
def correlation_and_cointegration(stock1, stock2):
    correlation = stock1.corr(stock2)

    coint_t, p_value, _ = coint(stock1, stock2)

    return correlation, p_value

if __name__=='__main__':
    symbol1='HDFCBANK.NS'
    symbol2='ICICIBANK.NS'
    data1=yf.get_data(symbol1,days=25)['Close']
    data2=yf.get_data(symbol2,days=25)['Close']
    print(len(data1),len(data2))
    print(correlation_and_cointegration(data1,data2))