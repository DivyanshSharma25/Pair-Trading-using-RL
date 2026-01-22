import yfinance as yf
from pytz import timezone
from datetime import datetime,timedelta
import pandas as pd
def get_data(symbol,interval='1m',days=30):
    
    ticker = symbol    
    days_back = days
    step = 7
    
    data_frames = []
    ist = timezone('Asia/Kolkata')
    
    for i in range(0, days_back, step):
        
        end = datetime.now(ist) - timedelta(days=i)
        print(end)
        start = end - timedelta(days=step)
        print(start,end)
        df = yf.download(ticker, interval=interval, start=start, end=end,multi_level_index=False)
        print(df)
        data_frames.append(df[::-1])

    full_data = pd.concat(data_frames)
    full_data=full_data[::-1]
    return full_data

if __name__=='__main__':
    print(get_data('AAPL'))