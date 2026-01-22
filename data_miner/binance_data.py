import requests
import pandas as pd
import time
from datetime import datetime
def get_binance_data(symbol, interval='1m', start_str='1 Jan 2020'):
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'startTime': int(pd.Timestamp(start_str).timestamp() * 1000), 'limit': 10000}
    data = []
    while True:
        r = requests.get(url, params=params).json()
        if not r:
            break
        data += r
        params['startTime'] = r[-1][0] + 1
        time.sleep(0)
        if len(r) < 1000:
            break
        print(r)
        print(datetime.fromtimestamp(params['startTime']/1000))
    df = pd.DataFrame(r, columns=['OpenTime','Open','High','Low','Close','Volume','CloseTime','QuoteAssetVolume','Trades','TakerBuyBase','TakerBuyQuote','Ignore'])
    df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    return df[['OpenTime','Close']].set_index('OpenTime').astype(float)

# Example: BTCUSDT (1-minute)
btc_usd = get_binance_data('GBP/USD', interval='1m', start_str='8 sep 2025')
print(btc_usd)
