import requests
import pandas as pd
import datetime
import time
headers = {
    'Content-Type': 'application/json'
}

TOKEN="4864b7dd9219edb7b82e44eb6fe98d0f408cd6e2"
TICKER="btcgbp"

# TOKEN="afc46f741813e2168d1090c70eaf1cd51f7d0939"
# TICKER="btceur"
INTERVAL="1min"
ADD_DAY=3
days=2000
CSV_PATH=f"data/{TICKER}_{INTERVAL}.csv"
log_file_path=f'{TICKER}_{INTERVAL}.txt'
log_file=open(log_file_path, 'w')
try:
    df=pd.read_csv(CSV_PATH)
    last_date=df["date"].iloc[-1]
    complete_end_date=df["date"].iloc[0]
    complete_end_date = datetime.datetime.fromisoformat(complete_end_date).date()-datetime.timedelta(2000)
    last_date = datetime.datetime.fromisoformat(last_date).date()
    print(f"Last date in CSV: {last_date}")
    log_file.write(f"Last date in CSV: {last_date}\n")
    
except Exception as e:
    print(f"Error reading CSV: {e}")
    log_file.write(f"Error reading CSV: {e}\n")
    
    last_date = None
    df=pd.DataFrame()
    
if last_date==None:
    end_date=datetime.date.today()
    complete_end_date=end_date - datetime.timedelta(days=2000)
else:
    end_date=last_date
start_date=end_date - datetime.timedelta(days=ADD_DAY)

f_end_date=end_date.strftime("%Y-%m-%d")
f_start_date=start_date.strftime("%Y-%m-%d")

complete_data=[]
wait=False
request_made=0
while start_date >= complete_end_date:
    
    if wait:
        time.sleep(3600)
        wait=False
    
    url=f"https://api.tiingo.com/tiingo/crypto/prices?tickers={TICKER}&startDate={f_start_date}&endDate={f_end_date}&resampleFreq={INTERVAL}&token={TOKEN}"
    try:
        print(f"-------Fetching data from {f_start_date} to {f_end_date}")
        requestResponse = requests.get(url, headers=headers)
        data=requestResponse.json()[0]["priceData"]
        request_made+=1
        
        data.reverse()
        print(f"fetched data  from {f_start_date} to {f_end_date}")
        log_file.write(f"fetched data from {f_start_date} to {f_end_date}\n")
        
    
        print(f"Current requests: {request_made}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("--------waiting for hour--------")
        log_file.write(f"Error fetching data: {e}\n--------waiting for hour\n")
        wait=True
        continue

    if data==[]:
        print(f"No data found for {f_start_date} to {f_end_date}")
        log_file.write(f"No data found for {f_start_date} to {f_end_date}\n")
        
        continue
    else:
        #df =pd.concat([df,pd.DataFrame(data)],axis=0, ignore_index=True)
        data_df = pd.DataFrame(data)
        print("saving data to csv")
        data_df.to_csv(CSV_PATH,mode='a', index=False,header=False)
        print(f"Data saved to {CSV_PATH}")
        log_file.write(f"saving data to csv\nData saved to {CSV_PATH}\n")
        
    
    f_end_date=f_start_date
    start_date = start_date - datetime.timedelta(days=ADD_DAY)
    f_start_date = start_date.strftime("%Y-%m-%d")

log_file.close()
    
print("data mining completed")

