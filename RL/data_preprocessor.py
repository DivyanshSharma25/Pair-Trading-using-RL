import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import delayed,Parallel
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm

start_time = time.time()

def calculate_spread_arr(prices_A, prices_B):
    """
    Calculates regression-based spread series, given prices of two assets.
    
    Parameters:
    - prices_A: np.ndarray, shape (N,), prices of asset A over time
    - prices_B: np.ndarray, shape (N,), prices of asset B over time
    
    Returns:
    - spreads: np.ndarray, shape (N,), spread series for each time step
    - beta_0: float, intercept of regression
    - beta_1: float, slope of regression
    """
    # Reshape prices_B for sklearn (needs 2D input for X)
    prices_B = prices_B.reshape(-1, 1)
    
    # Fit linear regression: p_A = beta_0 + beta_1 * p_B + epsilon_t
    reg = LinearRegression().fit(prices_B, prices_A)
    
    # Calculate predicted p_A from regression
    predicted_A = reg.predict(np.array(prices_B[-1]).reshape(-1,1))
    
    # Calculate spread (residuals)
    spreads = prices_A[-1] - predicted_A  # element-wise subtraction
    
    return spreads[0]

def calculate_spread(df1:pd.DataFrame,df2:pd.DataFrame,start_idx,n,rolling_win:int=900,keep_price=False):
    
    df=pd.DataFrame()
    df.index=df1.index
    df['spread']=0.0
    if start_idx+n<=len(df)-1:
        end_idx=start_idx+n
    else:
        end_idx=len(df)-1
    # for i in range(20,50):
    for i in range(start_idx,end_idx):
        print(i)
        spread=calculate_spread_arr(np.array(df1['close'][i-rolling_win+1:i+1]),np.array(df2['close'][i-rolling_win+1:i+1]))
        df.loc[i,'spread']=spread
        
        
    return df[start_idx:start_idx+n]

def calculate_spread_parallel(df1,df2,n,rw,save=False):
    task=[delayed(calculate_spread)(df1,df2,i,n,rw) for i in range(rw-1,len(df1)-1,n)]
    results=Parallel(n_jobs=-1)(task)
    print(results)
    df=pd.concat(results)
    if save:
        
        df.to_csv('spread_test.csv',index=False)
    return df

def calculate_norma_spread(df1,df2,save=False):
    df=pd.DataFrame()
    df['spread']=df1['close']-df2['close']
    return df

def calculate_spread_ols_point(stock_a, stock_b):
    """
    Calculate spread between two stocks using OLS regression method.
    
    Parameters:
        stock_a (pd.Series): Prices of Stock A (dependent variable).
        stock_b (pd.Series): Prices of Stock B (independent variable).
        
    Returns:
        spread (pd.Series): Spread time series
        hedge_ratio (float): Beta (hedge ratio)
    """
    # Ensure both inputs are aligned
    stock_a = pd.Series(stock_a)
    stock_b = pd.Series(stock_b)
    stock_a, stock_b = stock_a.align(stock_b, join='inner')
    
    # Add constant for regression
    stock_b_const = sm.add_constant(stock_b)
    
    # Run OLS regression (A ~ B)
    model = sm.OLS(stock_a, stock_b_const).fit()
    hedge_ratio = model.params.iloc[1]   # slope = beta
    # print(model.params)
    
    # Calculate spread
    spread = stock_a - hedge_ratio * stock_b
    
    return spread.iloc[-1]
def calculate_spread_ols(df1:pd.DataFrame,df2:pd.DataFrame,start_idx,n,rolling_win:int=900,keep_price=False):
    
    df=pd.DataFrame()
    df.index=df1.index
    df['spread']=0.0
    if start_idx+n<=len(df)-1:
        end_idx=start_idx+n
    else:
        end_idx=len(df)-1
    # for i in range(20,50):
    for i in range(start_idx,end_idx):
        print(i)
        spread=calculate_spread_ols_point(np.array(df1['close'][i-rolling_win+1:i+1]),np.array(df2['close'][i-rolling_win+1:i+1]))
        df.loc[i,'spread']=spread
        
        
    return df[start_idx:start_idx+n]

def calculate_spread_ols_parallel(df1,df2,n,rw,save=False):
    task=[delayed(calculate_spread_ols)(df1,df2,i,n,rw) for i in range(rw-1,len(df1)-1,n)]
    results=Parallel(n_jobs=-1)(task)
    print(results)
    df=pd.concat(results)
    if save:
        df.to_csv('spread_test.csv',index=False)
    return df

def calculate_z_score(df:pd.DataFrame,rw=900):
    scaler=StandardScaler()
    df['z_score']=0.0
    for i in range(rw-1,len(df)-1):
        scaler.fit(np.array(df['spread'][i-rw+1:i+1]).reshape(-1,1))
        # print(scaler.transform(np.array(df['spread'][i]).reshape(-1,1)))
        df.loc[i,'z_score']=scaler.transform(df['spread'][i].reshape(-1,1))[0][0]
        print(i)
        
    return df

def calculate_z_score_part(df:pd.DataFrame,start_idx,n=10000,rw=900):
    
    scaler=StandardScaler()
    df['z_score']=0.0
    end_idx=0
    if start_idx+n<=len(df)-1:
        end_idx=start_idx+n
    else:
        end_idx=len(df)-1
    for i in range(start_idx,end_idx):
        scaler.fit(np.array(df['spread'][i-rw+1:i+1]).reshape(-1,1))
        # print(scaler.transform(np.array(df['spread'][i]).reshape(-1,1)))
        df.loc[i,'z_score']=scaler.transform(df['spread'][i].reshape(-1,1))[0][0]
        print(i)
    return df[start_idx:start_idx+n]
    
def compute_parallel_zscore(df,n,rw,save=False):
    task=[delayed(calculate_z_score_part)(df,i,n,rw) for i in range(rw-1,len(df)-1,n)]
    results=Parallel(n_jobs=-1)(task)
    print(results)
    df=pd.concat(results)
    if save:
        df.to_csv('z_scoree_new.csv',index=False)
    return df
def process_zscore_df(df, zscore_col='z_score', open_thr=1.8, close_thr=0.4):
    df['zone'] = df[zscore_col].apply(assign_zone, args=(open_thr, close_thr))
    return df

def assign_zone(z_score, open_threshold=1.8, close_threshold=0.4):
    # Discrete(3): 0=open/short, 1=neutral, 2=open/long
    if z_score > open_threshold:
        return 2  # open long
    elif z_score < -open_threshold:
        return 0  # open short
    elif abs(z_score) < close_threshold:
        return 1  # neutral/close zone
    else:
        return 1  # neutral
# df_1=pd.read_csv('data\eur_1min_train.csv')         
# df_2=pd.read_csv('data\gbp_1min_train.csv') 

# spread_df=calculate_spread(df_1,df_2,900)
# spread_df.to_csv('spread.csv')

df1=pd.read_csv('AXISBANK__EQ__NSE__NSE__MINUTE.csv')
df2=pd.read_csv('ICICIBANK__EQ__NSE__NSE__MINUTE.csv')

# spread_df=calculate_spread_parallel(df1,df2,10000,900,False)
spread_df=calculate_norma_spread(df1,df2)
# spread_df=calculate_spread_ols_parallel(df1,df2,10000,100,False)
print(spread_df)
# plt.plot(spread_df['spread'])
# plt.show()\
# spread_df.to_csv('spread.csv')

# df=pd.read_csv('spread_new.csv')
spread_df['z_score']=0.0
z_socre_df=compute_parallel_zscore(spread_df,10000,50)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")

z_socre_df['spread']=spread_df['spread']
z_socre_df['p1']=df1['close']
z_socre_df['p2']=df2['close']

z_socre_df.to_csv('final_normal_rw50.csv',index=False)