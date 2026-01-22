import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment_A import PairTradingEnv
import gymnasium as gym
from gymnasium import register
import os
import enum
from stable_baselines3 import A2C
import random

def calculate_summary_metrics(closed_positions, initial_portfolio_value, final_portfolio_value):
    total_trades = len(closed_positions)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "winning_rate": 0,
            "average_win": 0,
            "average_loss": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "final_portfolio_value": initial_portfolio_value,
            "total_return_pct": 0,
            "trade_log": []
        }

    wins = [pos.pnl for pos in closed_positions if pos.pnl > 0]
    losses = [pos.pnl for pos in closed_positions if pos.pnl <= 0]

    winning_trades = len(wins)
    losing_trades = len(losses)

    winning_rate = winning_trades / total_trades if total_trades > 0 else 0
    average_win = np.mean(wins) if wins else 0
    average_loss = np.mean(losses) if losses else 0
    total_profit = sum(wins)
    total_loss = -sum(losses)  # losses are negative
    profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
    avg_drawdown = np.mean([pos.drawdown for pos in closed_positions]) if closed_positions else 0
    max_drawdown = min((pos.drawdown for pos in closed_positions), default=0)

    total_return_pct = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100

    trade_log = [{
        'index':pos.index,
        "entry_price_1": pos.op1,
        "entry_price_2": pos.op2,
        "exit_price_1": pos.cp1,
        "exit_price_2": pos.cp2,
        "side": pos.side,
        "pnl": pos.pnl,
        "drawdown": pos.drawdown
    } for pos in closed_positions]

    summary = {
        "total_trades": total_trades,
        "winning_rate": winning_rate,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "average draworn":avg_drawdown,
        "max_drawdown": max_drawdown,
        "final_portfolio_value": final_portfolio_value,
        "total_return_pct": total_return_pct,
      
    }

    return summary,trade_log
class SL_Mode(enum.Enum):
    signals=1
    momentum=2
    pct_tp_sl=3

class Position():
    def __init__(self,index,op1,op2,side,cp1=None,cp2=None):

        self.op1=op1
        self.op2=op2
        self.drawdown=0
        self.pnl=0
        self.current_pnl=0
        self.side=side
        self.index=index
        if cp1!=None:
            self.close(cp1,cp2)

    def close(self,cp1,cp2):
        self.cp1=cp1
        self.cp2=cp2
        # self.closed=True
       

        self.pnl=self.side*(self.cp1-self.op1 + self.op2-self.cp2)

        return self.pnl
    def update(self,p1,p2):
        self.current_pnl=self.side*(p1-self.op1 + self.op2-p2)

        if self.current_pnl<0 and self.current_pnl<self.drawdown:
            self.drawdown=self.current_pnl
        return self.current_pnl



def test_trading_model_with_metrics_and_plot(env_class, model, test_df: pd.DataFrame, initial_portfolio_value=10000,mode=SL_Mode.signals):
    env = env_class(test_df)
    obs, _ = env.reset()
    closed_positions=[]
    STOP_PCT=0.001
    open_position_side=None
    open_position=None
    position_open=False
    portfolio_value=initial_portfolio_value
    portfolio_values=[]
    position=0
    
    while True:
        try:
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, _, info = env.step(int(action))
        except:
            break
        prev_position = position
        # position=random.randint(-1,1)
        position = env.position
        # pnl=info['portfolio_reward']
        pnl=0
        if prev_position != position:
            if position_open:
                pnl=open_position.close(test_df['p1'].iloc[env.current_step],test_df['p2'].iloc[env.current_step])
                closed_positions.append(open_position)
                position_open=False
                open_position=None
            
            if position !=0:
                open_position=Position(env.current_step,test_df['p1'].iloc[env.current_step],test_df['p2'].iloc[env.current_step],side=position)
                open_position_side=position
                position_open=True
        current_pnl=0
        # if open_position!=None:
            # current_pnl=open_position.update(test_df['p1'].iloc[env.current_step],test_df['p2'].iloc[env.current_step]) if position_open else 0
            # if current_pnl<-STOP_PCT*(open_position.op1 + open_position.op2)/2 :
            #     pnl=open_position.close(test_df['p1'].iloc[env.current_step],test_df['p2'].iloc[env.current_step])
            #     closed_positions.append(open_position)
            #     position_open=False
            #     open_position=None
        print()
        portfolio_values.append(portfolio_value)
    
        portfolio_value+=pnl            
    
    
    
    summary,trade_log=calculate_summary_metrics(closed_positions, initial_portfolio_value, portfolio_value)

    plt.figure(figsize=(12,6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time During Testing')
    plt.legend()
    # for i in range(len(trade_log)):
        # try:
        #     pos=trade_log[i]
        #     if pos['side']==1:
        #         plt.axvline(x=pos['index'], color='g', linestyle='--', alpha=0.5)  # Long entry
        #         plt.axvline(x=trade_log[i+1]['index'], color='r', linestyle='--', alpha=0.5)  # Exit
        #     elif pos['side']==-1:
        #         plt.axvline(x=pos['index'], color='r', linestyle='--', alpha=0.5)  # Short entry
        #         plt.axvline(x=trade_log[i+1]['index'], color='g', linestyle='--', alpha=0.5)  # Exit
        # except:
        #     pass
    plt.grid(True)
    plt.figure(2,figsize=(12,6))
    plt.plot(test_df['z_score'].reset_index(drop=True))
    # for i in range(len(trade_log)):
    #     try:
    #         pos=trade_log[i]
    #         if pos['side']==1:
    #             plt.axvline(x=pos['index'], color='g', linestyle='--', alpha=0.5)  # Long entry
    #             # plt.axvline(x=trade_log[i+1]['index'], color='r', linestyle='--', alpha=0.5)  # Exit
    #         elif pos['side']==-1:
    #             plt.axvline(x=pos['index'], color='r', linestyle='--', alpha=0.5)  # Short entry
    #             # plt.axvline(x=trade_log[i+1]['index'], color='g', linestyle='--', alpha=0.5)  # Exit
    #     except:
    #         pass
    plt.show()

    return summary

df=pd.read_csv('data/final_normal_rw100.csv')[-30000:]
# df=pd.read_csv('final_normal_rw50.csv')
env_id= "PairTradingEnv-v0"
model_name='A2C1-v1'


# model_path = os.path.join(os.getcwd(), "models", model_name)
model_path=model_name

if os.path.exists(model_path+ ".zip"):
        print("Loading existing model")
        model = A2C.load(model_path)
        print("Loaded existing model")
# Usage:
summary = test_trading_model_with_metrics_and_plot(PairTradingEnv, model, df,1)
print("Test Summary:", summary)

