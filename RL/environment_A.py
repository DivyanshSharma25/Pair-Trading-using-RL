import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PairTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self,df=None):
        super(PairTradingEnv, self).__init__()

        # The dataframe with price and features
        if df is None:
            self.df=pd.read_csv('data/final_train.csv')
        else:
            self.df=df
        self.df = self.df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1

        # Action space: 0=short,1=hold,2=long (Discrete 3)
        self.action_space = spaces.Discrete(3)

        # Observation space: position (float between -1 and 1), zscore (float), zone (Discrete 3)
        self.observation_space = spaces.Dict({
            'position':spaces.Box(low=-1.0, high=1.0,shape=(1,), dtype=np.float32),        # position
            'z_score':spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) ,   # zscore
            'current_returns':spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) ,   # zscore
            # 'zone':spaces.Discrete(3)                                                     # zone
        })

        # Initial state variables
        self.current_step = 0
        self.position = 0.0    # -1 short, 0 no position, 1 long
        self.position_at=0
        self.prev_position=0
        self.done = False

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        # self.current_step = 0
        # self.position = 0.0
        # self.position_at=0
        self.done = False

        return self._get_observation(), {}

    def _get_observation(self):
        # Extract row features for current step
        row = self.df.iloc[self.current_step]
        zscore = np.array([row['z_score']], dtype=np.float32)
        zone = int(row['zone'])
        position = np.array([self.position], dtype=np.float32)
        current_spread = row['p1'] - row['p2']
        spread_return = current_spread - self.position_at
        portfolio_reward = self.prev_position * spread_return
        return {'position':position,
                'z_score':zscore, 
                'current_returns':np.array([portfolio_reward], dtype=np.float32)
                # 'zone':zone
                }

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        self.current_step+=1
        self.prev_position = self.position
        self.position = {0: -1, 1: 0, 2: 1}[action]


        self.current_row = self.df.iloc[self.current_step]
        prev_row = self.df.iloc[self.current_step - 1]
        portfolio_reward=0
        spread_return=0
        current_spread = self.current_row['p1'] - self.current_row['p2']
        spread_return = current_spread - self.position_at
        current_portfolio_reward= self.prev_position * spread_return
        if self.prev_position != self.position:
            portfolio_reward = current_portfolio_reward
            self.position_at=current_spread
            # print(prev_position,self.position,portfolio_reward)
            
    
        # prev_spread = prev_row['p1'] - prev_row['p2']
        # current_spread = current_row['p1'] - current_row['p2']
        # spread_return = current_spread - prev_spread

        # # Portfolio reward: profit/loss from spread movement * previous position
        # portfolio_reward = prev_position * spread_return

        # Action reward: reward if action matches zone logic
        # Zone mapping: 0=open short, 1=neutral/close, 2=open long
        action_reward = 0
        
        # zone = int(prev_row['zone'])
        # if zone == 0 and action == 0:         # short zone and short action
        #     action_reward = 0.1
        # elif zone == 1 and action == 1:       # close/neutral zone and hold action
        #     action_reward = 0.05
        # elif zone == 2 and action == 2:       # long zone and long action
        #     action_reward = 0.1
        # else:
        #     action_reward = -0.05
        
        # Transaction punishment: penalize position change magnitude
        transaction_punishment = abs(self.position - self.prev_position) * 0.1

        # Total reward
        reward = portfolio_reward -0.00005 #+ action_reward/100 #- transaction_punishment  
        # print(reward)
        obs = self._get_observation()
        info = {
            "current_step": self.current_step,
            "position": self.position,
            "spread_return": spread_return,
            "portfolio_reward": portfolio_reward,
            "action_reward": action_reward,
            "transaction_punishment": transaction_punishment
        }
        
        if self.current_step%2000==0:
            self.done=True

        return obs, reward, self.done, False, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Position: {self.position}")

