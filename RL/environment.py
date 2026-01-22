import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import sys
import copy
import pandas as pd
class PairTradingRL1(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    train_length=2747520
    window_length=50
    
    max_step=1000
    def __init__(self,render_mode=None):
        super(PairTradingRL1, self).__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
        "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "spread": spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
        "zone": spaces.Discrete(5)
        })
        self.action_space = spaces.Discrete(3) 
        self.episode_length = 1000
        self.step_passed=0
        self.current_index=0
        self.df=pd.read_csv('data/spread_rw900_train.csv')
        self.position=0
        self.opend_at=0
              
    def reset(self,options=None, seed=None):
        self.step_passed=0
        self.current_index=0
        self.position=0
        self.opend_at=0
        return {'position':0,'spread':0,'zone':0}, {}


    def step(self, action):
        reward=0
        done=False
        return {'position':0,'spread':0,'zone':0}, reward, done, False, {}

    def render(self):
        print("Rendering the environment...")
        self.renderer.draw(self.grid, self.current_open)
        