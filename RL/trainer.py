import gymnasium as gym
from environment_A import PairTradingEnv
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import re
def get_next_model_path(model_dir="models", prefix="newmodel", extension=".zip"):
    pattern = re.compile(r'-v(\d+)' + re.escape(extension) + r'$')
    max_version = -1

    for filename in os.listdir(model_dir):
        match = pattern.search(filename)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)

    next_version = max_version + 1
    next_model_name = f"{prefix}-v{next_version}"
    next_model_path = os.path.join(model_dir, next_model_name)
    return next_model_path



env_id= "PairTradingEnv-v0"
model_name='A2C_rw100NEW-v1'
df=pd.read_csv('data/final_normal_rw100.csv')[:-30000]
register(env_id,entry_point=PairTradingEnv)
# env= gym.make(env_id,render_mode="human")
# env= gym.make(env_id)
if __name__ == "__main__":
    # env_fns = [lambda: gym.make(env_id) for _ in range(20)]
    # env = gym.vector.AsyncVectorEnv(env_fns)  # Async for true multiprocessing
    # env = make_vec_env(env_id, n_envs=50)
    env= gym.make(env_id,df=df)
   
    model_path = os.path.join(os.getcwd(), "models", model_name)
    print(env.reset())
    if os.path.exists(model_path+ ".zip"):
            print("Loading existing model")
            model = A2C.load(model_path, env=env)
            # model = PPO.load(model_path, env=env)
            model.set_env(env)
            
            print("Loaded existing model")
    else:
        # model = PPO("MultiInputPolicy", env,verbose=1)
        model=A2C('MultiInputPolicy',verbose=1,env=env,
                    learning_rate=1e-4,       # small LR for stability (try 1e-4 to 5e-4)
                    n_steps=128,              # how many steps to run per update
                    gamma=0.99,               # discount factor
                    gae_lambda=0.95,          # bias-variance tradeoff for advantage estimation
                    ent_coef=0.01,            # entropy bonus → encourages exploration
                    vf_coef=0.5,              # value function loss weight
                    max_grad_norm=0.5,        # gradient clipping
                    use_rms_prop=True,        # default in A2C
                    normalize_advantage=True, # helps stability
                    seed=42,
                    )
        # model=PPO('MultiInputPolicy',verbose=1,env=env)
        #model = PPO("MultiInputPolicy",env, verbose=1,tensorboard_log="./ppo_tensorboard/")
        #model = PPO("MlpPolicy",env, verbose=1,tensorboard_log="./ppo_tensorboard/")
        
    obs,info = env.reset()
    
    last_saved_model=0
    
    try:
        while True:
            print("started training")
            model.learn(total_timesteps=100000,reset_num_timesteps=False) 
        
            model.save(model_path) 
            print("saved model")
            print("saving copy")
            next_path = get_next_model_path(prefix="A2C_rw100NEW")
            model.save(next_path) 
    except IndexError:
        print("Training completed - reached end of data")
        model.save(model_path) 
        