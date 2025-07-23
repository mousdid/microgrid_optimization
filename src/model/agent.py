from config import SCENARIO_TYPE
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

def train_agent(env, total_timesteps=100000):
    # Verify environment follows gymnasium API
    check_env(env)
    
    # Create a new environment for monitoring to avoid reusing the same env
    def make_env():
        return Monitor(env)
    
    vec_env = DummyVecEnv([make_env])
    
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(f"ppo_microgrid_model_{SCENARIO_TYPE}")
    return model