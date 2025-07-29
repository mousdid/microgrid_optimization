# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor

# def train_agent(env, total_timesteps=100000):

#     # Create a new environment for monitoring to avoid reusing the same env
#     def make_env():
#         return Monitor(env)
    
#     vec_env = DummyVecEnv([make_env])
    
#     model = PPO("MlpPolicy", vec_env, verbose=1)
#     model.learn(total_timesteps=total_timesteps)
#     model.save(f"ppo_microgrid_model_v1")
#     return model

# agent.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecEnv
from stable_baselines3.common.monitor import Monitor

def train_agent(env, total_timesteps=100_000,call_back=None, model_name="ppo_microgrid_model"):
    """
    env: either a gymnasium.Env or a VecEnv
    """

    # 1) If it's not already a VecEnv, wrap it into one:
    if not isinstance(env, VecEnv):
        # we need a fresh env factory so Monitor() sees a new env each time
        def make_env_fn():
            def _init():
                e = env  # your single MicrogridEnv instance
                return Monitor(e)  
            return _init

        env = DummyVecEnv([make_env_fn()])

    # Now add VecMonitor to track episode returns across all sub‐envs
    env = VecMonitor(env)

    # Create and train PPO
    model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=total_timesteps, callback=call_back)
    model.save(model_name)
    return model
