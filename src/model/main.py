import os
import pandas as pd
from env import MicrogridEnv
from agent import train_agent
from scenarios import generate_mixed_scenario_dataset
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from config import HORIZON

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PARAM_DIR = "data/parameters/1year"  # Directory containing parameter CSV files

def load_params(path):
    params = {}
    for fname in os.listdir(path):
        if fname.endswith(".csv"):
            key = fname.replace(".csv", "")
            params[key] = pd.read_csv(os.path.join(path, fname)).iloc[:, 0].tolist()
    return params


def make_env(seed_offset=19):
    def _init(seed=seed_offset):
        params = load_params(PARAM_DIR)
        scenarios = generate_mixed_scenario_dataset(params, seed=seed,number_events=5)
        return Monitor(MicrogridEnv({}, scenarios))
    return _init



if __name__ == "__main__":

    n_envs = 3 # Number of parallel environments
    seed_offset = 19
    

    # Test that the environment follows the gymnasium API
    #for i in range(n_envs):
        #check_env(env_fns[i])

    env_fns = [make_env(seed_offset + i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)



    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=1000 ,verbose=1)
    eval_cb = EvalCallback(
    env,
    callback_on_new_best=stop_cb,
    eval_freq=HORIZON * 50,       # run evaluation every 100k steps
    n_eval_episodes=8,
    best_model_save_path="./best_model/",
    verbose=1
)
    model = train_agent(
    env,
    total_timesteps=400_000,
    call_back=eval_cb,
    model_name="ppo_0.4_M_microgrid_model_cost_importance_0.1_v4"
)


    trackers = env.get_attr("reward_tracker")
    for i, tracker in enumerate(trackers):
        print(f"=== Env #{i} reward breakdown ===")
        tracker.plot()      # uses the class's plt.figure, plt.plot, etc.
        tracker.clear()


   # … assume `env` is your SubprocVecEnv, `model` is your trained PPO …

    # Reset all envs
    obs     = env.reset()                            # obs.shape == (n_envs, obs_dim)
    total_r = np.zeros(env.num_envs, dtype=float)    # accumulator for each env
    dones   = np.zeros(env.num_envs, dtype=bool)

    # 2) Rollout until every env hits done
    while not dones.all():
        # get one action per env
        actions, _ = model.predict(obs, deterministic=True)
        # step returns (obs, rewards, dones, infos)
        obs, rewards, dones, infos = env.step(actions)
        # accumulate each env's per‐step reward
        total_r += rewards

    # 3) Now total_r[i] is the sum of rewards for env i over its episode
    print("Per‑env total reward:", total_r)
    print("Mean total reward  :", total_r.mean())
    print("Sum of all rewards :", total_r.sum())


