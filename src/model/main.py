import os
import pandas as pd
from env import MicrogridEnv
from agent import train_agent
from scenarios import generate_scenario
from stable_baselines3.common.env_checker import check_env

PARAM_DIR = "data/parameters/default"

def load_params(path):
    params = {}
    for fname in os.listdir(path):
        if fname.endswith(".csv"):
            key = fname.replace(".csv", "")
            params[key] = pd.read_csv(os.path.join(path, fname)).iloc[:, 0].tolist()
    return params

if __name__ == "__main__":
    params = load_params(PARAM_DIR)
    scenario = generate_scenario(params, kind="normal")  
    env = MicrogridEnv({}, scenario)
    
    # Test that the environment follows the gymnasium API
    check_env(env)
    
    model = train_agent(env,total_timesteps=300000)

    # Evaluate final cost
    obs, _ = env.reset()  # Updated to unpack the tuple from reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)  # Updated to unpack all 5 return values
        done = terminated or truncated
        total_reward += reward

    print(f"Total reward : {total_reward:.2f}")
    print("Training complete. Model saved as 'ppo_microgrid_model'.")
    
    env.reward_tracker.plot()
    env.reward_tracker.clear()