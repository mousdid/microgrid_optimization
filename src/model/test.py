from stable_baselines3 import PPO
from config import SCENARIO_TYPE
import os
import pandas as pd
from env import MicrogridEnv
from scenarios import generate_scenario

# --- Load the trained model ---
model = PPO.load(f"ppo_microgrid_model_{SCENARIO_TYPE}")

# --- Load test parameters ---
TEST_PARAM_DIR = "data/testset/default"


def load_params(path):
    params = {}
    for fname in os.listdir(path):
        if fname.endswith(".csv"):
            key = fname.replace(".csv", "")
            params[key] = pd.read_csv(os.path.join(path, fname)).iloc[:, 0].tolist()
    return params


params = load_params(TEST_PARAM_DIR)
scenario = generate_scenario(params, kind="normal")  # or use window=48 if needed

# --- Create test environment ---
env = MicrogridEnv({}, scenario)

# --- Run the model on the test scenario ---
obs, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

# --- Convert reward to cost (if reward = -cost) ---
total_cost = -total_reward

print(f"Test scenario total reward: {total_reward:.2f}")


