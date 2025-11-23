import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import MicrogridEnv
from scenarios import generate_mixed_scenario_dataset
from baseline import BaselineController
import matplotlib.pyplot as plt
# from milp import solve_milp    # uncomment if you have a MILP solver function

SEED=19
# Utility to load parameter CSVs into a dict of lists
def load_params(path):
    params = {}
    for fname in os.listdir(path):
        if fname.endswith('.csv'):
            key = fname.replace('.csv','')
            params[key] = pd.read_csv(os.path.join(path, fname)).iloc[:,0].tolist()
    return params






def evaluate_policy(policy: PPO, scenario: dict, horizon: int) -> float:
    """
    Roll out `policy` for exactly one episode of length `horizon` on MicrogridEnv,
    and return the accumulated `total_true_cost` from each step's breakdown.
    
    Args:
        policy:    a loaded PPO model
        scenario:  a dict of parameter time‑series (including your 'scenario' tags)
        horizon:   number of steps to run (should match len(scenario['load']))
    
    Returns:
        total_cost: the sum of breakdown['total_true_cost'] over the episode
    """
    # 1) Construct the env
    env = MicrogridEnv({}, scenario, horizon=horizon)
    
    # 2) Reset; gymnasium reset returns (obs, info)
    obs, _    = env.reset()
    done      = False
    cost = []
    
    # 3) Step until termination
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, breakdown = env.step(action)
        
        # Breakdown contains your insights dict from compute_reward()
        # which now has 'total_true_cost' (raw, un‑normalized)
        cost.append(breakdown.get("total_true_cost", 0.0))

        done = terminated or truncated

    return sum(cost), cost







def evaluate_baseline(params: dict, horizon: int) -> float:
    """
    Roll out your BaselineController for one episode of exactly `horizon` steps,
    summing breakdown['total_cost'] each step.
    """
    env        = MicrogridEnv({}, params, horizon=horizon)
    obs, _     = env.reset()
    done       = False
    cost = []
    baseline   = BaselineController(params)

    while not done:
        t          = env.t                         # current timestep index
        action     = baseline.select_action(t)     # returns shape (7,)
        obs, _, terminated, truncated, breakdown = env.step(action)
        cost.append(breakdown.get("total_true_cost", 0.0))
        done = terminated or truncated

    return sum(cost),cost



def plot_cost_comparison(
    cost_list_1,
    cost_list_2,
    label_1="PPO",
    label_2="Baseline",
    graph_title="Cost Evolution Comparison",
    scenario_csv_path=None
):
    """
    Plots the evolution of one or two cost sequences over time,
    with optional scenario indicators from a CSV file.

    Parameters:
    - cost_list_1: First list of cost values (e.g., PPO).
    - cost_list_2: Optional second list of cost values (e.g., Baseline).
    - label_1: Label for the first cost list.
    - label_2: Label for the second cost list.
    - graph_title: Title of the plot.
    - scenario_csv_path: Optional path to CSV file containing a 'scenario' column.
    """
    hours = list(range(len(cost_list_1)))

    if cost_list_2 is not None and cost_list_1 is not None and len(cost_list_2) != len(cost_list_1):
        raise ValueError("Both cost lists must be of the same length.")

    plt.figure(figsize=(12, 6))

    if cost_list_1 is not None:
        plt.plot(hours, cost_list_1, marker='o', linestyle='-', linewidth=2, label=label_1)
        max_cost_1 =max(cost_list_1)
    else:
        max_cost_1 = 0
    
    if cost_list_2 is not None:
        plt.plot(hours, cost_list_2, marker='o', linestyle='-', linewidth=2, label=label_2)
        max_cost_2 =max(cost_list_2)
    else:
        max_cost_2 = 0
    
    
        
    max_cost = max(max_cost_1, max_cost_2,1)
    # Add scenario markers if CSV is provided
    if scenario_csv_path:
        df = pd.read_csv(scenario_csv_path)
        scenario_col = None
        for col in df.columns:
            if col.lower() == "scenario":
                scenario_col = col
                break

        if scenario_col is None:
            raise ValueError("No 'scenario' column found in the CSV file.")

        scenario_series = df[scenario_col].fillna("normal").astype(str).str.lower()
        unique_scenarios = sorted(set(scenario_series.unique()) - {"normal"})

        for scen in unique_scenarios:
            binary_line = [max_cost if s == scen else 0 for s in scenario_series]
            plt.plot(hours, binary_line, linestyle=':', label=f"Scenario: {scen}", alpha=0.7)

    plt.title(graph_title)
    plt.xlabel("Time (Hours)")
    plt.ylabel("Operational Cost / Scenario Indicator")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def plot_cost_evolution(cost_list, graph_title="Cost Evolution Over Time"):
    """
    Plots the evolution of cost over time.

    Parameters:
    - cost_list: List of cost values over time.
    - graph_title: Title of the plot.
    """
    hours = list(range(len(cost_list)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(hours, cost_list, marker='o', linestyle='-', linewidth=2)
    plt.title(graph_title)
    plt.xlabel("Time (Hours)")
    plt.ylabel("Operational Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def moving_average_list(data, window_size=24):
    """
    Compute the moving average of a list and return a list of the same length,
    replacing the initial (window_size - 1) values with 0.

    Parameters:
    - data: List of floats or ints.
    - window_size: Integer, the number of elements to average.

    Returns:
    - List of moving average values with 0 padding at the start.
    """
    if window_size <= 1 or window_size > len(data):
        return data

    averages = []
    for i in range(len(data)):
        if i < window_size - 1:
            averages.append(0.0)  # Replace None with 0.0
        else:
            window = data[i - window_size + 1:i + 1]
            averages.append(sum(window) / window_size)
    return averages



if __name__ == '__main__':
    # --- 48-hour test (no random scenarios) ---
    #path_48h = 'data/testset/48'
    #params_48h = load_params(path_48h)
    #'ppo_microgrid_model_3m_3env_baseseed19':ppo7 WITHOUT EALY STOP
    #'ppo_3_M_microgrid_model':ppo11 7 7 3 3
    #'ppo_3_M_microgrid_model_cost_importance_0.5':ppo12
    #'ppo_3_M_microgrid_model_cost_importance_0.1':PPO 13
    #best_model_path = "./best_model/best_model"
    
    policy = PPO.load("ppo_0.4_M_microgrid_model_cost_importance_0.1_1env_withcvar_x2_withweights") 
    


    #ppo_cost_48h, ppo_cost_breakdown_48h = evaluate_policy(policy, scenario=params_48h,horizon=48)
    # # milp_cost_48h   = solve_milp(params_48h))

    #print(f"48h  PPO cost: {ppo_cost_48h:.2f}")
    # # print(f"48h  MILP cost: {milp_cost_48h:.2f}")


    # --- 20% of 1-year test with random scenarios ---
    path_1y = 'data/testset/1year'
    
    params_1y = load_params(path_1y)
    
    # use 20% of 8760= ~1,752 timesteps window
    
    scenario_1y = generate_mixed_scenario_dataset(params_1y, seed=SEED,total_hours=1752,number_events=1)

    ppo_cost_1y, ppo_cost_breakdown_1y = evaluate_policy(policy, scenario=scenario_1y,horizon=1752)
    baseline_cost_1y, baseline_cost_breakdown_1y = evaluate_baseline(scenario_1y, horizon=1752)

    print(f"20%1y PPO cost: {ppo_cost_1y:.2f}")
    print(f"20%1y Baseline cost: {baseline_cost_1y:.2f}")
    #plot_cost_evolution(moving_average_list(ppo_cost_breakdown_1y), graph_title="PPO Cost Evolution Over 48 Hours")
    #plot_cost_evolution(moving_average_list(baseline_cost_breakdown_1y), graph_title="Baseline Cost Evolution Over 48 Hours")
    #plot_cost_evolution(baseline_cost_breakdown_1y, graph_title="Baseline Cost Evolution Over 48 Hours")
    plot_cost_comparison(cost_list_1=moving_average_list(ppo_cost_breakdown_1y), cost_list_2=moving_average_list(baseline_cost_breakdown_1y), label_1="PPO", label_2="Baseline", graph_title="20% 1-Year Cost Comparison", scenario_csv_path="../output/scenarios/scenario_seed_19_horizon_1752.csv")
