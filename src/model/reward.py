import numpy as np
import math
from config import LOAD_BALANCE_WEIGHT,COST_WEIGHT

def compute_reward(
    t,
    # Power variables
    p_import, p_export, p_wt, p_pv, p_dg, p_dis_es, p_ch_es,
    # Price parameters
    price_import, price_export, 
    # Cost parameters
    Cop_ma_wt, Cop_ma_pv, rho_fuel, C_startup, C_degrad_es,
    # Efficiency parameters
     eta_dg,
    # Binary states (current and previous)
    u_dg,  prev_u_dg,
    # Load 
    load,
    # Energy states
    soc_es, ees_min, ees_max,

):
    """
    Compute reward based on economic cost and constraint satisfaction
    
    Returns:
        float: reward value (negative cost minus penalties)
    """
    
    # === 1. COST COMPONENTS ===
    
    # Grid costs
    cost_import = price_import * p_import
    revenue_export = price_export * p_export
    
    # Generation costs
    cost_wt = Cop_ma_wt * p_wt
    cost_pv = Cop_ma_pv * p_pv
    fuel_dg = rho_fuel * p_dg / eta_dg if eta_dg > 0 else 0
    
    # Startup costs

    startup_dg = C_startup * max(0, u_dg - prev_u_dg)
    
    # storage degradation
    
    degrad_cost = C_degrad_es * p_dis_es
    


    #definiton of epsilon
    #epsilon=1e-6
    # Total cost calculation

    total_cost = (
        cost_import
        -revenue_export
        +cost_wt
        + cost_pv
        + fuel_dg
        + startup_dg
        + degrad_cost
        )

    #scale_cost=max(math.sqrt(total_gain**2+total_loss**2),epsilon)
    
    # === 2. CONSTRAINT PENALTIES ===
    
    # Power balance violation
    total_supply = p_import + p_wt + p_pv + p_dg + p_dis_es
    total_demand = p_export + load  
    load_balance_violation = abs(total_supply - total_demand)
    #scale_power=max(math.sqrt(total_supply**2+total_demand**2),epsilon)
    
    

    
    
     
    

    

    # === 3.  Normalization ===
    norm_load_balance_violation = load_balance_violation 
    normalized_cost = total_cost 


    
    

    # === 4. FINAL REWARD ===

    insights = {
        'total_true_cost': total_cost,
        'total_cost': normalized_cost,
        'penalty_load': norm_load_balance_violation,

    }
    
#     reward_load = (
#     +5.0 * (1.0 - norm_load_balance_violation)
#     - LOAD_BALANCE_WEIGHT * norm_load_balance_violation
# )
    reward_load = LOAD_BALANCE_WEIGHT * (1.0-norm_load_balance_violation)
    reward_cost=normalized_cost


    





    
    reward = -reward_cost + reward_load
    #reward = np.clip(reward, -10, 10)

    return reward, insights