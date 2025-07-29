import numpy as np
from config import LOAD_BALANCE_WEIGHT, HEAT_BALANCE_WEIGHT, BATTERY_BOUNDS_WEIGHT, EV_SOC_BOUNDS_WEIGHT

def compute_reward(
    t,
    # Power variables
    p_import, p_export, p_wt, p_pv, p_chp, p_dg, p_dis_es, p_ch_es, p_ch_ev,
    # Price parameters
    price_import, price_export, price_ev,
    # Cost parameters
    Cop_ma_wt, Cop_ma_pv, rho_gas, rho_fuel, C_startup, C_degrad_es,
    # Efficiency parameters
    eta_chp, eta_dg,
    # Binary states (current and previous)
    u_chp, u_dg, prev_u_chp, prev_u_dg,
    # Load and heat
    load, H_demand, H_chp,
    # Energy states
    soc_es, soc_ev, ees_min, ees_max,
    # EV parameters
    leave_possible, Eev_required
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
    fuel_chp = rho_gas * p_chp / eta_chp if eta_chp > 0 else 0
    fuel_dg = rho_fuel * p_dg / eta_dg if eta_dg > 0 else 0
    
    # Startup costs
    startup_chp = C_startup * max(0, u_chp - prev_u_chp)
    startup_dg = C_startup * max(0, u_dg - prev_u_dg)
    
    # EV charging and storage degradation
    ev_charge_cost = price_ev * p_ch_ev
    degrad_cost = C_degrad_es * p_dis_es
    
    # Total cost calculation
    total_cost = (
        cost_import
        - revenue_export
        - ev_charge_cost
        + cost_wt
        + cost_pv
        + fuel_chp
        + fuel_dg
        + startup_chp
        + startup_dg
        + degrad_cost
    )
    
    # === 2. CONSTRAINT PENALTIES ===
    
    # Power balance violation
    total_supply = p_import + p_wt + p_pv + p_chp + p_dg + p_dis_es
    total_demand = p_export + load + p_ch_es + p_ch_ev
    load_balance_violation = abs(total_supply - total_demand)
    
    
    # Heat balance violation
    heat_deficit = max(0, H_demand - H_chp)
    
    
    # Battery SOC bounds violation
    battery_violation = max(0, ees_min - soc_es) + max(0, soc_es - ees_max)
     
    
    # EV SOC bounds violation (only when leaving is possible)
    penalty_ev = 0
    if leave_possible == 1:
        ev_violation = max(0, Eev_required - soc_ev) + max(0, soc_ev - 70)
        
    else:
        ev_violation = 0
    

    # === 3.  Normalization ===
    norm_load_balance_violation = load_balance_violation / (load + 1e-6)
    norm_heat_deficit = heat_deficit / (H_demand + 1e-6)
    norm_battery_violation = battery_violation / (ees_max - ees_min + 1e-6)
    norm_ev_violation = ev_violation / (Eev_required + 1e-6)
    normalized_cost = total_cost / (load + H_demand + 1e-6)

    # === 4. COEFFICIENTS ===
    penalty_load = -LOAD_BALANCE_WEIGHT * norm_load_balance_violation
    penalty_heat = -HEAT_BALANCE_WEIGHT * norm_heat_deficit
    penalty_batt = -BATTERY_BOUNDS_WEIGHT * norm_battery_violation
    penalty_ev = -EV_SOC_BOUNDS_WEIGHT * norm_ev_violation

    # === 5. FINAL REWARD ===

    insights = {
        'total_true_cost': total_cost,
        'total_cost': normalized_cost,
        'penalty_load': norm_load_balance_violation,
        'penalty_heat': norm_heat_deficit,
        'penalty_batt': norm_battery_violation,
        'penalty_ev': norm_ev_violation
    }
    
    reward_load = (
    +5.0 * (1.0 - norm_load_balance_violation)
    - LOAD_BALANCE_WEIGHT * norm_load_balance_violation
)

    reward_heat = (
    +2.0 * (1.0 - norm_heat_deficit)
    - HEAT_BALANCE_WEIGHT * norm_heat_deficit
)

    reward_batt = (
    +1.0 * (1.0 - norm_battery_violation)
    - BATTERY_BOUNDS_WEIGHT * norm_battery_violation
)

    reward_ev = (
    +1.0 * (1.0 - norm_ev_violation)
    - EV_SOC_BOUNDS_WEIGHT * norm_ev_violation
    )

    




    # At the end of compute_reward function:
    
    reward = -normalized_cost + reward_load + reward_heat + reward_batt + reward_ev
    #reward = np.clip(reward, -10, 10)

    return reward, insights