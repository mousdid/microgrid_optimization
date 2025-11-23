import numpy as np
from gymnasium import spaces
from config import OBS_DIM, ACT_DIM

def get_observation_space():
    """
    Define observation space with appropriate bounds for each state variable.
    State vector includes all parameters from data/parameters/default plus decision variables.
    """
    
    # We'll build the bounds dynamically based on the specification
    low_bounds = []
    high_bounds = []
    
    # === PARAMETERS from data/parameters/default ===
    
    # Load parameter (power value)
    low_bounds.append(0.0)
    high_bounds.append(1000.0)
    
    # Price parameters (grid-related prices)
    low_bounds.extend([-100.0, -100.0])  # price_import, price_export
    high_bounds.extend([100.0, 100.0])
    
    # Cost parameters (cost-related)
    low_bounds.extend([ 0.0, 0.0, 0.0, 0.0, 0.0])  # Cop_ma_wt, Cop_ma_pv, rho_fuel, C_startup, C_degrad_es
    high_bounds.extend([ 100.0, 100.0, 100.0, 100.0, 100.0])
    
    # Efficiency parameters (binary/scalar factors)
    low_bounds.extend([0.0, 0.0, 0.0])  #  eta_dg, eta_ch_es, eta_dis_es
    high_bounds.extend([1.0, 1.0, 1.0])
    
 
    

    
    # Power capacity limits (power values)
    low_bounds.extend([0.0] * 7)  # P_grid_import_max, P_grid_export_max, PWT_max, PPV_max, PDG_max, Pdis_es_max, Pch_es_max
    high_bounds.extend([1000.0] * 7)
    
 
    
    # Energy capacity limits (energy values)
    low_bounds.extend([0.0, 0.0])  # Ees_min, Ees_max
    high_bounds.extend([10000.0, 10000.0])
    

    
    # === DECISION VARIABLES ===
    
    # Power values (current timestep)
    low_bounds.extend([0.0] * 7)  # p_import[t-1], p_export[t-1], p_wt[t-1], p_pv[t-1], p_dg[t-1], p_ch_es[t-1], p_dis_es[t-1]
    high_bounds.extend([1000.0] * 7)
    

    
    # Energy states (current and previous) - energy values
    low_bounds.extend([0.0])  # ees[t-1]
    high_bounds.extend([10000.0])
    
    # Binary states (current and previous) - binary/scalar factors
    low_bounds.extend([0.0] * 3)  #   u_dg[t-1], u_es[t-1] u_maingrid[t-1]
    high_bounds.extend([1.0] * 3)
    
    # Convert to numpy arrays
    low = np.array(low_bounds, dtype=np.float32)
    high = np.array(high_bounds, dtype=np.float32)
    
    # Ensure we have the right dimension
    if len(low) != OBS_DIM:
        print(f"Warning: Computed observation dimension {len(low)} doesn't match OBS_DIM {OBS_DIM}")
        # Pad or trim to match OBS_DIM
        if len(low) < OBS_DIM:
            # Pad with default bounds
            low = np.concatenate([low, np.zeros(OBS_DIM - len(low), dtype=np.float32)])
            high = np.concatenate([high, np.ones(OBS_DIM - len(high), dtype=np.float32) * 1000.0])
        else:
            # Trim to fit
            low = low[:OBS_DIM]
            high = high[:OBS_DIM]
    
    return spaces.Box(low=low, high=high, dtype=np.float32)

def get_action_space():
    """
    Define action space according to specification:
    - 2 continuous actions in [-1, 1]: for p_import/p_export and p_ch_es/p_dis_es
    - 3 continuous actions in [0, 1]: for p_wt, p_dg, p_pv
    """
    
    # Action vector structure:
    # [0] p_import/p_export control ([-1, 1]: negative=export, positive=import)
    # [1] p_ch_es/p_dis_es control ([-1, 1]: negative=discharge, positive=charge)
    # [2] p_wt control ([0, 1])
    # [3] p_dg control ([0, 1])
    # [4] p_pv control ([0, 1])
    
    low = np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    # Ensure we have the right dimension
    if len(low) != ACT_DIM:
        print(f"Warning: Computed action dimension {len(low)} doesn't match ACT_DIM {ACT_DIM}")
        # Adjust ACT_DIM in config.py if needed, or pad/trim here
        if len(low) < ACT_DIM:
            # Pad with [0, 1] bounds
            low = np.concatenate([low, np.zeros(ACT_DIM - len(low), dtype=np.float32)])
            high = np.concatenate([high, np.ones(ACT_DIM - len(high), dtype=np.float32)])
        else:
            # Trim to fit
            low = low[:ACT_DIM]
            high = high[:ACT_DIM]
    
    return spaces.Box(low=low, high=high, dtype=np.float32)