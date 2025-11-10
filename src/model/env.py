import numpy as np
import os
import gymnasium as gym 
from state_action import get_observation_space, get_action_space
from config import OBS_DIM,HORIZON, CVAR_ALPHA, CVAR_VIOL_WEIGHT
from dynamics import update_battery_soc, update_ev_soc, process_grid_action, process_battery_action
from reward import compute_reward
from monitor import RewardTracker  




class MicrogridEnv(gym.Env):
    def __init__(self, data, params,horizon=HORIZON):
        print(f"[ENV __init__ PID={os.getpid()}]")
        super().__init__()
        self.data = data
        self.params = params
        self.T = len(params.get("load", [horizon]))  #
        self.t = 0
        

        # Initialize storage and startup trackers
        self.soc_es = params.get('Ees_min', [0])[0] if 'Ees_min' in params and len(params['Ees_min']) > 0 else 0  # Battery to min
        self.soc_ev = 0.0                   
        self.prev_u_chp = 0
        self.prev_u_dg = 0
        self.prev_u_es = 0
        self.prev_u_maingrid = 0
        
        # Store previous decision variables for state vector
        self.prev_powers = {
            'p_import': 0, 'p_export': 0, 'p_wt': 0, 'p_pv': 0, 'p_chp': 0, 
            'p_dg': 0, 'p_ch_es': 0, 'p_dis_es': 0, 'p_ch_ev': 0
        }
        self.prev_H_chp = 0
        self.prev_soc_es = 0
        self.prev_soc_ev = 0

        self.observation_space = get_observation_space()
        self.action_space = get_action_space()
        self.reward_tracker = RewardTracker()  # Track reward components

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        self.t = 0
        self.soc_es = self.params.get('Ees_min', [0])[0] if 'Ees_min' in self.params and len(self.params['Ees_min']) > 0 else 0
        self.soc_ev = 0.0
        self.prev_u_chp = 0
        self.prev_u_dg = 0
        self.prev_u_es = 0
        self.prev_u_maingrid = 0
        
        # Reset previous decision variables
        self.prev_powers = {
            'p_import': 0, 'p_export': 0, 'p_wt': 0, 'p_pv': 0, 'p_chp': 0, 
            'p_dg': 0, 'p_ch_es': 0, 'p_dis_es': 0, 'p_ch_ev': 0
        }
        self.prev_H_chp = 0
        self.prev_soc_es = self.soc_es
        self.prev_soc_ev = self.soc_ev
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Build observation vector according to state_action.py specification"""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        
        # Safe parameter access
        def safe_param(name, default=0):
            if name in self.params and self.t < len(self.params[name]):
                return self.params[name][self.t]
            return default
        
        idx = 0
        
        # === PARAMETERS from data/parameters/default ===
        
        # Load parameter (power value)
        obs[idx] = safe_param("load")
        idx += 1
        
        # Price parameters (grid-related prices)
        obs[idx:idx+3] = [safe_param("price_import"), safe_param("price_export"), safe_param("price_ev")]
        idx += 3
        
        # Cost parameters (cost-related)
        obs[idx:idx+6] = [safe_param("rho_gas"), safe_param("Cop_ma_wt"), safe_param("Cop_ma_pv"), 
                          safe_param("rho_fuel"), safe_param("C_startup"), safe_param("C_degrad_es")]
        idx += 6
        
        # Efficiency parameters (binary/scalar factors)
        obs[idx:idx+6] = [safe_param("eta_chp"), safe_param("eta_dg"), safe_param("eta_ch_es"), 
                          safe_param("eta_dis_es"), safe_param("eta_ch_ev"), safe_param("alpha_chp")]
        idx += 6
        
        # Heat demand (power/heat value)
        obs[idx] = safe_param("H_demand")
        idx += 1
        
        # Power capacity limits (power values)
        obs[idx:idx+8] = [safe_param("P_grid_import_max"), safe_param("P_grid_export_max"), 
                          safe_param("PWT_max"), safe_param("PPV_max"), safe_param("PCHP_max"), 
                          safe_param("PDG_max"), safe_param("Pdis_es_max"), safe_param("Pch_es_max")]
        idx += 8
        
        # EV max power (power value)
        obs[idx] = safe_param("PEV_max")
        idx += 1
        
        # Energy capacity limits (energy values)
        obs[idx:idx+2] = [safe_param("Ees_min"), safe_param("Ees_max")]
        idx += 2
        
        # EV required energy (energy value)
        obs[idx] = safe_param("Eev_required")
        idx += 1
        
        # EV availability and session flags (binary/scalar factors)
        obs[idx:idx+3] = [safe_param("A"), safe_param("session_start"), safe_param("leave_possible")]
        idx += 3
        
        # === DECISION VARIABLES ===
        
        # Power values from previous timestep
        obs[idx:idx+9] = [self.prev_powers['p_import'], self.prev_powers['p_export'], 
                          self.prev_powers['p_wt'], self.prev_powers['p_pv'], 
                          self.prev_powers['p_chp'], self.prev_powers['p_dg'], 
                          self.prev_powers['p_ch_es'], self.prev_powers['p_dis_es'], 
                          self.prev_powers['p_ch_ev']]
        idx += 9
        
        # Heat variable from previous timestep
        obs[idx] = self.prev_H_chp
        idx += 1
        
        # Energy states from previous timestep
        obs[idx:idx+2] = [self.prev_soc_es, self.prev_soc_ev]
        idx += 2
        
        # Binary states from previous timestep
        obs[idx:idx+4] = [self.prev_u_chp, self.prev_u_dg, self.prev_u_es, self.prev_u_maingrid]
        idx += 4
        
        return obs

    def step(self, action):
        """
        Implement state transition logic based on action and current parameters
        
        Action structure:
        [0] p_import/p_export control ([-1, 1]: negative=export, positive=import)
        [1] p_ch_es/p_dis_es control ([-1, 1]: negative=discharge, positive=charge)
        [2] p_wt control ([0, 1])
        [3] p_chp control ([0, 1])
        [4] p_dg control ([0, 1])
        [5] p_ch_ev control ([0, 1])
        [6] p_pv control ([0, 1])
        """
        
        # Safe parameter access
        def safe_param(name, default=0):
            if name in self.params and self.t < len(self.params[name]):
                return self.params[name][self.t]
            return default
        
        # === STEP 1: Extract Parameters ===
        # Load all required parameters for current timestep
        p_import_max = safe_param("P_grid_import_max", 100)
        p_export_max = safe_param("P_grid_export_max", 100)
        p_ch_es_max = safe_param("Pch_es_max", 50)
        p_dis_es_max = safe_param("Pdis_es_max", 50)
        p_wt_max = safe_param("PWT_max", 50)
        p_pv_max = safe_param("PPV_max", 50)
        p_chp_max = safe_param("PCHP_max", 25)
        p_dg_max = safe_param("PDG_max", 30)
        p_ev_max = safe_param("PEV_max", 60)
        
        # Get efficiency parameters
        eta_ch_es = safe_param("eta_ch_es", 0.9)
        eta_dis_es = safe_param("eta_dis_es", 0.9)
        eta_ch_ev = safe_param("eta_ch_ev", 0.95)
        alpha_chp = safe_param("alpha_chp", 0.8)
        
        # Get EV session parameters
        is_session_start = safe_param("session_start", 0) == 1
        ev_availability = safe_param("A", 0)
        
        # === STEP 2: Process Actions ===
        
        # Process grid action using dynamics function
        p_import, p_export, u_maingrid = process_grid_action(action[0], p_import_max, p_export_max)
        
        # Process battery action using dynamics function
        p_ch_es, p_dis_es, u_es = process_battery_action(action[1], p_ch_es_max, p_dis_es_max)
        
        # Scale power values directly from actions [2-6]
        p_wt = action[2] * p_wt_max
        p_chp = action[3] * p_chp_max
        p_dg = action[4] * p_dg_max
        p_ch_ev = action[5] * p_ev_max * ev_availability  # Only charge if EV is available
        p_pv = action[6] * p_pv_max
        
        # === STEP 3: Compute Decision Variables ===
        
        # Heat production from CHP
        H_chp = alpha_chp * p_chp
        
        # Binary unit status based on power output
        u_chp = 1 if p_chp > 0 else 0
        u_dg = 1 if p_dg > 0 else 0
        
        # === STEP 4: Update Energy States ===
        
        # Update battery SOC using dynamics function
        new_soc_es = update_battery_soc(self.soc_es, p_ch_es, p_dis_es, eta_ch_es, eta_dis_es)
        
        # Apply battery capacity constraints
        ees_min = safe_param("Ees_min", 0)
        ees_max = safe_param("Ees_max", 300)
        new_soc_es = max(ees_min, min(new_soc_es, ees_max))
        
        # Update EV SOC using dynamics function
        new_soc_ev = update_ev_soc(self.soc_ev, p_ch_ev, eta_ch_ev, is_session_start)
        
        new_soc_ev = update_ev_soc(self.soc_ev, p_ch_ev, eta_ch_ev, is_session_start)

        # Enforce physical SOC limits (e.g. 0–70 kWh)
        Eev_min = 0.2*70
        Eev_max = 70
        new_soc_ev = np.clip(new_soc_ev, Eev_min, Eev_max)


        
        # === STEP 5: Store Current State for Next Timestep ===
        
        # Store previous values for state vector
        self.prev_powers = {
            'p_import': p_import, 'p_export': p_export, 'p_wt': p_wt, 'p_pv': p_pv, 
            'p_chp': p_chp, 'p_dg': p_dg, 'p_ch_es': p_ch_es, 'p_dis_es': p_dis_es, 
            'p_ch_ev': p_ch_ev
        }
        self.prev_H_chp = H_chp
        self.prev_soc_es = self.soc_es
        self.prev_soc_ev = self.soc_ev
        self.prev_u_chp = u_chp
        self.prev_u_dg = u_dg
        self.prev_u_es = u_es
        self.prev_u_maingrid = u_maingrid
        
        # Update current energy states
        self.soc_es = new_soc_es
        self.soc_ev = new_soc_ev
        
        # === STEP 6: Compute Reward ===
        
        # Get current parameters needed for reward calculation
        current_load = safe_param("load", 0)
        H_demand = safe_param("H_demand", 0)
        leave_possible = safe_param("leave_possible", 0)
        Eev_required = safe_param("Eev_required", 0)
        
        # Calculate reward using the reward function
        reward,breakdown = compute_reward(
            t=self.t,
            # Power variables
            p_import=p_import, p_export=p_export, p_wt=p_wt, p_pv=p_pv, 
            p_chp=p_chp, p_dg=p_dg, p_dis_es=p_dis_es, p_ch_es=p_ch_es, p_ch_ev=p_ch_ev,
            # Price parameters
            price_import=safe_param("price_import", 0),
            price_export=safe_param("price_export", 0),
            price_ev=safe_param("price_ev", 0),
            # Cost parameters
            Cop_ma_wt=safe_param("Cop_ma_wt", 0),
            Cop_ma_pv=safe_param("Cop_ma_pv", 0),
            rho_gas=safe_param("rho_gas", 0),
            rho_fuel=safe_param("rho_fuel", 0),
            C_startup=safe_param("C_startup", 0),
            C_degrad_es=safe_param("C_degrad_es", 0),
            # Efficiency parameters
            eta_chp=safe_param("eta_chp", 0.9),
            eta_dg=safe_param("eta_dg", 0.9),
            # Binary states (current and previous)
            u_chp=u_chp, u_dg=u_dg, 
            prev_u_chp=self.prev_u_chp, prev_u_dg=self.prev_u_dg,
            # Load and heat
            load=current_load, H_demand=H_demand, H_chp=H_chp,
            # Energy states
            soc_es=self.soc_es, soc_ev=self.soc_ev,
            ees_min=ees_min, ees_max=ees_max,
            # EV parameters
            leave_possible=leave_possible, Eev_required=Eev_required
        )
        
        # === STEP 7: Update Time and Check Termination ===
        self.t += 1
        terminated = self.t >= self.T
        truncated = False
        

        self.reward_tracker.log(breakdown)
        if terminated:
            cvar_vio = self.reward_tracker.compute_cvar(alpha=CVAR_ALPHA)
            reward -= CVAR_VIOL_WEIGHT * cvar_vio
            breakdown["cvar_vio"] = cvar_vio

        
        return self._get_obs(), float(reward), terminated, truncated, breakdown

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
