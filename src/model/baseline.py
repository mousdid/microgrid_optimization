# baseline_controller.py

import numpy as np

class BaselineController:
    """
    Rule‑based controller:
      - WT, PV always at full capacity
      - DG runs at full only during 'outage'
      - Grid import/export (action[0]) set to exactly balance the remainder of the load
      - Battery (action[1]) and EV charge (action[5]) remain idle

    Action vector (length 7):
      [0]: grid ([-1=export, +1=import])
      [1]: storage ([-1=discharge, +1=charge])
      [2]: WT ([0,1])
      [3]: DG ([0,1])
      [4]: PV ([0,1])
    """

    def __init__(self, params: dict):
        # 'params' must include all time‑series lists (load, PWT_max, PPV_max,
        # PDG_max, P_grid_import_max, and 'scenario')
        self.params = params

    def select_action(self, t: int) -> np.ndarray:
        action = np.zeros(7, dtype=float)

        # Always dispatch renewables and CHP
        action[2] = 1.0  # WT
        action[4] = 1.0  # PV

        # DG only on outage
        action[3] = 1.0 if self.params['scenario'][t] == 'outage' else 0.0
        action[3] = 0.0

        

        # Calculate raw supply (kW)
        p_wt  = self.params['PWT_max'][t]   * action[2]
        p_pv  = self.params['PPV_max'][t]   * action[4]
        p_dg  = self.params['PDG_max'][t]   * action[3]

        # Compute net = load – supply
        load       = self.params['load'][t]
        p_grid_max = self.params['P_grid_import_max'][t]
        net        = load - (p_wt + p_pv  + p_dg)

        # storage  idle
        if self.params['scenario'][t] == 'storage_failure':
            action[1] = 0.0
        elif self.params['scenario'][t] == 'outage': 
            action[1] = -1.0
        elif self.params['scenario'][t] == 'load_spike':
            action[1] = -1.0
        elif net>0 :
            action[1] = 1.0
        else:
            action[1] = 0.0



        # Normalize to [-1,1] for the grid action
        action[0] = np.clip(net / (p_grid_max + 1e-8), -1.0, 1.0)

        return action
