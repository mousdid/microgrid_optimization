import matplotlib.pyplot as plt
from config import LOAD_BALANCE_WEIGHT,COST_WEIGHT

class RewardTracker:
    def __init__(self):
        self.history = []

    def log(self, breakdown):
        # breakdown should be a dict with all reward components
        self.history.append(breakdown.copy())

    def plot(self):
        if not self.history:
            print("No reward data to plot.")
            return

        keys = [k for k in self.history[0].keys() if k not in ("timestep", "total_true_cost")]
        timesteps = [r.get('timestep', i) for i, r in enumerate(self.history)]

        plt.figure(figsize=(12, 8))
        for key in keys:
            values = [r[key] for r in self.history]
            plt.plot(timesteps, values, label=key)
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.title("Reward Component Evolution")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def clear(self):
        self.history = []
    # def compute_cvar(self, alpha=0.05):
    #     import numpy as np
    #     if not self.history:
    #         return 0.0

    #     vals = []
    #     for r in self.history:
    #         v = 0.0
    #         v += r.get('penalty_load', 0.0)

    #         vals.append(v)

    #     if len(vals) == 0:
    #         return 0.0

    #     q = np.quantile(vals, 1.0 - alpha)
    #     tail = [v for v in vals if v >= q]
    #     return float(np.mean(tail)) if tail else 0.0
    

class CVARTracker:
    def __init__(self):
        self.episode_vals = []

    def log(self, breakdown):
        # Only store the penalty (scalar)
        self.episode_vals.append(LOAD_BALANCE_WEIGHT*breakdown.get('penalty_load', 0.0))
        self.episode_vals.append(COST_WEIGHT*breakdown.get('total_cost', 0.0))

    def compute_cvar(self, alpha=0.05):
        import numpy as np
        if not self.episode_vals:
            return 0.0

        vals = np.array(self.episode_vals)
        q = np.quantile(vals, 1.0 - alpha)
        tail = vals[vals >= q]
        return float(tail.mean()) if len(tail) > 0 else 0.0

    def clear(self):
        self.episode_vals = []
