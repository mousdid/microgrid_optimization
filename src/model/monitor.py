import matplotlib.pyplot as plt

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
    def compute_cvar(self, alpha=0.05):
        import numpy as np
        if not self.history:
            return 0.0

        vals = []
        for r in self.history:
            v = 0.0
            v += r.get('penalty_load', 0.0)
            v += r.get('penalty_heat', 0.0)
            v += r.get('penalty_batt', 0.0)
            v += r.get('penalty_ev', 0.0)
            vals.append(v)

        if len(vals) == 0:
            return 0.0

        q = np.quantile(vals, 1.0 - alpha)
        tail = [v for v in vals if v >= q]
        return float(np.mean(tail)) if tail else 0.0
w