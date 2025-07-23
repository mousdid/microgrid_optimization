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

        keys = [k for k in self.history[0].keys() if k != 'timestep']
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