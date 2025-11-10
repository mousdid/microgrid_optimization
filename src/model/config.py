HORIZON = 7008
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
LEARNING_RATE = 3e-4
N_STEPS = 2048
N_EPOCHS = 10
BATCH_SIZE = 64
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


# Reward weights for constraint penalties
LOAD_BALANCE_WEIGHT = 0.1 # Power balance constraint penalty 7


SCENARIO_TYPES = ["outage"]

# State and action dimensions
OBS_DIM = 32  # Updated observation dimension
ACT_DIM = 5  # Updated action dimension

# --- Risk (CVaR) settings ---
CVAR_ALPHA = 0.05           # worst 5% tail
CVAR_VIOL_WEIGHT = 5.0      # how hard to push tail violations down (tune 2–10)
