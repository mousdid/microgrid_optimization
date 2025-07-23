HORIZON = 720 # time steps per episode
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
LOAD_BALANCE_WEIGHT = 7 # Power balance constraint penalty
HEAT_BALANCE_WEIGHT = 7     # Heat demand constraint penalty
BATTERY_BOUNDS_WEIGHT = 3  # Battery SOC bounds violation penalty
EV_SOC_BOUNDS_WEIGHT = 3   # EV SOC bounds violation penalty

SCENARIO_TYPE = "normal"# "storage_failure", "load_spike", "outage"]

# State and action dimensions
OBS_DIM = 48  # Updated observation dimension
ACT_DIM = 7  # Updated action dimension