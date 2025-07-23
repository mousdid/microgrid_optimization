import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import HOURS_NUMBER

HOURS = 4000  # or 8760 for full year
train_ratio = 0.8

# --- Define folders ---
input_csv = "data/raw/Payra_Original_load.csv"
input_price_csv = "data/raw/rt_hrl_lmps.csv"

output_dir = "data/trainset/default"
output_test_dir = "data/testset/default"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# --- Load and preprocess ---
df = pd.read_csv(input_csv)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

df['load'] = df['total_electrical_load_served_(kw)']
df['PPV_max'] = df['photovoltaic_panel_power_output_(kw)']
df['PWT_max'] = df['wind_turbine_power_output_(kw)']
df['PDG_max'] = df['generator_power_output_(kw)']
df['Pch_es_max'] = df['battery_charge_power_(kw)']
df['Pdis_es_max'] = df['battery_discharge_power_(kw)']
df['soc'] = df['battery_state_of_charge_(%)'] / 100

# --- Constants ---
constants = {
    'rho_gas': 0.3,
    'eta_chp': 0.40,
    'Cop_ma_wt': 0.02,
    'Cop_ma_pv': 0.01,
    'rho_fuel': 11.5,
    'eta_dg': 0.30,
    'C_startup': 5.0,
    'C_degrad_es': 0.02,
    'PEV_max': 60.0,
    'eta_ch_es': 0.90,
    'eta_dis_es': 0.90,
    'eta_ch_ev': 0.95,
    'alpha_chp': 0.80,
    'H_demand': 5.0,
    'PCHP_max': 25.0,
    'P_grid_import_max': 1.5 * df['load'].max(),
    'P_grid_export_max': df['load'].max(),
    'PDG_max': df['PDG_max'].max()
}

# --- Estimate Ees_max ---
P_t = df['battery_charge_power_(kw)'] - df['battery_discharge_power_(kw)']
SOC = df['battery_state_of_charge_(%)'] / 100
P_tp1 = P_t.shift(-1)
SOC_tp1 = SOC.shift(-1)
P_avg = (P_t + P_tp1) / 2
delta_soc = SOC_tp1 - SOC
valid = delta_soc.abs() > 0.01
ees_estimates = P_avg[valid] / delta_soc[valid]
ees_estimates = ees_estimates[(ees_estimates > 10) & (ees_estimates < 1e5)]
Ees_max_estimate = ees_estimates.median()
df['Ees_max'] = Ees_max_estimate
df['Ees_min'] = 0.2 * Ees_max_estimate

# --- Split into train and test ---
df = df.iloc[:HOURS]
split_index = int(train_ratio * HOURS)
df_train = df.iloc[:split_index].copy()
df_test = df.iloc[split_index:].copy()

# --- Save time series parameters ---
time_series_params = [
    'load', 'PPV_max', 'PWT_max', 'PDG_max',
    'Pch_es_max', 'Pdis_es_max', 'Ees_max', 'Ees_min'
]

for param in time_series_params:
    df_train[[param]].to_csv(f"{output_dir}/{param}.csv", index=False)
    df_test[[param]].to_csv(f"{output_test_dir}/{param}.csv", index=False)

# --- Save constants ---
for param, value in constants.items():
    pd.DataFrame({param: [value] * len(df_train)}).to_csv(f"{output_dir}/{param}.csv", index=False)
    pd.DataFrame({param: [value] * len(df_test)}).to_csv(f"{output_test_dir}/{param}.csv", index=False)

# --- Generate EV availability ---
hours_per_year = len(df)
session_length = 10
presence_chance = 0.5
random_seed = 19

np.random.seed(random_seed)
A = np.zeros(hours_per_year, dtype=int)
t = 0
while t + session_length <= hours_per_year:
    if np.random.rand() < presence_chance:
        A[t:t + session_length] = 1
        t += session_length + 1
    else:
        t += 1

session_start = np.zeros_like(A)
leave_possible = np.zeros_like(A)

for i in range(hours_per_year):
    if A[i] == 1 and (i == 0 or A[i - 1] == 0):
        session_start[i] = 1
    if A[i] == 1 and (i == hours_per_year - 1 or A[i + 1] == 0):
        leave_possible[i] = 1

availability_df = pd.DataFrame({
    'A': A,
    'session_start': session_start,
    'leave_possible': leave_possible
})

# --- Split and save EV availability ---
availability_train = availability_df.iloc[:split_index]
availability_test = availability_df.iloc[split_index:]
Eev_required = np.zeros(hours_per_year)

for t in range(hours_per_year):
    if session_start[t] == 1:
        required_energy = np.random.uniform(50, 70)
        i = t
        while i < hours_per_year and A[i] == 1:
            Eev_required[i] = required_energy
            i += 1

Eev_required_train = Eev_required[:split_index]
Eev_required_test = Eev_required[split_index:]

availability_train[['A']].to_csv(f"{output_dir}/A.csv", index=False)
availability_test[['A']].to_csv(f"{output_test_dir}/A.csv", index=False)

availability_train[['session_start']].to_csv(f"{output_dir}/session_start.csv", index=False)
availability_test[['session_start']].to_csv(f"{output_test_dir}/session_start.csv", index=False)

availability_train[['leave_possible']].to_csv(f"{output_dir}/leave_possible.csv", index=False)
availability_test[['leave_possible']].to_csv(f"{output_test_dir}/leave_possible.csv", index=False)

pd.DataFrame({'Eev_required': Eev_required_train}).to_csv(f"{output_dir}/Eev_required.csv", index=False)
pd.DataFrame({'Eev_required': Eev_required_test}).to_csv(f"{output_test_dir}/Eev_required.csv", index=False)

# --- EV price generation ---
def generate_ev_price_csv_simple():
    rates = {
        'Peak': 0.38079,
        'Off-Peak': 0.18878,
        'Super Off-Peak': 0.16212
    }

    def get_tou_period(hour):
        if 16 <= hour < 21:
            return 'Peak'
        elif (21 <= hour <= 23) or (0 <= hour < 9) or (14 <= hour < 16):
            return 'Off-Peak'
        else:
            return 'Super Off-Peak'

    start_time = datetime(2007, 1, 1, 0)
    end_time = datetime(2007, 12, 31, 23)
    hours_per_year = pd.date_range(start=start_time, end=end_time, freq='h')
    price_ev = [rates[get_tou_period(dt.hour)] for dt in hours_per_year][:HOURS]
    return pd.Series(price_ev, name='price_ev')

price_ev = generate_ev_price_csv_simple()
price_ev[:split_index].to_frame().to_csv(f"{output_dir}/price_ev.csv", index=False)
price_ev[split_index:].to_frame().to_csv(f"{output_test_dir}/price_ev.csv", index=False)

# --- Price import/export ---
def extract_pjm_rto_prices(input_csv):
    df = pd.read_csv(input_csv)
    pjm_rto_df = df[df['pnode_name'] == 'PJM-RTO']
    price_series = pjm_rto_df['total_lmp_rt'].iloc[:HOURS] / 1000
    return price_series.rename('price_import'), price_series.rename('price_export')

price_import, price_export = extract_pjm_rto_prices(input_price_csv)
price_import[:split_index].to_frame().to_csv(f"{output_dir}/price_import.csv", index=False)
price_import[split_index:].to_frame().to_csv(f"{output_test_dir}/price_import.csv", index=False)

price_export[:split_index].to_frame().to_csv(f"{output_dir}/price_export.csv", index=False)
price_export[split_index:].to_frame().to_csv(f"{output_test_dir}/price_export.csv", index=False)

print(f"✅ All PARAM_FILES saved in '{output_dir}' (train) and '{output_test_dir}' (test)")
