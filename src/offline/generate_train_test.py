import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import HOURS_NUMBER_T_T,train_ratio

HOURS = HOURS_NUMBER_T_T # or 8760 for full year


# --- Define folders ---
input_csv = "data/raw/Payra_Original_load.csv"
input_price_csv = "data/raw/rt_hrl_lmps.csv"

output_dir = "data/trainset/1year"
output_test_dir = "data/testset/1year"
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
    'Cop_ma_wt': 0.02,
    'Cop_ma_pv': 0.01,
    'rho_fuel': 11.5,
    'eta_dg': 0.30,
    'C_startup': 5.0,
    'C_degrad_es': 0.02,
    'eta_ch_es': 0.90,
    'eta_dis_es': 0.90,
    'P_grid_import_max':1.5 * df['load'].max(),
    'P_grid_export_max': df['load'].max(),
    'PDG_max': df['PDG_max'].max(),
    'Pch_es_max':df['Pch_es_max'].max(),
    'Pdis_es_max':df['Pdis_es_max'].max()
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

# --- Save time series parameters so far ---
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
