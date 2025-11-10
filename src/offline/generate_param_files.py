import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import  HOURS_NUMBER


HOURS = HOURS_NUMBER  # or 8760 for full year


#defining folders
input_csv = "data/raw/Payra_Original_load.csv" 
input_price_csv = "data/raw/rt_hrl_lmps.csv"  

output_dir = "data/parameters/1year"

os.makedirs(output_dir, exist_ok=True)


#loading
df = pd.read_csv(input_csv)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]


# extracting matching columns
df['load'] = df['total_electrical_load_served_(kw)']
df['PPV_max'] = df['photovoltaic_panel_power_output_(kw)']
df['PWT_max'] = df['wind_turbine_power_output_(kw)']
df['PDG_max'] = df['generator_power_output_(kw)']
df['Pch_es_max'] = df['battery_charge_power_(kw)']
df['Pdis_es_max'] = df['battery_discharge_power_(kw)']
df['soc'] = df['battery_state_of_charge_(%)'] / 100



#  Defining constant values 
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



# estimating the energy maximal

# Calculate signed power flow (positive: charge, negative: discharge)
P_t = df['battery_charge_power_(kw)'] - df['battery_discharge_power_(kw)']

# Compute SOC (0 to 1)
SOC = df['battery_state_of_charge_(%)'] / 100

# Shift for t+1
P_tp1 = P_t.shift(-1)
SOC_tp1 = SOC.shift(-1)

# Trapezoidal power average and delta SOC
P_avg = (P_t + P_tp1) / 2
delta_soc = SOC_tp1 - SOC

# Avoid division by zero or small delta
valid = delta_soc.abs() > 0.01

# Estimate Ees_max at each valid step
ees_estimates = (P_avg[valid]) / delta_soc[valid]

# Filter extreme outliers and take median
ees_estimates = ees_estimates[(ees_estimates > 10) & (ees_estimates < 1e5)]
Ees_max_estimate = ees_estimates.median()

# Fill into dataframe
df['Ees_max'] = Ees_max_estimate
df['Ees_min'] = 0.2 * Ees_max_estimate

# generating all the files so far
time_series_params = [
    'load', 'PPV_max', 'PWT_max', 'PDG_max',
    'Pch_es_max', 'Pdis_es_max', 'Ees_max', 'Ees_min'
]

for param in time_series_params:
     df[[param]].to_csv(f"{output_dir}/{param}.csv", index=False)

df = df.iloc[:HOURS]
# Saving constant files as repeated 8760-row 
for param, value in constants.items():
    pd.DataFrame({param: [value] * len(df)}).to_csv(f"{output_dir}/{param}.csv", index=False)




#gerneating the prices

def extract_pjm_rto_prices(input_csv, import_output='price_import.csv', export_output='price_export.csv'):
  
    # Load full PJM pricing data
    df = pd.read_csv(input_csv)

    # Filter to only PJM-RTO rows
    pjm_rto_df = df[df['pnode_name'] == 'PJM-RTO']

    # Extract the total_lmp_rt column and rename it to 'price'
    prices_import = pjm_rto_df[['total_lmp_rt']].rename(columns={'total_lmp_rt': 'price_import'})
    prices_export = pjm_rto_df[['total_lmp_rt']].rename(columns={'total_lmp_rt': 'price_export'})
    prices_export=prices_export.head(HOURS)/1000  # Limit to first 48 hours
    prices_import = prices_import.head(HOURS)/1000  # Limit to first 48 hours

    # Save to two separate files
    prices_import.to_csv(import_output, index=False)
    prices_export.to_csv(export_output, index=False)

    return prices_import, prices_export

extract_pjm_rto_prices(input_csv=input_price_csv, 
                          import_output=f"{output_dir}/price_import.csv", 
                          export_output=f"{output_dir}/price_export.csv")




print(f"✅ All PARAM_FILES saved in '{output_dir}' for {len(df)} hourly steps")
