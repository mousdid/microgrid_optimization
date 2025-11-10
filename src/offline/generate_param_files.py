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
    'PCHP_max': 25.0,
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

# generating all the files
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

# generating EV availability with 10-hour random sessions
hours_per_year = len(df)
session_length = 10 #10 hours
presence_chance = 0.5
random_seed = 19 

np.random.seed(random_seed)

A = np.zeros(hours_per_year, dtype=int)
t = 0
while t + session_length <= hours_per_year:
    if np.random.rand() < presence_chance:
        A[t:t + session_length] = 1
        t += session_length + 1  # skip next hour to avoid overlap
    else:
        t += 1

# deriving session_start and leave_possible from A
session_start = np.zeros_like(A)
leave_possible = np.zeros_like(A)

for i in range(hours_per_year):
    if A[i] == 1 and (i == 0 or A[i - 1] == 0):
        session_start[i] = 1
    if A[i] == 1 and (i == hours_per_year - 1 or A[i + 1] == 0):
        leave_possible[i] = 1

# saving the availability and related flags
availability_df = pd.DataFrame({
    'A': A,
    'session_start': session_start,
    'leave_possible': leave_possible
})

availability_df[['A']].to_csv(f"{output_dir}/A.csv", index=False)
availability_df[['session_start']].to_csv(f"{output_dir}/session_start.csv", index=False)
availability_df[['leave_possible']].to_csv(f"{output_dir}/leave_possible.csv", index=False)

# Generate Eev_required: random requirement between 50 and 70 for each session
Eev_required = np.zeros(hours_per_year)
for t in range(hours_per_year):
    if session_start[t] == 1:
        # Assign a random value to the whole session
        required_energy = np.random.uniform(50, 70)
        i = t
        while i < hours_per_year and A[i] == 1:
            Eev_required[i] = required_energy
            i += 1
pd.DataFrame({'Eev_required': Eev_required}).to_csv(f"{output_dir}/Eev_required.csv", index=False)





def generate_ev_price_csv_simple(output_path='price_ev.csv'):#business rules


    # BEV-1 rates ($/kWh)
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
        else:  # 9:00 to 14:00
            return 'Super Off-Peak'

    # Generate hourly range for 2007
    start_time = datetime(2007, 1, 1, 0)
    end_time = datetime(2007, 12, 31, 23)
    hours_per_year = pd.date_range(start=start_time, end=end_time, freq='h')
    


    # Compute price_ev
    price_ev = [rates[get_tou_period(dt.hour)] for dt in hours_per_year][:HOURS]

    # Create DataFrame with a single column
    df = pd.DataFrame({'price_ev': price_ev})

    # Save to CSV
    df.to_csv(output_path, index=False)
    return df

generate_ev_price_csv_simple(output_path=f"{output_dir}/price_ev.csv")





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


def generate_heat_demand_csv(output_path='H_demand.csv'):
    # Create hourly timestamps for the full year
    hours = pd.date_range("2007-01-01", "2007-12-31 23:00", freq="H")

    # Parameters
    mean_liters_per_day = 100
    temp_inlet = 25
    temp_target = 45
    delta_T = temp_target - temp_inlet

    # Physical constants
    c = 4186  # J/kg·°C
    rho = 1000  # kg/m³

    # Base daily usage profile (6–9 AM, 6–9 PM)
    base_profile = np.zeros(24)
    base_profile[6:10] = 0.5
    base_profile[18:22] = 0.5
    base_profile /= base_profile.sum()

    # Generate hourly heat demand with jittered profiles
    hourly_liters = []
    random_seed = 19  # For reproducibility
    np.random.seed(random_seed)
    for _ in range(365):
        daily_liters = np.random.normal(mean_liters_per_day, 10)

        # Add random jitter to the daily usage pattern
        jitter = np.random.uniform(0.9, 1.1, size=24)
        daily_profile = base_profile * jitter
        daily_profile /= daily_profile.sum()

        hourly_split = daily_profile * daily_liters
        hourly_liters.extend(hourly_split)

    # Convert to kWh
    hourly_liters = np.array(hourly_liters)
    Q_J = hourly_liters / 1000 * rho * c * delta_T
    Q_kWh = Q_J / 3.6e6

    # Output DataFrame with just one column
    df = pd.DataFrame({"H_demand": Q_kWh}, index=hours)

    # Save to CSV
    df.to_csv(output_path, index=False)
    return df

generate_heat_demand_csv(output_path=f"{output_dir}/H_demand.csv")

print(f"✅ All PARAM_FILES saved in '{output_dir}' for {len(df)} hourly steps")
