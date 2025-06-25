import pandas as pd
import matplotlib.pyplot as plt

# loading data
file_path = "data/raw/Payra_Original_load.csv"  
df = pd.read_csv(file_path)


df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#Summary Statistics 
max_load = df['total_electrical_load_served_(kw)'].max()
max_ch = df['battery_charge_power_(kw)'].max()
max_dis = df['battery_discharge_power_(kw)'].max()
max_soc = df['battery_state_of_charge_(%)'].max()
min_soc = df['battery_state_of_charge_(%)'].min()

print("---- Summary Statistics ----")
print(f"Max Load: {max_load:.2f} kW")
print(f"Max Battery Charge: {max_ch:.2f} kW")
print(f"Max Battery Discharge: {max_dis:.2f} kW")
print(f"SOC Range: {min_soc:.1f}% to {max_soc:.1f}%")

# === Scale to 100 kW System ===
df['scaled_load'] = df['total_electrical_load_served_(kw)'] / max_load * 100
df['scaled_p_ch_es'] = df['battery_charge_power_(kw)']
df['scaled_p_dis_es'] = df['battery_discharge_power_(kw)']



#Plot Scaled Load
plt.figure(figsize=(10, 4))
plt.plot(df['scaled_load'], label="Scaled Load (kW)")
plt.title("Scaled Load Profile (Max 100 kW)")
plt.xlabel("Hour")
plt.ylabel("Load (kW)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Estimate Ees_max (Battery Energy Capacity) 
soc_pct = df['battery_state_of_charge_(%)'] / 100
net_energy = (df['battery_charge_power_(kw)'] - df['battery_discharge_power_(kw)']).cumsum()
net_energy = net_energy - net_energy.min()  

valid = soc_pct > 0
estimated_ees_max = (net_energy[valid] / soc_pct[valid]).median()

print(f"\nEstimated Battery Capacity (Ees_max): {estimated_ees_max:.2f} kWh")
