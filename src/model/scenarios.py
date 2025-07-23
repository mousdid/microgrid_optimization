from config import SCENARIO_TYPE
import numpy as np

def generate_scenario(data, window=10, kind=SCENARIO_TYPE):
    # Safety check to prevent randint error when window is too large
    if len(data["price_import"]) <= window:
        window = max(1, len(data["price_import"]) - 1)
        print(f"Warning: Window size reduced to {window} due to data constraints")

    idx = np.random.randint(0, max(1, len(data["price_import"]) - window))
    
    # Slice the window segment from the data
    segment = {k: v[idx:idx+window].copy() for k, v in data.items() if isinstance(v, list) and len(v) > idx}

    # Apply scenario-specific modifications
    
    if kind == "storage_failure":
        if "pch_es_max" in segment:
            segment["pch_es_max"] = [0.0] * window
        if "pdis_es_max" in segment:
            segment["pdis_es_max"] = [0.0] * window
        segment["scenario"] = "storage_failure"

    elif kind == "load_spike":
        if "load" in segment:
            spike_factor = np.random.uniform(1.5, 2.5)
            segment["load"] = [l * spike_factor for l in segment["load"]]
        segment["scenario"] = "load_spike"

    elif kind == "outage":
        if "P_grid_export_max" in segment:
            segment["P_grid_export_max"] = [0.0] * window
        if "P_grid_import_max" in segment:
            segment["P_grid_import_max"] = [0.0] * window
        segment["scenario"] = "outage"

    else:
        segment["scenario"] = "normal"

    return segment
