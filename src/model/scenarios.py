from config import HORIZON

import numpy as np
import copy
import random
import os
import pandas as pd

def generate_mixed_scenario_dataset(data, total_hours=HORIZON, seed=19, number_events=5):
    """
    Returns a full-length dataset with embedded scenario episodes.
    """
    random.seed(seed)
    np.random.seed(seed)
    new_data = {k: copy.deepcopy(v[:total_hours]) for k, v in data.items() if isinstance(v, list)}
    scenario_tags = ['normal'] * total_hours
    
    def insert_event(kind, duration_range, count):
        for _ in range(count):
            duration = np.random.randint(*duration_range)
            start = np.random.randint(0, total_hours - duration)
            for t in range(start, start + duration):
                scenario_tags[t] = kind

                if kind == "storage_failure":
                    if "pch_es_max" in new_data:
                        new_data["pch_es_max"][t] = 0.0
                    if "pdis_es_max" in new_data:
                        new_data["pdis_es_max"][t] = 0.0
                elif kind == "outage":
                    if "P_grid_import_max" in new_data:
                        new_data["P_grid_import_max"][t] = 0.0
                    if "P_grid_export_max" in new_data:
                        new_data["P_grid_export_max"][t] = 0.0
                elif kind == "load_spike":
                    if "load" in new_data:
                        new_data["load"][t] *= np.random.uniform(1.5, 2.5)


    def save_scenario(data, total_hours, seed, output_dir='../output/scenarios'):
        """
        Saves the scenario data as a CSV file in the specified directory.

        Args:
            data (pd.DataFrame): The scenario data to save.
            total_hours (int): Horizon value used in the filename.
            seed (int): Seed value used in the filename.
            output_dir (str): The directory to save the CSV file.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create the filename
        filename = f'scenario_seed_{seed}_horizon_{total_hours}.csv'

        # Construct full path
        filepath = os.path.join(output_dir, filename)

        # Save DataFrame to CSV
        data.to_csv(filepath, index=False)
        
        print(f"Scenario saved to {filepath}")


    # Insert events (customize as needed)
    insert_event("outage", duration_range=(24, 72), count=number_events)
    #insert_event("storage_failure", duration_range=(48, 72), count=1)
    #insert_event("load_spike", duration_range=(1, 3), count=1)

    new_data["scenario"] = scenario_tags
    save_scenario(pd.DataFrame(new_data), total_hours, seed)
    return new_data
