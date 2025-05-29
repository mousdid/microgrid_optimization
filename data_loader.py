import os
import pandas as pd
from config import DATA_DIR, PARAM_FILES, TIME_STEPS
#dict of parameters to load from CSV files
def load_parameters():
    params = {}
    for name, fname in PARAM_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, index_col=0)
        series = df.iloc[:, 0].to_dict()
        params[name] = {t: series.get(t, 0) for t in TIME_STEPS}
    return params 