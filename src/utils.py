import pandas as pd
import glob
import os
import json
from src.config import DATA_PATH, PROCESSED_DATA_PATH

def load_raw_data(verbose = True, partial = False, use_relative = False) -> pd.DataFrame:
    TRUE_DATA_PATH = DATA_PATH if not use_relative else "../" + DATA_PATH
    if partial:
        return pd.read_csv(TRUE_DATA_PATH + '/Kickstarter.csv')

    all_files = sorted(glob.glob(os.path.join(TRUE_DATA_PATH, "Kickstarter*.csv")))
    full_data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    if verbose:
        print(f"{len(all_files)} FILES LOADED | TOTAL ROWS: {full_data.__len__()}")
    return full_data

def load_processed_data(use_relative = False) -> pd.DataFrame:
    TRUE_PROCESSED_DATA_PATH = PROCESSED_DATA_PATH if not use_relative else "../" + PROCESSED_DATA_PATH
    return pd.read_parquet(TRUE_PROCESSED_DATA_PATH)

def parse_json_feature(raw):
    return json.loads(raw)