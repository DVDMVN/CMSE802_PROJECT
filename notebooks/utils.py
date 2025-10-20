import pandas as pd
import glob
import os

def load_data(verbose = False, partial = False) -> pd.DataFrame:
    data_path = "../data/raw/Kickstarter_2025-09-11T03_20_29_905Z"

    if partial:
        return pd.read_csv(data_path + '/Kickstarter.csv')

    all_files = sorted(glob.glob(os.path.join(data_path, "Kickstarter*.csv")))
    full_data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    if verbose:
        print(f"{len(all_files)} FILES LOADED | TOTAL ROWS: {full_data.__len__()}")
    return full_data