"""
Lightweight utility helpers shared across modules.

Currently includes:
- loading raw CSV data from disk
- loading processed parquet data
- small JSON parsing helper

Author: Alex (Ze) Chen
Date: 2025-10-24
"""

import pandas as pd
import glob
import os
import json
from src.config import DATA_PATH, PROCESSED_DATA_PATH

def load_raw_data(verbose = True, partial = False, use_relative = False) -> pd.DataFrame:
    """
    Load the raw Kickstarter CSV files from DATA_PATH.

    Parameters
    ----------
    verbose : bool, default=True
        If True, print the number of files and total rows.
    partial : bool, default=False
        If True, load a single CSV named 'Kickstarter.csv'
        instead of concatenating all Kickstarter*.csv files.
    use_relative : bool, default=False
        If True, prefix '../' to the data path (for use inside notebooks
        running from a subdirectory).

    Returns
    -------
    pandas.DataFrame
        Raw combined dataset.
    """
    TRUE_DATA_PATH = DATA_PATH if not use_relative else "../" + DATA_PATH
    if partial:
        return pd.read_csv(TRUE_DATA_PATH + '/Kickstarter.csv')

    all_files = sorted(glob.glob(os.path.join(TRUE_DATA_PATH, "Kickstarter*.csv")))
    full_data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    if verbose:
        print(f"{len(all_files)} FILES LOADED | TOTAL ROWS: {full_data.__len__()}")
    return full_data

def load_processed_data(use_relative = False) -> pd.DataFrame:
    """
    Load the post-processed parquet dataset from disk.

    Parameters
    ----------
    use_relative : bool, default=False
        If True, prefix '../' to the processed data path
        (useful when called from notebooks in subfolders).

    Returns
    -------
    pandas.DataFrame
        Fully preprocessed dataset ready for modeling.
    """
    TRUE_PROCESSED_DATA_PATH = PROCESSED_DATA_PATH if not use_relative else "../" + PROCESSED_DATA_PATH
    return pd.read_parquet(TRUE_PROCESSED_DATA_PATH)

def parse_json_feature(raw):
    """
    Parse a JSON string field from the dataset into a Python dict.

    Parameters
    ----------
    raw : str
        JSON string.

    Returns
    -------
    dict
        Parsed JSON object.
    """
    return json.loads(raw)