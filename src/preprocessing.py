"""
Full preprocessing / feature engineering pipeline for Kickstarter data.

This module:
- handles duplicates & missingness
- removes leakage columns
- engineers text, category, and time-based features
- writes a final parquet output

Run this file directly to read raw CSVs, preprocess them, and save
the processed parquet to disk.

Author: Alex (Ze) Chen
Date: 2025-10-24
"""

import pandas as pd
from src.utils import parse_json_feature, load_raw_data
import json
import re

# ---------------------------------------------------------------
#               DATA INTEGRITY ANALYSIS FUNCTIONS
# ---------------------------------------------------------------

### ----- ----- ----- ----- DEDUPLICATION ----- ----- ----- -----

def apply_deduplication_steps(kickstarter) -> pd.DataFrame:
    """
    Remove duplicate projects and keep the "best" row per project ID.

    Strategy:
    1. Drop exact duplicate rows
    2. For rows with the same 'id', keep the one with the highest
       backers_count (assumes that represents the final/most supported version)

    Parameters
    ----------
    kickstarter : pandas.DataFrame
        Raw concatenated data.

    Returns
    -------
    pandas.DataFrame
        Deduplicated dataframe with a single row per project id.
    """
    kick_nodup = kickstarter.copy()

    # Eliminate true duplicates
    kick_nodup = kick_nodup.drop_duplicates()

    # Narrow down by most backers (built in recency)
    max_per_id = kick_nodup.groupby("id")["backers_count"].transform("max") # Group rows by id, get backers_count
                                                                            # transform("max") gives us a series 1:1 with original except filling each row with the group max
    kick_nodup = kick_nodup[kick_nodup["backers_count"] == max_per_id]

    # Drop the rest of the duplicates by `id`
    kick_nodup = kick_nodup.drop_duplicates(subset=['id']).reset_index()
    return kick_nodup

### ----- ----- ----- ----- MISSINGNESS ----- ----- ----- -----

def impute_blurb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing 'blurb' using 'name'.
    """
    df = df.copy()
    df['blurb'] = df['blurb'].fillna(df['name'])
    return df

def impute_usd_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing 'usd_type' with the mode.
    """
    df = df.copy()
    df["usd_type"] = df["usd_type"].fillna(df["usd_type"].mode()[0])
    return df

def impute_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 'location' JSON into columns and impute missing
    location fields using the most common value per country.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 'loc_name', 'loc_state', 'loc_type' added,
        and original 'location' column removed.
    """
    def parse_json_feature(raw):
        return json.loads(raw)
    locations_not_null = df[df['location'].notnull()]['location'].apply(parse_json_feature)
    location_cols = locations_not_null.apply(
        lambda d: pd.Series({
            'loc_name': d.get('name'),
            'loc_state': d.get('state'),
            'loc_type': d.get('type'),
        })
    )

    df = df.join(location_cols, how='left')

    mode_loc_name_by_country = (
        df.loc[df["loc_name"].notnull()]
        .groupby("country")["loc_name"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    df["loc_name"] = df["loc_name"].fillna(df["country"].map(mode_loc_name_by_country))

    mode_loc_state_by_country = (
        df.loc[df["loc_state"].notnull()]
        .groupby("country")["loc_state"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    df["loc_state"] = df["loc_state"].fillna(df["country"].map(mode_loc_state_by_country))

    mode_loc_type_by_country = (
        df.loc[df["loc_type"].notnull()]
        .groupby("country")["loc_type"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    df["loc_type"] = df["loc_type"].fillna(df["country"].map(mode_loc_type_by_country))

    df = df.drop(columns = ['location'])
    return df


def impute_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing monetary/currency-related fields using relationships
    between pledged amounts and exchange rates.
    """
    # Impute usd_pledged
    df['usd_pledged'] = df['usd_pledged'].fillna(df['pledged'] * df['static_usd_rate'])

    # Impute usd_exchange_rate
    mean_usd_exchange_rate_by_currency = df.groupby(by = 'currency')['usd_exchange_rate'].mean()
    df['usd_exchange_rate'] = df['usd_exchange_rate'].fillna(df['currency'].map(mean_usd_exchange_rate_by_currency))

    # Impute converted_pledged_amount
    df['converted_pledged_amount'] = df['converted_pledged_amount'].fillna(df['pledged'] * df['usd_exchange_rate'])
    return df

# This is transformation rather than imputation, but for our purpose, it is transformation in service to imputate
def impute_video(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 'has_video' boolean and drop raw 'video' blob-like field.
    """
    df['has_video'] = df['video'].notnull()
    df = df.drop(columns = ['video'])
    return df

def impute_is_in_post_campaign_pledging_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop 'is_in_post_campaign_pledging_phase' (not needed).
    """
    df = df.drop(columns = ['is_in_post_campaign_pledging_phase'])
    return df

def apply_missingness_handling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all imputation / cleanup steps for missing data.
    """
    df = impute_blurb(df)
    df = impute_usd_type(df)
    df = impute_location(df)
    df = impute_money_features(df)
    df = impute_video(df)
    df = impute_is_in_post_campaign_pledging_phase(df)
    return df


# ------------------------------------------------------------------------------------------------------------------------------
#                  DATA FILTERING, LEAKAGE PREVENTION, AND FEATURE RELEVANCE ASSESSMENT FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------------------

def keep_only_final_outcome_states(df: pd.DataFrame):
    """
    Keep only rows where final state is 'successful' or 'failed'.
    """
    df = df[df['state'].isin(['successful', 'failed'])].copy()
    return df

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that leak final outcome (pledged dollars, backers, etc).
    """
    leakage_columns = [
        "backers_count",
        "pledged",
        "usd_pledged",
        "converted_pledged_amount",
        "percent_funded",
        "usd_exchange_rate",
        "state_changed_at",
        "spotlight",
        # "is_in_post_campaign_pledging_phase", # Already removed by missingness handling
    ]

    df = df.drop(columns=leakage_columns)
    return df

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop metadata / UI columns not useful for prediction.
    """
    irrelevant_columns = [
        "id",
        "slug",
        "source_url",
        "urls",
        "currency_symbol",
        "currency_trailing_code",
        "current_currency",
        "is_liked",
        "is_disliked",
        "is_starrable",
        "country_displayable_name",
    ]

    df = df.drop(columns = irrelevant_columns)
    return df

def drop_photo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove 'photo' (image metadata structure).
    """
    df = df.drop(columns = ['photo'])
    return df

def drop_creator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove 'creator' (nested JSON-like blob).
    """
    df = df.drop(columns = ['creator']).copy()
    return df

def drop_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove 'profile' (not predictive / often sparse).
    """
    df = df.drop(columns = ['profile'])
    return df

def apply_data_filtering_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all final filtering steps:
    - keep only success/fail
    - remove leakage columns
    - drop noisy metadata blobs
    """
    df = keep_only_final_outcome_states(df)
    df = drop_leakage_columns(df)
    df = drop_irrelevant_columns(df)
    df = drop_photo(df)
    df = drop_creator(df)
    df = drop_profile(df)
    return df

# ---------------------------------------------------------------
#                  FEATURE ENGINEERING FUNCTIONS
# ---------------------------------------------------------------

def extract_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the nested 'category' JSON into flat columns.
    """
    categories = df['category'].apply(parse_json_feature)
    df['cat_name'] = categories.apply(lambda x: x.get('name'))
    df['cat_position'] = categories.apply(lambda x: x.get('position'))
    df['cat_parent_name'] = categories.apply(lambda x: x.get('parent_name'))
    df = df.drop(columns = ['category'])
    return df

def get_words(text: str) -> list[str]:
    """
    Tokenize text into alphanumeric 'words', lowercase.
    """
    words = re.findall(r"\b\w+\b", text.lower()) # Ignores punctuation
    return words

def get_avg_word_len(text: str) -> float:
    """
    Average word length in a text string.
    """
    words = get_words(text)
    return (sum(w.__len__() for w in words) / words.__len__()) if words else 0.0

def engineer_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic length / word-length features from name/blurb.
    """
    df['blurb_len'] = df['blurb'].apply(lambda x: x.strip().__len__())
    df['name_len'] = df['name'].apply(lambda x: x.strip().__len__())
    df['blurb_avg_word_len'] = df['blurb'].apply(get_avg_word_len)
    df['name_avg_word_len'] = df['name'].apply(get_avg_word_len)
    return df

def convert_timestamp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw Unix timestamps to pandas datetime.
    """
    df['created_at'] = pd.to_datetime(df['created_at'], unit='s')
    df['deadline'] = pd.to_datetime(df['deadline'], unit='s')
    df['launched_at'] = pd.to_datetime(df['launched_at'], unit='s')
    return df

def engineer_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add campaign duration (days between launch and deadline).
    """
    df['duration'] = (df['deadline'] - df['launched_at']).dt.days
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering in sequence:
    - categories
    - text stats
    - timestamp conversion
    - duration
    """
    df = extract_categories(df)
    df = engineer_text_features(df)
    df = convert_timestamp_features(df)
    df = engineer_duration(df)
    return df

# ---------------------------------------------------------------
#          ADDITIONAL PREPROCESSING ADDED ON DURING EDA
# ---------------------------------------------------------------

def drop_additional_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop leftover columns judged unhelpful after EDA.
    """
    irrelevant_cols = [
        'disable_communication',
        'is_launched'
    ]
    df = df.drop(columns = irrelevant_cols)
    return df

def apply_additional_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for final cleanup after EDA review.
    """
    df = drop_additional_irrelevant_columns(df)
    return df

# ===============================================================
#                             MAIN
# ===============================================================

from src.config import PROCESSED_DATA_PATH

def apply_preprocessing(df: pd.DataFrame, verbose = True) -> pd.DataFrame:
    """
    Full preprocessing driver.

    Steps (in order):
    1. Deduplicate projects
    2. Impute/fix missing values
    3. Filter out leakage and unused metadata
    4. Engineer features (text stats, categories, durations, timestamps)
    5. Drop final irrelevant cols from EDA

    Parameters
    ----------
    df : pandas.DataFrame
        Raw concatenated Kickstarter data.
    verbose : bool, default=True
        If True, print step info and shapes.

    Returns
    -------
    pandas.DataFrame
        Final cleaned/engineered dataset, ready for modeling.
    """
    print("Starting shape: ", df.shape)
    if verbose: print("\tApplying deduplication handling...")
    df = apply_deduplication_steps(df)

    if verbose: print("\tApplying missingness handling...")
    df = apply_missingness_handling(df)

    if verbose: print("\tApplying data filtering steps...")
    df = apply_data_filtering_steps(df)
    
    if verbose: print("\tApplying feature engineering...")
    df = apply_feature_engineering(df)
    
    if verbose: print("\tApplying additional preprocessing...")
    df = apply_additional_preprocessing(df)
    df = df.reset_index(drop = True)

    print("Finished preprocessing, new shape: ", df.shape)
    return df

def save_post_processing(df: pd.DataFrame) -> None:
    """
    Save the final processed dataframe to disk as parquet.
    """
    df.to_parquet(PROCESSED_DATA_PATH, index = False)

if __name__ == '__main__':
    print("LOADING DATA...")
    kick_clean = load_raw_data()
    kick_clean = apply_preprocessing(kick_clean)
    save_post_processing(kick_clean)
    print(f"SUCCESSFULLY SAVED TO {PROCESSED_DATA_PATH}")
