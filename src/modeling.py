"""
Final machine-learning pipeline:
- last-step preprocessing for modeling (encoding/scaling)
- train/test split
- baseline and RandomForest models
- per-category specialist models
- evaluation and CSV export of results

Run this file directly to train and evaluate models and
write accuracy results into the `results/` folder.

Author: Alex (Ze) Chen
Date: 2025-10-24
"""

import numpy as np
import pandas as pd
from src.utils import load_processed_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ---------------------------------------------------------------
#              MACHINE READY PREPROCESSING FUNCTIONS
# ---------------------------------------------------------------

def drop_final_irrelevant_columns_and_encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove human-only columns and create binary target is_successful.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed Kickstarter data with 'state' column.

    Returns
    -------
    pandas.DataFrame
        Copy of df with text/meta columns dropped and new boolean
        column 'is_successful'. Original 'state' column is removed.
    """
        
    # These are useful for lookup / human insight, but not for machine learning
    df = df.copy()
    irrelevant_columns = [
        "index",
        "name",
        "blurb",
    ]
    df = df.drop(columns=irrelevant_columns)

    # Encode target
    df["is_successful"] = df["state"] == "successful"
    df = df.drop(columns=["state"])
    return df

def handle_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive time-based features and drop raw timestamps.

    Currently:
    - age_days : days since campaign launch

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing 'created_at', 'deadline', 'launched_at'
        as datetime columns.

    Returns
    -------
    pandas.DataFrame
        Copy with derived features added and original timestamp
        columns removed.
    """
    datetime_columns = [
        "created_at",
        "deadline",
        "launched_at"
    ]

    df["age_days"] = (pd.Timestamp.now() - df["launched_at"]).dt.days

    df = df.drop(columns = datetime_columns)
    return df

def machine_ready_preprocessing(
    df: pd.DataFrame, testing=False, cat_encoders = {}, num_scaler=None
) -> tuple[pd.DataFrame, LabelEncoder, StandardScaler]:
    """
    Prepare features for modeling:
    - derive time features / drop timestamps
    - drop unused columns and encode target
    - label-encode categoricals
    - standard-scale numerics

    Parameters
    ----------
    df : pandas.DataFrame
        Output of earlier preprocessing (categoricals still strings,
        numericals unscaled).
    testing : bool, default=False
        If False, fit new encoders/scaler on df.
        If True, assume encoders/scaler are provided and reuse them.
    cat_encoders : dict, default={}
        Maps column name -> fitted LabelEncoder.
        Filled on training; reused on testing.
    num_scaler : StandardScaler or None
        Fitted scaler for numeric columns.

    Returns
    -------
    df_out : pandas.DataFrame
        Encoded and scaled features (including 'is_successful').
    cat_encoders : dict
        Updated mapping of categorical col -> LabelEncoder.
    num_scaler : StandardScaler
        Fitted scaler for numeric columns.
    """
    df = handle_datetime_features(df)
    df = drop_final_irrelevant_columns_and_encode_target(df)
    # Training set will fit the encoders / scalers, the test set will use the fitted encoders / scalers
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns
    if not testing:
        # Encode categorical features
        for cat_col in categorical_cols:
            cat_encoder = LabelEncoder()
            cat_encoder.fit(df[cat_col].astype(str))
            cat_encoders[cat_col] = cat_encoder

        # Standardize numerical features
        num_scaler = StandardScaler()
        num_scaler.fit(df[numerical_cols])

    
    for cat_col in categorical_cols:
        df[cat_col] = cat_encoders[cat_col].transform(df[cat_col].astype(str))

    df[numerical_cols] = num_scaler.transform(df[numerical_cols])
    return df, cat_encoders, num_scaler


# ---------------------------------------------------------------
#                          TT SPLIT
# ---------------------------------------------------------------


def perform_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train/test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Fully transformed dataset containing 'is_successful'.

    Returns
    -------
    X_train, X_test : pandas.DataFrame
        Feature matrices with 'is_successful' removed.
    y_train, y_test : pandas.Series
        Target labels (True=successful, False=failed).
    """
    X = df.drop(columns = ['is_successful'])
    y = df['is_successful']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------
#                       MODELING FUNCTIONS
# ---------------------------------------------------------------


### ----- ----- ----- ----- BASELINE ----- ----- ----- -----

def model_naive_baseline(X_train, X_test, y_train, y_test):
    """
    Majority-class baseline.

    Predict the most common class in y_train for all test rows.

    Returns
    -------
    float
        Accuracy on the test set.
    """
    majority_class = y_train.value_counts().sort_values(ascending = False).keys()[0]
    predictions = np.full_like(y_test, majority_class)

    accuracy = (predictions == y_test).mean()

    print(f"Majority class baseline model accuracy: {accuracy * 100:.2f}%")
    return accuracy

### ----- ----- ----- ----- GENERAL RF MODEL ----- ----- ----- -----

def model_general_rf(X_train, X_test, y_train, y_test):
    """
    Train a single RandomForest model on all categories at once.

    Returns
    -------
    float
        Accuracy on the test set.
    """
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Random Forest model accuracy: {accuracy * 100:.2f}%")
    return accuracy

### ----- ----- ----- ----- CATEGORY SPECIFIC RF MODEL ----- ----- ----- -----

def model_category_specific_rf(X_train, X_test, y_train, y_test):
    """
    Train one RandomForest per parent category.

    For each unique encoded value of 'cat_parent_name':
    - train a model only on that subset
    - evaluate only on that subset in X_test
    - report accuracy

    Returns
    -------
    rf_models : dict[int, RandomForestClassifier]
        Mapping category_value -> fitted model.
    accuracies : dict[int, float]
        Mapping category_value -> accuracy for that category.
    """
    rf_models = {} # trained model per category
    accuracies = {} # model accuracy per category

    cat_name_encoder = cat_encoders["cat_parent_name"]
    for cat_val, X_train_group in X_train.groupby("cat_parent_name"):
        # find matching rows in train and test for this category
        train_idx = X_train_group.index
        test_idx  = X_test[X_test["cat_parent_name"] == cat_val].index

        # pull the feature matrices without the category column itself
        X_train_cat = X_train.loc[train_idx].drop(columns=["cat_parent_name"])
        X_test_cat  = X_test.loc[test_idx].drop(columns=["cat_parent_name"])

        # pull the targets for just those rows
        y_train_cat = y_train.loc[train_idx]
        y_test_cat  = y_test.loc[test_idx]

        # skip if we don't have test data for this category
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        # train model for this category
        rf = RandomForestClassifier()
        rf.fit(X_train_cat, y_train_cat)

        # evaluate and store
        y_pred = rf.predict(X_test_cat)
        accuracy = (y_pred == y_test_cat).mean()

        rf_models[cat_val] = rf
        accuracies[cat_val] = accuracy

        # pulling the name off the encoder
        cat_parent_name = cat_name_encoder.inverse_transform([cat_val])[0]
        print(f"\t{cat_parent_name}: accuracy = {accuracy * 100:.2f}%")

    return rf_models, accuracies

def weighted_accuracy_of_category_specific_rf(rf_models, X_test, y_test):
    """
    Compute overall accuracy across all category-specific models.

    For each category model:
    - collect its predictions on its own category rows
    - concatenate all predictions and truths
    - take global accuracy

    Parameters
    ----------
    rf_models : dict
        Output of model_category_specific_rf.
    X_test : pandas.DataFrame
        Test features (contains 'cat_parent_name').
    y_test : pandas.Series
        Test labels.

    Returns
    -------
    float
        Weighted/overall accuracy across all categories.
    """
    all_preds = []
    all_trues = []

    for cat_val, rf in rf_models.items():
        test_idx = X_test[X_test["cat_parent_name"] == cat_val].index
        X_test_cat = X_test.loc[test_idx].drop(columns=["cat_parent_name"])
        y_test_cat = y_test.loc[test_idx]

        y_pred_cat = rf.predict(X_test_cat)

        all_preds.append(y_pred_cat)
        all_trues.append(y_test_cat)

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    weighted_acc = (all_preds == all_trues).mean()
    print(f"Overall (weighted) accuracy: {weighted_acc * 100:.2f}%")
    return weighted_acc

# ===============================================================
#                             MAIN
# ===============================================================
if __name__ == '__main__':
    print("LOAINDG PROCESSED DATA...")
    kick_clean = load_processed_data()

    print("APPLYING FINAL MACHINE READY PREPROCESSING...")
    kick_transformed, cat_encoders, num_scaler = machine_ready_preprocessing(kick_clean)

    X_train, X_test, y_train, y_test = perform_train_test_split(kick_transformed)
    
    print("TRAINING + EVALUATING MODELS:")
    print("\tNAIVE BASELINE...")
    naive_baseline_acc = model_naive_baseline(X_train, X_test, y_train, y_test)
    print("\tRANDOM FOREST...")
    general_rf_acc = model_general_rf(X_train, X_test, y_train, y_test)
    print("\tRANDOM FOREST (CATEGORY SPECIFIC)...")
    rf_models, accuracies = model_category_specific_rf(X_train, X_test, y_train, y_test)
    cat_weighted_rf_model_accuracy = weighted_accuracy_of_category_specific_rf(rf_models, X_test, y_test)
    print("FINISHED TRAIN TEST")

    # Save results to csv
    overall_df = pd.DataFrame([
        {"model": "Majority Class Baseline", "accuracy": naive_baseline_acc},
        {"model": "Random Forest (General)", "accuracy": general_rf_acc},
        {"model": "Random Forest (Category specific, weighted overall)",
         "accuracy": cat_weighted_rf_model_accuracy},
    ])
    
    cat_name_encoder = cat_encoders["cat_parent_name"]
    per_cat_df = pd.DataFrame([
        {"category": cat_name_encoder.inverse_transform([cat])[0], "accuracy": acc}
        for cat, acc in accuracies.items()
    ]).sort_values("category").reset_index(drop=True)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_df.to_csv(out_dir / "overall_results.csv", index=False)
    per_cat_df.to_csv(out_dir / "per_category_accuracy.csv", index=False)

    print("SAVED RESULTS:")
    print("\t", out_dir / "overall_results.csv")
    print("\t", out_dir / "per_category_accuracy.csv")
    