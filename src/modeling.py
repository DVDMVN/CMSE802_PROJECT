"""
Final machine-learning pipeline:
- last-step preprocessing for modeling (encoding/scaling)
- stratified K-fold CV

Models:
- naive baseline (majority)
- RandomForest
- XGBoost
- Logistic Regression
- per-category specialist models (XGBoost ensemble)
- evaluation and CSV export of results

Run this file directly to train and evaluate models and
write accuracy results into the `results/` folder.

Author: Alex (Ze) Chen
Date: 2025-10-24
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pathlib import Path

from src.utils import load_processed_data

# ====================================
#               CONFIG
# ====================================
K_splits = 5
RANDOM_STATE = 1337

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

def handle_types(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=["number"]).columns
    df[numerical_cols] = df[numerical_cols].astype("float64")
    return df

def machine_ready_preprocessing(df: pd.DataFrame)-> pd.DataFrame:
    """
    Apply final preprocessing steps required before modeling.

    This function:
    - derives time-based features and drops raw timestamp columns
      via :func:`handle_datetime_features`
    - drops human-only columns and encodes the binary target
      via :func:`drop_final_irrelevant_columns_and_encode_target`

    Parameters
    ----------
    df : pandas.DataFrame
        Input Kickstarter data after earlier cleaning steps.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame including the binary `is_successful`
        target column and no raw timestamp or text/meta columns.
    """
    df = handle_datetime_features(df)
    df = drop_final_irrelevant_columns_and_encode_target(df)
    df = handle_types(df)
    return df

def preprocess_data(X_train, X_test, categorical_cols, numerical_cols):
    # Training set will fit the encoders / scalers, the test set will use the fitted encoders / scalers
    # ====================================
    #        CATEGORICAL ENCODING
    # ====================================
    # Encode categorical features
    # OneHotEncoding for features that have relatively low cardinality (< 100 unique values)
    # FrequencyEncoding for features that have relatively high cardinality (>= 100 unique values)
    for cat_col in categorical_cols:
        if X_train[cat_col].nunique() < 100: # OneHotEncoding for low cardinality
            oh_encoder = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )
            oh_encoder.fit(X_train[[cat_col]])

            # Transform and join back to dataframe
            X_train_oh = pd.DataFrame(
                oh_encoder.transform(X_train[[cat_col]]),
                columns=[f"{cat_col}__{cat}" for cat in oh_encoder.categories_[0]],
                index=X_train.index
            )
            X_test_oh = pd.DataFrame(
                oh_encoder.transform(X_test[[cat_col]]),
                columns=[f"{cat_col}__{cat}" for cat in oh_encoder.categories_[0]],
                index=X_test.index
            )

            X_train = X_train.drop(columns=[cat_col]).join(X_train_oh)
            X_test = X_test.drop(columns=[cat_col]).join(X_test_oh)
        else: # FrequencyEncoding for high cardinality
            freq_encoding = X_train[cat_col].value_counts(normalize=True)
            X_train.loc[:, cat_col] = X_train[cat_col].map(freq_encoding)
            X_test.loc[:, cat_col] = X_test[cat_col].map(freq_encoding).fillna(1 / (X_train.__len__() + X_test.__len__())) # Fill unseen with "rare" frequency
            X_train[cat_col] = X_train[cat_col].astype("float64")
            X_test[cat_col] = X_test[cat_col].astype("float64")
    
    # ====================================
    #         NUMERICAL SCALING
    # ====================================

    num_scaler = StandardScaler()
    num_scaler.fit(X_train[numerical_cols])
    X_train.loc[:, numerical_cols] = num_scaler.transform(X_train[numerical_cols])
    X_test.loc[:, numerical_cols] = num_scaler.transform(X_test[numerical_cols])

    return X_train, X_test

# ---------------------------------------------------------------
#            MODEL TRAINING & EVALUATION FUNCTIONS
# ---------------------------------------------------------------

def perform_CV(
        data: pd.DataFrame, 
        model_class, 
        model_params, 
        verbose = False
    ) -> list[dict[str, float]]:
    df = data.copy()
    train = df.drop(columns=["is_successful"])
    target = df["is_successful"]

    # ====================================
    #        CROSS-VALIDATION LOOP
    # ====================================
    fold_scores = []
    sk_fold = StratifiedKFold(n_splits=K_splits, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_idx, test_idx) in enumerate(sk_fold.split(train, target)):
        if verbose:
            print(f"Fold {i+1}")

        X_train, X_test = train.iloc[train_idx], train.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        # ====================================
        #         TRAIN AND EVALUTATE
        # ====================================
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        fold_result = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        fold_scores.append(fold_result)

        if verbose:
            print(
                f"\taccuracy={fold_result['accuracy']:.4f}, \n"
                f"\tprecision={fold_result['precision']:.4f}, \n"
                f"\trecall={fold_result['recall']:.4f}, \n"
                f"\tf1={fold_result['f1']:.4f}"
            )
    return fold_scores

def fit_model_on_full_train(
        train_data: pd.DataFrame,
        model_class,
        model_params,
        verbose = False
    ):
    X_train = train_data.drop(columns=["is_successful"])
    y_train = train_data["is_successful"]

    # TRAINING
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    if verbose:
        print(f"Trained {model.__class__.__name__} on full training data.")

    return model


def perform_test_evaluation(
        test_data: pd.DataFrame,
        model,
        verbose = True
    ) -> dict[str, float]:
    X_test = test_data.drop(columns = ['is_successful'])
    y_test = test_data['is_successful']

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    test_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    if verbose:
        print(
            f"{model.__class__.__name__} test evaluation results: \n"
            f"\taccuracy={accuracy:.4f}, \n"
            f"\tprecision={precision:.4f}, \n"
            f"\trecall={recall:.4f}, \n"
            f"\tf1={f1:.4f}"
        )
    return test_results

# ---------------------------------------------------------------
#                        NAIVE BASELINE
# ---------------------------------------------------------------
def model_naive_baseline(data: pd.DataFrame, verbose = False):
    target = data['is_successful']
    majority_class = target.value_counts().sort_values(ascending = False).keys()[0]
    predictions = np.full_like(target, majority_class)

    accuracy = (predictions == target).mean()
    precision = precision_score(target, predictions)
    recall = recall_score(target, predictions)
    f1 = f1_score(target, predictions)
    
    if verbose:
        print(f"\tMajority class baseline model accuracy: {accuracy:.4f}")
        print(f"\tMajority class baseline model precision: {precision:.4f}")
        print(f"\tMajority class baseline model recall: {recall:.4f}")
        print(f"\tMajority class baseline model f1: {f1:.4f}")

    naive_baseline_results = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1]
    })
    path = Path("./results/naive_baseline_results.csv")
    naive_baseline_results.to_csv(path, index = False)
    print("\tSAVED NAIVE BASELINE RESULTS:", path)
    return accuracy, precision, recall, f1

# ---------------------------------------------------------------
#                  BASELINE MODEL EVALUATION
# ---------------------------------------------------------------
def evaluate_base_models(train: pd.DataFrame, test: pd.DataFrame):
    # ====================================
    #           RANDOM FOREST
    # ====================================
    rf_model = fit_model_on_full_train(
        train_data = train,
        model_class = RandomForestClassifier,
        model_params = {}
    )
    rf_score = perform_test_evaluation(
        test_data = test,
        model = rf_model,
        verbose = False
    )

    # ====================================
    #               XGBOOST
    # ====================================
    xgb_model = fit_model_on_full_train(
        train_data = train,
        model_class = XGBClassifier,
        model_params = {}
    )
    xgb_score = perform_test_evaluation(
        test_data = test,
        model = xgb_model,
        verbose = False
    )

    # ====================================
    #         LOGISTIC REGRESSION
    # ====================================
    lr_model = fit_model_on_full_train(
        train_data = train,
        model_class = LogisticRegression,
        model_params = {"max_iter": 1000}
    )
    lr_score = perform_test_evaluation(
        test_data = test,
        model = lr_model,
        verbose = False
    )

    # Aggregating results for comparison
    all_model_results = [
        rf_score,
        xgb_score,
        lr_score
    ]
    
    all_model_results_df = pd.DataFrame(all_model_results, index = ["RandomForest", "XGBoost", "LogisticRegression"])
    all_model_results_df.to_csv("./results/untuned_model_results.csv")
    print("SAVED UNTUNED MODEL RESULTS: ./results/untuned_model_results.csv")

# ---------------------------------------------------------------
#                   AFTER OPTIMIZATION FUNCTIONS
# ---------------------------------------------------------------
def get_best_model_params():
    xgb_best_params_df = pd.read_csv("./results/optimization_trials/optuna_xgb_trials.csv")
    best_trial = xgb_best_params_df.loc[xgb_best_params_df['value'].idxmax()]
    best_xgb_params = {
        'n_estimators': int(best_trial['params_n_estimators']),
        'max_depth': int(best_trial['params_max_depth']),
        'learning_rate': float(best_trial['params_learning_rate']),
        'gamma': float(best_trial['params_gamma']),
        'eval_metric': 'logloss',
    }
    return best_xgb_params

def perform_test_eval_on_cat_slices(test_data: pd.DataFrame, model, categories: list[str]):
    X_test = test_data.drop(columns = ['is_successful'])
    y_test = test_data['is_successful']
    per_category_results = {}
    for category in categories:
        test_mask = (X_test[f'cat_parent_name__{category}'] == 1)
        X_test_cat = X_test[test_mask]
        y_pred_cat = model.predict(X_test_cat)

        accuracy = accuracy_score(y_test[test_mask], y_pred_cat)
        precision = precision_score(y_test[test_mask], y_pred_cat, zero_division=0)
        recall = recall_score(y_test[test_mask], y_pred_cat, zero_division=0)
        f1 = f1_score(y_test[test_mask], y_pred_cat, zero_division=0)
        per_category_results[category] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    return per_category_results

def retrain_best_category_models(train: pd.DataFrame, categories):
    trained_models = {}

    for category in categories:
        best_params_df = pd.read_csv(f"./results/optimization_trials/xgb_categorical/optuna_{category}_xgb_trials.csv")
        best_trial = best_params_df.loc[best_params_df['value'].idxmax()]
        best_params = {
            'n_estimators': int(best_trial['params_n_estimators']),
            'max_depth': int(best_trial['params_max_depth']),
            'learning_rate': float(best_trial['params_learning_rate']),
            'gamma': float(best_trial['params_gamma']),
            'eval_metric': 'logloss',
        }

        train_slice = train[train[f'cat_parent_name__{category}'] == 1]
        fit_model = fit_model_on_full_train(
            train_data = train_slice,
            model_class = XGBClassifier,
            model_params = best_params,
            verbose = False,
        )
        trained_models[category] = fit_model
    return trained_models

def perform_test_evaluation_ensemble(
        test_data: pd.DataFrame,
        trained_models: dict[str, XGBClassifier],
        verbose = True
    ) -> dict[str, float]:
    X_test = test_data.drop(columns = ['is_successful'])
    y_test = test_data['is_successful']

    per_category_results = {}
    y_pred_full = pd.Series(index=X_test.index, dtype=int)
    for cat_parent_name, model in trained_models.items():
        # get data for this category
        test_mask = (X_test[f'cat_parent_name__{cat_parent_name}'] == 1)
        X_test_cat = X_test[test_mask]
        y_pred_cat = model.predict(X_test_cat)

        # write predictions back into the global y_pred_full
        y_pred_full.loc[test_mask] = y_pred_cat

        # And also do per-category evaluations
        accuracy = accuracy_score(y_test[test_mask], y_pred_cat)
        precision = precision_score(y_test[test_mask], y_pred_cat, zero_division=0)
        recall = recall_score(y_test[test_mask], y_pred_cat, zero_division=0)
        f1 = f1_score(y_test[test_mask], y_pred_cat, zero_division=0)
        per_category_results[cat_parent_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    accuracy = accuracy_score(y_test, y_pred_full)
    precision = precision_score(y_test, y_pred_full, zero_division=0)
    recall = recall_score(y_test, y_pred_full, zero_division=0)
    f1 = f1_score(y_test, y_pred_full, zero_division=0)
    test_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    if verbose:
        print(
            f"XGB ensemble overall test evaluation results: \n"
            f"\taccuracy={accuracy:.4f}, \n"
            f"\tprecision={precision:.4f}, \n"
            f"\trecall={recall:.4f}, \n"
            f"\tf1={f1:.4f}"
        )
    return test_results, per_category_results

# ===============================================================
#                             MAIN
# ===============================================================

def main_default():
    """
    Default entry point for training and evaluating untuned models.

    Steps
    -----
    1. Load processed Kickstarter data via :func:`load_processed_data`.
    2. Apply final preprocessing via :func:`machine_ready_preprocessing`.
    3. Compute and save the majority-class baseline via
       :func:`model_naive_baseline`.
    4. Train and evaluate untuned baseline models via
       :func:`evaluate_base_models`.

    Side Effects
    ------------
    Writes multiple result CSVs under the ``./results/`` directory
    and prints progress and evaluation summaries to stdout.
    """
    print("LOADING PROCESSED DATA...")
    kick_clean = load_processed_data()

    print("APPLYING FINAL MACHINE READY PREPROCESSING...")
    kick_transformed = machine_ready_preprocessing(kick_clean)
    kick_train, kick_test = train_test_split(kick_transformed, test_size = 0.2, random_state = RANDOM_STATE)

    categorical_cols = kick_transformed.select_dtypes(include=["object"]).columns
    numerical_cols = kick_transformed.select_dtypes(include=["number"]).columns
    kick_train, kick_test = preprocess_data(kick_train, kick_test, categorical_cols, numerical_cols)

    print("EVALUATING BASELINE NAIVE MODEL...")
    _, _, _, _ = model_naive_baseline(kick_test, verbose=True)

    print("TRAINING AND EVALUATING UNTUNED MODELS:")
    evaluate_base_models(kick_train, kick_test)

def evaluate_optimized():
    """
    Entry point for evaluating tuned XGBoost models and ensembles.

    Steps
    -----
    1. Load processed Kickstarter data via :func:`load_processed_data`.
    2. Apply final preprocessing via :func:`machine_ready_preprocessing`.
    3. Evaluate the globally tuned XGBoost model
    4. Evaluate the globally tuned XGBoost model on each category subset
    5. Evaluate the tuned per-category XGBoost ensemble
    6. Evaluate tuned per-category XGBoost models on their own subsets

    Side Effects
    ------------
    Writes multiple CSV result files into the ``./results/`` directory
    and prints progress messages to stdout.
    """
    print("LOADING PROCESSED DATA...")
    kick_clean = load_processed_data()

    print("APPLYING FINAL MACHINE READY PREPROCESSING...")
    kick_transformed = machine_ready_preprocessing(kick_clean)
    kick_train, kick_test = train_test_split(kick_transformed, test_size = 0.2, random_state = RANDOM_STATE)

    categorical_cols = kick_transformed.select_dtypes(include=["object"]).columns
    numerical_cols = kick_transformed.select_dtypes(include=["number"]).columns
    kick_train, kick_test = preprocess_data(kick_train, kick_test, categorical_cols, numerical_cols)

    print("EVALUATING FINAL TUNED XGB MODEL:")
    best_params = get_best_model_params()
    best_model = fit_model_on_full_train(
        train_data = kick_train,
        model_class = XGBClassifier,
        model_params = best_params,
        verbose = False
    )
    xgb_score = perform_test_evaluation(
        test_data = kick_test,
        model = best_model,
        verbose = False
    )
    xgb_scores_df = pd.DataFrame([xgb_score], index = ['Tuned XGBoost'])
    xgb_scores_df.to_csv("./results/xgb_results.csv")
    print("SAVED XGB RESULTS TO ./results/xgb_results.csv")

    xgb_score_per_cat = perform_test_eval_on_cat_slices(kick_test, best_model, kick_transformed['cat_parent_name'].unique())
    xgb_score_per_cat_df = pd.DataFrame.from_dict(xgb_score_per_cat, orient='index')
    xgb_score_per_cat_df.to_csv("./results/xgb_category_results.csv")
    print("SAVED XGB PER-CATEGORY RESULTS TO ./results/xgb_category_results.csv")

    print("EVALUATING TUNED CATEGORICAL ENSEMBLE MODEL:")
    cat_xgb_models = retrain_best_category_models(kick_train, kick_transformed['cat_parent_name'].unique())
    cat_xgb_ensemble_test_results, cat_xgb_ensemble_per_category_results = perform_test_evaluation_ensemble(
        test_data = kick_test,
        trained_models = cat_xgb_models,
        verbose = False
    )
    cat_xgb_ensemble_test_results_df = pd.DataFrame([cat_xgb_ensemble_test_results], index = ['Categorical Tuned XGBoost Ensemble'])
    cat_xgb_ensemble_test_results_df.to_csv("./results/cat_xgb_ensemble_results.csv")
    print("SAVED CATEGORICAL XGB ENSEMBLE RESULTS TO ./results/cat_xgb_ensemble_results.csv")

    cat_xgb_ensemble_per_category_results_df = pd.DataFrame.from_dict(cat_xgb_ensemble_per_category_results, orient='index')
    cat_xgb_ensemble_per_category_results_df.to_csv("./results/cat_xgb_category_results.csv")
    print("SAVED CATEGORICAL XGB ENSEMBLE PER-CATEGORY RESULTS TO ./results/cat_xgb_category_results.csv")

if __name__ == '__main__':
    import sys
    # Map command names to functions
    commands = {
        "default": main_default,
        "evaluate_optimized": evaluate_optimized,
    }

    # First CLI argument after the script name, or "default" if none
    cmd = sys.argv[1] if len(sys.argv) > 1 else "default"

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print("Available commands:", ", ".join(commands.keys()))
        sys.exit(1)

    # Call the chosen function
    commands[cmd]()