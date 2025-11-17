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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pathlib import Path

from src.utils import load_processed_data

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
    df: pd.DataFrame
) -> tuple[pd.DataFrame, LabelEncoder, StandardScaler]:
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
    return df

def preprocess_data(X_train, X_test, categorical_cols, numerical_cols):
    """
    Encode categorical features and scale numerical features for a train/test split.

    For each categorical column:
    - If the column has < 100 unique values, apply one-hot encoding.
    - Otherwise, apply frequency encoding, mapping categories to their
      relative frequency in the training set (unseen values in the test
      set are assigned a small "rare" frequency).

    Numerical columns are standardized using :class:`StandardScaler`
    fitted on the training data and then applied to both train and test.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    X_test : pandas.DataFrame
        Test feature matrix.
    categorical_cols : iterable of str
        Names of columns to treat as categorical.
    numerical_cols : iterable of str
        Names of columns to treat as numerical and scale.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Tuple of (X_train_processed, X_test_processed) with encoded
        categorical columns and scaled numerical columns.
    """
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

def perform_CV(
        data: pd.DataFrame, 
        model_class, 
        model_params, 
        verbose = False
    ) -> list[dict[str, float]]:
    """
    Run stratified K-fold cross-validation for a given model class.

    The function:
    - splits the data into K stratified folds
    - applies preprocessing (categorical encoding + numerical scaling)
      separately within each fold via :func:`preprocess_data`
    - trains the specified model on each training fold
    - evaluates on the corresponding validation fold using accuracy,
      precision, recall, and F1 score.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed dataset including an `is_successful` target column.
    model_class : callable
        Class (or factory) used to instantiate the model, e.g.
        :class:`RandomForestClassifier`, :class:`XGBClassifier`.
    model_params : dict
        Keyword arguments passed to `model_class` when constructing
        the model for each fold.
    verbose : bool, default False
        If True, print per-fold metric values.

    Returns
    -------
    list of dict
        A list of length K, where each element is a dict with keys
        ``"accuracy"``, ``"precision"``, ``"recall"``, and ``"f1"``.
    """
    # ====================================
    #               CONFIG
    # ====================================
    K_splits = 5
    random_state = 1337
    
    df = data.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns
    train = df.drop(columns=["is_successful"])
    target = df["is_successful"]

    # Ensure numerical columns are float64
    train[numerical_cols] = train[numerical_cols].astype("float64")

    # ====================================
    #        CROSS-VALIDATION LOOP
    # ====================================
    fold_scores = []
    sk_fold = StratifiedKFold(n_splits=K_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(sk_fold.split(train, target)):
        if verbose:
            print(f"Fold {i+1}")

        X_train, X_test = train.iloc[train_idx], train.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        # APPLY PREPROCESSING
        X_train, X_test = preprocess_data(X_train, X_test, categorical_cols, numerical_cols)

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

# ---------------------------------------------------------------
#                        NAIVE BASELINE
# ---------------------------------------------------------------
def model_naive_baseline(data: pd.DataFrame, verbose = False):
    """
    Compute and save a majority-class baseline for `is_successful`.

    The baseline predicts the most frequent class in the target for
    every observation and reports the resulting metrics.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset including a boolean `is_successful` target column.
    verbose : bool, default False
        If True, print baseline accuracy, precision, recall, and F1.

    Returns
    -------
    tuple of float
        (accuracy, precision, recall, f1) for the majority-class baseline.

    Side Effects
    ------------
    Writes a CSV file with one row containing the metrics to
    ``./results/naive_baseline_results.csv``.
    """

    target = data['is_successful']
    majority_class = target.value_counts().sort_values(ascending = False).keys()[0]
    predictions = np.full_like(target, majority_class)

    accuracy = (predictions == target).mean()
    precision = precision_score(target, predictions)
    recall = recall_score(target, predictions)
    f1 = f1_score(target, predictions)
    
    if verbose:
        print(f"\tMajority class baseline accuracy: {accuracy:.4f}")
        print(f"\tMajority class baseline precision: {precision:.4f}")
        print(f"\tMajority class baseline recall: {recall:.4f}")
        print(f"\tMajority class baseline f1: {f1:.4f}")

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

def evaluate_base_models(data: pd.DataFrame, verbose = False):
    """
    Train and evaluate untuned baseline models using cross-validation.

    Models evaluated:
    - Random Forest
    - XGBoost
    - Logistic Regression

    For each model, this function:
    - runs K-fold cross-validation via :func:`perform_CV`
    - aggregates mean accuracy, precision, recall, and F1 across folds
    - optionally prints the average metrics
    - compares models based on mean F1 score

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed dataset including an `is_successful` target column.
    verbose : bool, default False
        If True, print mean metrics for each model.

    Returns
    -------
    None

    Side Effects
    ------------
    Saves a CSV file with mean metrics per model to
    ``./results/untuned_model_results.csv`` and prints the best model
    according to F1 score.
    """

    def print_verbose(df, verbose):
        if verbose:
            for metric in ["accuracy", "precision", "recall", "f1"]:
                print(f"\t{metric}: {df[metric].mean():.4f}")

    # ====================================
    #           RANDOM FOREST
    # ====================================
    print("\tRANDOM FOREST...")
    rf_fold_scores = perform_CV(
        data = data,
        model_class = RandomForestClassifier,
        model_params = {},
        verbose = False
    )
    rf_results_df = pd.DataFrame(rf_fold_scores)
    print_verbose(rf_results_df, verbose)
    # ====================================
    #               XGBOOST
    # ====================================
    print("\tXGBOOST...")
    xgb_fold_scores = perform_CV(
        data = data,
        model_class = XGBClassifier,
        model_params = {},
        verbose = False
    )
    xgb_results_df = pd.DataFrame(xgb_fold_scores)
    print_verbose(xgb_results_df, verbose)
    # ====================================
    #         LOGISTIC REGRESSION
    # ====================================
    print("\tLOGISTIC REGRESSION...")
    lr_fold_scores = perform_CV(
        data = data,
        model_class = LogisticRegression,
        model_params = {'max_iter': 1000},
        verbose = False
    )
    lr_results_df = pd.DataFrame(lr_fold_scores)
    print_verbose(lr_results_df, verbose)

    # Aggregating results for comparison
    all_model_results = {
        "Random Forest": rf_results_df.mean(),
        "XGBoost": xgb_results_df.mean(),
        "Logistic Regression": lr_results_df.mean()
    }
    all_model_results_df = pd.DataFrame(all_model_results)

    print(f"Best model based on F1 score: {all_model_results_df.loc['f1'].idxmax()} with F1 = {all_model_results_df.loc['f1'].max():.4f}")
    path = Path("./results/untuned_model_results.csv")
    all_model_results_df.to_csv(path)
    print("\tSAVED UNTUNED MODEL RESULTS:", path)

def perform_CV_xgb_ensemble(
        data: pd.DataFrame,
        verbose = False
    ) -> list[dict[str, float]]:
    """
    Evaluate a per-category XGBoost ensemble using stratified K-fold CV.

    For each fold:
    - splits the data into train/test indices
    - for each `cat_parent_name` value:
        * loads tuned XGBoost hyperparameters for that category
          from ``./results/optimization_trials/xgb_categorical/``.
        * preprocesses the subset for that category using
          :func:`preprocess_data`
        * trains an XGBoost model and predicts on the category-specific
          portion of the test fold
    - combines all category predictions into a full test-fold prediction
    - computes accuracy, precision, recall, and F1 for the ensemble.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed dataset including `is_successful` and
        `cat_parent_name` columns.
    verbose : bool, default False
        If True, print metrics per fold.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame containing mean accuracy, precision,
        recall, and F1 across folds.

    Side Effects
    ------------
    Saves the mean metrics to
    ``./results/cat_xgb_ensemble_results.csv``.
    """

    # ====================================
    #               CONFIG
    # ====================================
    K_splits = 5
    random_state = 1337
    # Same config as before for consistency, same splits

    df = data.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns
    train = df.drop(columns=["is_successful"])
    target = df["is_successful"]

    train[numerical_cols] = train[numerical_cols].astype("float64")

    # ====================================
    #        CROSS-VALIDATION LOOP
    # ====================================
    fold_scores = []
    skf = StratifiedKFold(n_splits=K_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(skf.split(train, target)):
        if verbose:
            print(f"\nFold {i+1}")

        X_train, X_test = train.iloc[train_idx], train.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        # container for predictions on the full test fold
        y_pred_full = pd.Series(index=X_test.index, dtype=int)
        
        # ====================================
        #           TRAIN AND EVALUTATE (ONE PER CATEGORY PARENT)
        # ====================================
        for cat_parent_name in data['cat_parent_name'].unique():
            # get data for this category
            train_mask = (X_train['cat_parent_name'] == cat_parent_name)
            test_mask = (X_test['cat_parent_name'] == cat_parent_name)
            X_train_cat = X_train[train_mask]
            y_train_cat = y_train[train_mask]
            X_test_cat = X_test[test_mask]
            # y_test_cat = y_test[test_mask]

            # get best params:
            best_params_df = pd.read_csv(f"./results/optimization_trials/xgb_categorical/optuna_{cat_parent_name}_xgb_trials.csv")
            best_trial = best_params_df.loc[best_params_df['value'].idxmax()]
            best_params = {
                'n_estimators': int(best_trial['params_n_estimators']),
                'max_depth': int(best_trial['params_max_depth']),
                'learning_rate': float(best_trial['params_learning_rate']),
                'gamma': float(best_trial['params_gamma']),
                'eval_metric': 'logloss',
            }

            # preprocess per category (fit on cat-train, apply to cat-test)
            X_train_proc, X_test_proc = preprocess_data(
                X_train_cat,
                X_test_cat,
                categorical_cols=categorical_cols,
                numerical_cols=numerical_cols,
            )

            model = XGBClassifier(**best_params)
            model.fit(X_train_proc, y_train_cat)

            y_pred_cat = model.predict(X_test_proc)

            # write predictions back into the global y_pred_full
            y_pred_full.loc[test_mask] = y_pred_cat

        # Now evaluate ensemble on this fold
        y_true_fold = y_test
        y_pred_fold = y_pred_full

        fold_result = {
            "accuracy": accuracy_score(y_true_fold, y_pred_fold),
            "precision": precision_score(y_true_fold, y_pred_fold, zero_division=0),
            "recall": recall_score(y_true_fold, y_pred_fold, zero_division=0),
            "f1": f1_score(y_true_fold, y_pred_fold, zero_division=0),
        }
        fold_scores.append(fold_result)

        if verbose:
            print(
                f"\taccuracy={fold_result['accuracy']:.4f}, \n"
                f"\tprecision={fold_result['precision']:.4f}, \n"
                f"\trecall={fold_result['recall']:.4f}, \n"
                f"\tf1={fold_result['f1']:.4f}"
            )

    ensemble_scores_df = pd.DataFrame(pd.DataFrame(fold_scores).mean()).T
    path = Path("./results/cat_xgb_ensemble_results.csv")
    ensemble_scores_df.to_csv(path, index=False)
    print("\tSAVED ENSEMBLE RESULTS TO", path)

    return ensemble_scores_df

def perform_final_xgb_evaluation(data: pd.DataFrame):
    """
    Evaluate the globally tuned XGBoost model using cross-validation.

    This function:
    - loads the best global XGBoost hyperparameters from
      ``./results/optimization_trials/optuna_xgb_trials.csv``
    - runs stratified K-fold CV via :func:`perform_CV`
    - aggregates mean accuracy, precision, recall, and F1 over folds.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed dataset including an `is_successful` target column.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with mean accuracy, precision, recall,
        and F1 across folds.

    Side Effects
    ------------
    Saves the mean metrics to ``./results/xgb_results.csv``.
    """

    xgb_best_params_df = pd.read_csv("./results/optimization_trials/optuna_xgb_trials.csv")
    best_trial = xgb_best_params_df.loc[xgb_best_params_df['value'].idxmax()]
    best_xgb_params = {
        'n_estimators': int(best_trial['params_n_estimators']),
        'max_depth': int(best_trial['params_max_depth']),
        'learning_rate': float(best_trial['params_learning_rate']),
        'gamma': float(best_trial['params_gamma']),
        'eval_metric': 'logloss',
    }

    fold_scores = perform_CV(data, XGBClassifier, best_xgb_params, verbose=False)
    xgb_scores_df = pd.DataFrame(pd.DataFrame(fold_scores).mean()).T

    path = Path("./results/xgb_results.csv")
    xgb_scores_df.to_csv(path, index=False)
    print("\tSAVED FINAL XGB RESULTS TO", path)

    return xgb_scores_df

def evaluate_categorical_xgb_models(data: pd.DataFrame):
    """
    Evaluate tuned per-category XGBoost models on their own subsets.

    For each unique `cat_parent_name`:
    - loads tuned XGBoost hyperparameters for that category from
      ``./results/optimization_trials/xgb_categorical/``
    - restricts `data` to that category only
    - runs stratified K-fold CV via :func:`perform_CV`
    - computes mean accuracy, precision, recall, and F1.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed dataset including `is_successful` and
        `cat_parent_name` columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by `cat_parent_name`, with columns
        ``mean_accuracy``, ``mean_precision``, ``mean_recall``,
        and ``mean_f1``.

    Side Effects
    ------------
    Saves the per-category metrics to
    ``./results/cat_xgb_individual_results.csv``.
    """
    mean_scores = {}
    for cat_parent_name in data['cat_parent_name'].unique():
        # get best params:
        best_params_df = pd.read_csv(f"./results/optimization_trials/xgb_categorical/optuna_{cat_parent_name}_xgb_trials.csv")
        # Get best trial parameters:
        best_trial = best_params_df.loc[best_params_df['value'].idxmax()]
        best_xgb_params = {
            'n_estimators': int(best_trial['params_n_estimators']),
            'max_depth': int(best_trial['params_max_depth']),
            'learning_rate': float(best_trial['params_learning_rate']),
            'gamma': float(best_trial['params_gamma']),
            'eval_metric': 'logloss',
        }
        fold_scores = perform_CV(
            data=data[data['cat_parent_name'] == cat_parent_name],
            model_class=XGBClassifier,
            model_params=best_xgb_params,
            verbose=False,
        )
        mean_scores[cat_parent_name] = {
            "mean_accuracy": np.mean([fold["accuracy"] for fold in fold_scores]),
            "mean_precision": np.mean([fold["precision"] for fold in fold_scores]),
            "mean_recall": np.mean([fold["recall"] for fold in fold_scores]),
            "mean_f1": np.mean([fold["f1"] for fold in fold_scores]),
        }
    mean_scores_df = pd.DataFrame.from_dict(mean_scores, orient='index')
    path = Path("./results/cat_xgb_individual_results.csv")
    mean_scores_df.to_csv(path)
    print("\tSAVED CATEGORICAL XGB INDIVIDUAL CATEGORY RESULTS TO", path)
    return mean_scores_df

def evaluate_global_xgb_on_categorical_subsets(data: pd.DataFrame):
    """
    Evaluate the globally tuned XGBoost model on each category subset.

    This function:
    - loads the best global XGBoost hyperparameters from
      ``./results/optimization_trials/optuna_xgb_trials.csv``
    - for each unique `cat_parent_name`, restricts the data to that
      subset and runs :func:`perform_CV`
    - aggregates mean accuracy, precision, recall, and F1 for each subset.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed dataset including `is_successful` and
        `cat_parent_name` columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by `cat_parent_name`, with columns
        ``mean_accuracy``, ``mean_precision``, ``mean_recall``,
        and ``mean_f1``.

    Side Effects
    ------------
    Saves the per-category metrics to
    ``./results/xgb_individual_results.csv``.
    """
    mean_scores = {}

    optuna_xgb_trials = pd.read_csv("./results/optimization_trials/optuna_xgb_trials.csv")
    best_trial = optuna_xgb_trials.loc[optuna_xgb_trials['value'].idxmax()]
    best_xgb_params = {
        'n_estimators': int(best_trial['params_n_estimators']),
        'max_depth': int(best_trial['params_max_depth']),
        'learning_rate': float(best_trial['params_learning_rate']),
        'gamma': float(best_trial['params_gamma']),
        'eval_metric': 'logloss',
    }

    # Use perform_CV on categorical subsets using best params
    for cat_parent_name in data['cat_parent_name'].unique():
        # print(f"Evaluating best XGB model for category: {cat_parent_name}")
        fold_scores = perform_CV(
            data=data[data['cat_parent_name'] == cat_parent_name],
            model_class=XGBClassifier,
            model_params=best_xgb_params,
            verbose=False,
        )
        mean_scores[cat_parent_name] = {
            "mean_accuracy": np.mean([fold["accuracy"] for fold in fold_scores]),
            "mean_precision": np.mean([fold["precision"] for fold in fold_scores]),
            "mean_recall": np.mean([fold["recall"] for fold in fold_scores]),
            "mean_f1": np.mean([fold["f1"] for fold in fold_scores]),
        }
    mean_scores_df = pd.DataFrame.from_dict(mean_scores, orient='index')
    path = Path("./results/xgb_individual_results.csv")
    mean_scores_df.to_csv(path)
    print("\tSAVED GLOBAL XGB INDIVIDUAL CATEGORY RESULTS TO", path)

    return mean_scores_df

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

    print("EVALUATING BASELINE NAIVE MODEL...")
    _, _, _, _ = model_naive_baseline(kick_transformed, verbose=True)

    print("TRAINING AND EVALUATING MODELS:")
    evaluate_base_models(kick_transformed, verbose=True)

def evaluate_optimized():
    """
    Entry point for evaluating tuned XGBoost models and ensembles.

    Steps
    -----
    1. Load processed Kickstarter data via :func:`load_processed_data`.
    2. Apply final preprocessing via :func:`machine_ready_preprocessing`.
    3. Evaluate the globally tuned XGBoost model via
       :func:`perform_final_xgb_evaluation`.
    4. Evaluate the tuned per-category XGBoost ensemble via
       :func:`perform_CV_xgb_ensemble`.
    5. Evaluate the globally tuned XGBoost model on each category subset
       via :func:`evaluate_global_xgb_on_categorical_subsets`.
    6. Evaluate tuned per-category XGBoost models on their own subsets via
       :func:`evaluate_categorical_xgb_models`.

    Side Effects
    ------------
    Writes multiple CSV result files into the ``./results/`` directory
    and prints progress messages to stdout.
    """
    print("LOADING PROCESSED DATA...")
    kick_clean = load_processed_data()

    print("APPLYING FINAL MACHINE READY PREPROCESSING...")
    kick_transformed = machine_ready_preprocessing(kick_clean)

    print("EVALUATING OPTIMIZED MODELS:")
    print("EVALUATING FINAL TUNED XGB MODEL:")
    perform_final_xgb_evaluation(kick_transformed)

    print("EVALUATING TUNED CATEGORICAL ENSEMBLE MODEL:")
    perform_CV_xgb_ensemble(kick_transformed, verbose = False)

    print("EVALUTING FINAL TUNED XGB MODEL ON CATEGORICAL SUBSETS:")
    evaluate_global_xgb_on_categorical_subsets(kick_transformed)

    print("EVALUATING TUNED CATEGORICAL ENSEMBLE MODEL ON CATEGORICAL SUBSETS:")
    evaluate_categorical_xgb_models(kick_transformed)

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