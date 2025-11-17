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
from src.modeling import machine_ready_preprocessing, perform_CV

import optuna

# ---------------------------------------------------------------
#          OPTUNA HYPERPARAMETER OPTIMIZATION ROUTINES
# ---------------------------------------------------------------

def perform_optuna_rf_optimization(data: pd.DataFrame):
    def objective_rf(trial: optuna.Trial) -> float:
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

        fold_scores = perform_CV(
            data=data,
            model_class=RandomForestClassifier,
            model_params=model_params,
            verbose=False,
        )

        mean_f1 = float(np.mean([fold["f1"] for fold in fold_scores]))
        return mean_f1
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study_rf = optuna.create_study(direction="maximize", study_name="rf_f1")
    study_rf.optimize(objective_rf, n_trials=50)

    print("RF best F1:", study_rf.best_value)
    print("RF best params:", study_rf.best_params)

    df_rf_cat = study_rf.trials_dataframe()
    df_rf_cat.to_csv("./results/optimization_trials/optuna_rf_trials.csv", index=False)

def perform_optuna_xgb_optimization(data: pd.DataFrame):
    def objective_xgb(trial: optuna.Trial) -> float:
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "eval_metric": "logloss",
        }

        fold_scores = perform_CV(
            data=data,
            model_class=XGBClassifier,
            model_params=model_params,
            verbose=False,
        )

        mean_f1 = float(np.mean([fold["f1"] for fold in fold_scores]))
        return mean_f1

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study_xgb = optuna.create_study(direction="maximize", study_name="xgb_f1")
    study_xgb.optimize(objective_xgb, n_trials=50)

    print("XGB best F1:", study_xgb.best_value)
    print("XGB best params:", study_xgb.best_params)

    df_xgb_cat = study_xgb.trials_dataframe()
    df_xgb_cat.to_csv("./results/optimization_trials/optuna_xgb_trials.csv", index=False)

def perform_categorical_optuna_xgb_optimization(data: pd.DataFrame):
    optuna.logging.set_verbosity(optuna.logging.WARN)

    best_xgb_cat_results = []
    for cat_parent_name in data['cat_parent_name'].unique():
        print(f"Optimizing for category: {cat_parent_name}")

        def objective_cat(trial: optuna.Trial) -> float:
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "eval_metric": "logloss",
            }

            fold_scores = perform_CV(
                data=data[data['cat_parent_name'] == cat_parent_name],
                model_class=XGBClassifier,
                model_params=model_params,
                verbose=False,
            )

            mean_f1 = float(np.mean([fold["f1"] for fold in fold_scores]))
            return mean_f1

        study_cat = optuna.create_study(direction="maximize", study_name=f"xgb_f1_{cat_parent_name}")
        study_cat.optimize(objective_cat, n_trials=100)

        print(f"XGB best F1 for {cat_parent_name}:", study_cat.best_value)
        print(f"XGB best params for {cat_parent_name}:", study_cat.best_params)
        best_xgb_cat_results.append({
            "cat_parent_name": cat_parent_name,
            "best_f1": study_cat.best_value,
            "best_params": study_cat.best_params
        })
        
        # XGBoost study
        df_xgb_cat = study_cat.trials_dataframe()
        df_xgb_cat.to_csv(f"./results/optimization_trials/xgb_categorical/optuna_{cat_parent_name}_xgb_trials.csv", index=False)

