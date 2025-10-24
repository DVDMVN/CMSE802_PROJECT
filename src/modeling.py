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
    X = df.drop(columns = ['is_successful'])
    y = df['is_successful']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------
#                       MODELING FUNCTIONS
# ---------------------------------------------------------------


### ----- ----- ----- ----- BASELINE ----- ----- ----- -----

def model_naive_baseline(X_train, X_test, y_train, y_test):
    majority_class = y_train.value_counts().sort_values(ascending = False).keys()[0]
    predictions = np.full_like(y_test, majority_class)

    accuracy = (predictions == y_test).mean()

    print(f"Majority class baseline model accuracy: {accuracy * 100:.2f}%")
    return accuracy

### ----- ----- ----- ----- GENERAL RF MODEL ----- ----- ----- -----

def model_general_rf(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Random Forest model accuracy: {accuracy * 100:.2f}%")
    return accuracy

### ----- ----- ----- ----- CATEGORY SPECIFIC RF MODEL ----- ----- ----- -----

def model_category_specific_rf(X_train, X_test, y_train, y_test):
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
    