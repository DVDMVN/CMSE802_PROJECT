"""
Unit tests for src.modeling.

Tests include:
- Machine-ready preprocessing transformations
- Encoder/scaler reuse logic
- Baseline model accuracy computation

Author: Alex (Ze) Chen
Date: 2025-10-24
"""

import pytest
import pandas as pd
import numpy as np
from src.modeling import (
    machine_ready_preprocessing,
    model_naive_baseline,
)

def _toy_df():
    """
    Create a minimal DataFrame that mimics post-preprocessing input.
    """
    # Minimal frame that looks like post-preprocessing output
    return pd.DataFrame({
        "index": [0, 1],
        "name": ["Cool Thing", "Another Thing"],
        "blurb": ["Please back me", "Hi"],
        "cat_parent_name": ["Games", "Games"],
        "goal": [1000.0, 2000.0],
        "state": ["successful", "failed"],
        "created_at": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "deadline": pd.to_datetime(["2020-01-10", "2020-01-12"]),
        "launched_at": pd.to_datetime(["2020-01-01", "2020-01-02"]),
    })



def test_machine_ready_preprocessing_fits_and_transforms_training():
    """
    Verify correct fitting and transformation during training mode.

    Ensures:
    - new encoders/scaler are fitted
    - datetimes dropped, target added
    - numeric scaling and encoding applied
    """
    df = _toy_df()

    transformed, cat_encoders, num_scaler = machine_ready_preprocessing(
        df,
        testing=False,
        cat_encoders={},
        num_scaler=None
    )

    # Target should exist
    assert "is_successful" in transformed.columns
    # Original 'state' should be dropped
    assert "state" not in transformed.columns

    # Datetime columns should be gone
    for col in ["created_at", "deadline", "launched_at"]:
        assert col not in transformed.columns

    # Encoded categorical should now be numeric dtype
    assert np.issubdtype(transformed["cat_parent_name"].dtype, np.number)

    # Scaled numeric column 'goal' should be mean ~0 after scaling
    # (allow rough tolerance for tiny sample)
    assert abs(transformed["goal"].mean()) < 1e-6

    # Age feature should exist
    assert "age_days" in transformed.columns


def test_machine_ready_preprocessing_testing_mode_reuses_encoders():
    """
    Verify reuse of encoders/scaler in testing=True mode.
    """
    df = _toy_df()

    # First pass: fit
    transformed_train, cat_encoders, num_scaler = machine_ready_preprocessing(
        df,
        testing=False,
        cat_encoders={},
        num_scaler=None
    )

    # Second pass: reuse (testing=True)
    transformed_test, cat_encoders2, num_scaler2 = machine_ready_preprocessing(
        df,
        testing=True,
        cat_encoders=cat_encoders,
        num_scaler=num_scaler
    )

    # Should not refit encoders / scaler, so objects should be identical
    assert cat_encoders2 is cat_encoders
    assert num_scaler2 is num_scaler

    # Shapes should match
    assert set(transformed_train.columns) == set(transformed_test.columns)


def test_machine_ready_preprocessing_testing_mode_raises_without_fitted_objects():
    """
    Confirm that missing fitted encoders/scaler triggers an exception in testing mode.
    """
    df = _toy_df()
    # Calling with testing=True but without encoders/scaler should break
    with pytest.raises(Exception):
        machine_ready_preprocessing(
            df,
            testing=True,
            cat_encoders={},
            num_scaler=None
        )


def test_model_naive_baseline_accuracy():
    """
    Test majority-class baseline accuracy calculation.
    """
    X_train = pd.DataFrame({"dummy": [0, 1, 2, 3]})
    y_train = pd.Series([True, True, True, False])
    X_test = pd.DataFrame({"dummy": [10, 11, 12]})
    y_test = pd.Series([True, False, True])

    acc = model_naive_baseline(X_train, X_test, y_train, y_test)

    # majority in y_train is True (3 vs 1)
    # predictions = [True, True, True]
    # y_test       [True, False, True]
    # matches in positions 0 and 2 -> 2/3 == 0.666...
    assert pytest.approx(acc, rel=1e-6) == 2/3
