"""
Unit tests for src.preprocessing.

Covers:
- Deduplication logic
- Missing-value imputations
- Leakage column removal
- Feature engineering transformations

Author: Alex (Ze) Chen
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    apply_deduplication_steps,
    apply_missingness_handling,
    apply_data_filtering_steps,
    apply_feature_engineering,
)


def test_apply_deduplication_steps_keeps_highest_backers():
    """
    Test that duplicate project IDs keep the entry with the highest backers_count.
    """
    df = pd.DataFrame([
        {"id": 1, "backers_count": 10, "other_col": "x"},
        {"id": 1, "backers_count": 50, "other_col": "y"},  # should keep this one
        {"id": 2, "backers_count": 5,  "other_col": "z"},
    ])

    out = apply_deduplication_steps(df)

    # should only have 2 rows left, one per id
    assert out["id"].nunique() == 2
    # check we kept the 50 row for id=1
    row_id1 = out[out["id"] == 1].iloc[0]
    assert row_id1["backers_count"] == 50
    assert row_id1["other_col"] == "y"


def test_apply_missingness_handling_imputes_blurb_and_money():
    """
    Test imputation of textual and monetary fields and derived flags.
    """
    df = pd.DataFrame({
        "blurb": [None, "hello"],
        "name": ["Project A", "Project B"],
        "usd_type": [None, "domestic"],
        "location": [
            '{"name": "Detroit", "state": "MI", "type": "Town"}',
            '{"name": "Detroit", "state": "MI", "type": "Town"}',
        ],
        "country": ["US", "US"],
        "pledged": [100.0, 200.0],
        "static_usd_rate": [1.2, 1.2],
        "usd_pledged": [None, 240.0],
        "usd_exchange_rate": [np.nan, 1.3],
        "currency": ["USD", "USD"],
        "converted_pledged_amount": [None, 260.0],
        "video": [None, "some_video_blob"],
        "is_in_post_campaign_pledging_phase": [True, False],
    })

    out = apply_missingness_handling(df)

    # blurb imputed from name
    assert out.loc[0, "blurb"] == "Project A"

    # has_video exists and 'video' dropped
    assert "video" not in out.columns
    assert "has_video" in out.columns
    assert out["has_video"].tolist() == [False, True]

    # usd_pledged imputed = pledged * static_usd_rate
    assert pytest.approx(out.loc[0, "usd_pledged"], rel=1e-6) == 100.0 * 1.2

    # usd_exchange_rate imputed from per-currency mean (1.3)
    assert pytest.approx(out.loc[0, "usd_exchange_rate"], rel=1e-6) == 1.3

    # converted_pledged_amount imputed = pledged * usd_exchange_rate
    expected0 = 100.0 * 1.3
    assert pytest.approx(out.loc[0, "converted_pledged_amount"], rel=1e-6) == expected0

    # dropped is_in_post_campaign_pledging_phase
    assert "is_in_post_campaign_pledging_phase" not in out.columns


def test_apply_missingness_handling_raises_if_required_cols_absent():
    """
    Test that KeyError is raised when required monetary columns are missing.
    """
    # missing 'pledged', should KeyError inside impute_money_features
    df_bad = pd.DataFrame({"blurb": ["x"]})
    with pytest.raises(KeyError):
        apply_missingness_handling(df_bad)


def test_apply_data_filtering_steps_removes_leakage_and_filters_states():
    """
    Test that leakage columns are removed and only final outcome states remain.
    """
    df = pd.DataFrame({
        # final outcome states
        "state": ["successful", "failed", "live", "canceled"],

        # leakage columns that should get dropped
        "backers_count": [10, 20, 99, 99],
        "pledged": [100, 200, 999, 999],
        "usd_pledged": [100, 200, 999, 999],
        "converted_pledged_amount": [110, 210, 999, 999],
        "percent_funded": [1.0, 2.0, 99, 99],
        "usd_exchange_rate": [1.2, 1.3, 9, 9],
        "state_changed_at": [1, 2, 3, 4],
        "spotlight": [True, False, True, False],
        "country_displayable_name": ["US", "US", "US", "US"],

        # columns that should survive
        "other_feature": [5, 6, 7, 8],

        # columns that drop_irrelevant_columns() will try to drop
        "id": [111, 222, 333, 444],
        "slug": ["a", "b", "c", "d"],
        "source_url": ["u", "v", "w", "t"],
        "urls": ["u1", "u2", "u3", "u4"],
        "currency_symbol": ["$", "$", "$", "$"],
        "currency_trailing_code": [True, True, True, True],
        "current_currency": ["USD", "USD", "USD", "USD"],
        "is_liked": [False, False, False, False],
        "is_disliked": [False, False, False, False],
        "is_starrable": [False, False, False, False],

        # columns that drop_photo / drop_creator / drop_profile will try to drop
        "photo": ["x", "y", "z", "w"],
        "creator": ["c1", "c2", "c3", "c4"],
        "profile": ["p1", "p2", "p3", "p4"],
    })

    out = apply_data_filtering_steps(df)

    # Only successful/failed survive
    assert set(out["state"].unique()) <= {"successful", "failed"}

    # Forbidden columns must be completely gone after filtering
    forbidden = {
        "backers_count",
        "pledged",
        "usd_pledged",
        "converted_pledged_amount",
        "percent_funded",
        "usd_exchange_rate",
        "state_changed_at",
        "spotlight",
        "photo",
        "creator",
        "profile",
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
        "id",  # <- also should be dropped
    }
    assert not (forbidden & set(out.columns)), "Leakage columns still present"

    # But legit features should still exist
    assert "other_feature" in out.columns



def test_apply_feature_engineering_creates_expected_columns_and_duration():
    """
    Test that category JSON expansion, text stats, and duration are computed.
    """
    df = pd.DataFrame({
        "category": [
            '{"name":"Comics","position":1,"parent_name":"Art"}',
            '{"name":"Games","position":2,"parent_name":"Entertainment"}',
        ],
        "blurb": ["Cool project", ""],
        "name": ["My Project", "B"],
        "created_at": [1_600_000_000, 1_600_086_400],   # epoch seconds
        "deadline": [1_600_086_400, 1_600_172_800],     # +1 day, +2 days
        "launched_at": [1_600_000_000, 1_600_086_400],  # match created_at
    })

    out = apply_feature_engineering(df)

    # category column replaced with flat columns
    assert "category" not in out.columns
    assert {"cat_name", "cat_position", "cat_parent_name"} <= set(out.columns)

    # text-based features exist
    assert {"blurb_len", "name_len", "blurb_avg_word_len", "name_avg_word_len"} <= set(out.columns)

    # timestamps converted
    assert pd.api.types.is_datetime64_ns_dtype(out["created_at"])

    # duration is days between deadline and launched_at
    assert out.loc[0, "duration"] == 1  # 1 day
    assert out.loc[1, "duration"] == 1  # (172800 - 86400)/86400 = 1 day
