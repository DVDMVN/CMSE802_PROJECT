"""
Unit tests for src.utils.

These tests cover JSON parsing helper functionality to ensure
robust handling of valid, empty, and malformed JSON strings.

Author: Alex (Ze) Chen
Date: 2025-10-24
"""

import pytest
from src.utils import parse_json_feature
import json


def test_parse_json_feature_valid():
    """
    Test parsing of a valid JSON string.

    Ensures correct conversion to Python dict with expected keys/values.
    """
    raw = '{"a": 1, "b": "cat"}'
    out = parse_json_feature(raw)
    assert out == {"a": 1, "b": "cat"}
    assert isinstance(out, dict)


def test_parse_json_feature_empty_object():
    """
    Test parsing of an empty JSON object.

    Verifies that '{}' produces an empty dictionary.
    """
    raw = "{}"
    out = parse_json_feature(raw)
    assert out == {}


def test_parse_json_feature_invalid():
    """
    Test that malformed JSON strings raise JSONDecodeError.
    """
    # malformed JSON should raise
    raw = "{ not valid json "
    with pytest.raises(json.JSONDecodeError):
        parse_json_feature(raw)
