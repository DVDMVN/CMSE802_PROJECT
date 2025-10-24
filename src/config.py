"""
Project-wide configuration and paths.

This module centralizes dataset locations so other modules
can import them instead of hardcoding strings.

Author: Alex (Ze) Chen
Date: 2025-10-24
"""

DATA_LINK = "https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2025-09-11T03_20_29_905Z.zip"
DATA_PATH = "data/raw/Kickstarter_2025-09-11T03_20_29_905Z"
PROCESSED_DATA_PATH = "data/processed/post_processing_kick.parquet"
