"""Data preprocessing pipeline — extracted from the notebook.

Handles type conversions, boolean mapping, and column cleanup.
Designed with fit/transform API for consistent train/serve behavior.
"""

from __future__ import annotations

import logging
from typing import Self

import joblib
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

# Columns that should be mapped to boolean
_BOOL_MAP_YES_NO = {
    "Partner": {"Yes": True, "No": False},
    "Dependents": {"Yes": True, "No": False},
    "PhoneService": {"Yes": True, "No": False},
    "PaperlessBilling": {"Yes": True, "No": False},
}

_BOOL_MAP_NUMERIC = {
    "SeniorCitizen": {1: True, 0: False},
}

_TARGET_MAP = {"Yes": True, "No": False}


class Preprocessor:
    """Cleans raw data into a format suitable for feature engineering.

    Mirrors the notebook's data cleaning steps:
    1. TotalCharges: str → numeric, fill NaN with 0
    2. Boolean columns: map to True/False
    3. Churn target: map to True/False
    4. Drop ID and leaky columns
    """

    def __init__(self) -> None:
        self.drop_columns: list[str] = list(settings.data.drop_columns)
        self.target_column: str = settings.data.target_column
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit the preprocessor (currently stateless, but kept for API consistency)."""
        # Validate required columns exist
        required = list(_BOOL_MAP_YES_NO.keys()) + list(_BOOL_MAP_NUMERIC.keys()) + ["TotalCharges"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for preprocessing: {missing}")
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, *, is_training: bool = True) -> pd.DataFrame:
        """Apply preprocessing transformations.

        Args:
            df: Raw DataFrame.
            is_training: If True, processes the target column and drops it from features.

        Returns:
            Cleaned DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit before transform. Call .fit() first.")

        df = df.copy()

        # 1. TotalCharges: str → numeric, fill NaN with 0
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        # 2. Boolean conversions
        for col, mapping in _BOOL_MAP_YES_NO.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).astype(bool)

        for col, mapping in _BOOL_MAP_NUMERIC.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).astype(bool)

        # 3. Target column
        if is_training and self.target_column in df.columns:
            df[self.target_column] = df[self.target_column].map(_TARGET_MAP).astype(bool)

        # 4. Drop ID and leaky columns
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        logger.info("Preprocessed: %d rows, %d columns", len(df), len(df.columns))
        return df

    def fit_transform(self, df: pd.DataFrame, *, is_training: bool = True) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df, is_training=is_training)

    def save(self, path: str) -> None:
        """Persist preprocessor to disk."""
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str) -> Preprocessor:
        """Load a persisted preprocessor."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Preprocessor, got {type(obj)}")
        logger.info("Preprocessor loaded from %s", path)
        return obj
