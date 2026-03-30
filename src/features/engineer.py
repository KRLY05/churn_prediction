"""Feature engineering — one-hot encoding with fit/transform API.

Mirrors the notebook's `pd.get_dummies(drop_first=True)` approach
but stores the fitted column mapping for consistent inference.
"""

from __future__ import annotations

import logging
from typing import Self

import joblib
import pandas as pd

from src.config import settings
from src.features.schema import CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms preprocessed data into model-ready features.

    - One-hot encodes categorical columns (drop_first=True)
    - Stores the fitted column order for consistent inference
    - Adds derived features (e.g., AvgMonthlyCharge)
    """

    def __init__(self) -> None:
        self.categorical_columns: list[str] = list(CATEGORICAL_FEATURES)
        self.target_column: str = settings.data.target_column
        self.fitted_columns: list[str] | None = None
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit: learn the column order from training data."""
        df_encoded = self._encode(df)
        self.fitted_columns = [c for c in df_encoded.columns if c != self.target_column]
        self._is_fitted = True
        logger.info("FeatureEngineer fitted with %d feature columns", len(self.fitted_columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted column mapping.

        Ensures output always has the same columns as training,
        filling missing one-hot columns with 0.
        """
        if not self._is_fitted or self.fitted_columns is None:
            raise RuntimeError("FeatureEngineer must be fit before transform.")

        df_encoded = self._encode(df)

        # Align columns with training: add missing, drop extra
        for col in self.fitted_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Keep target if present, plus fitted feature columns
        output_cols = list(self.fitted_columns)
        if self.target_column in df_encoded.columns:
            output_cols.append(self.target_column)

        df_encoded = df_encoded[output_cols]
        logger.info("Transformed: %d rows, %d columns", len(df_encoded), len(df_encoded.columns))
        return df_encoded

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding and derived features."""
        df = df.copy()

        # Derived feature: average monthly charge
        df["AvgMonthlyCharge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

        # One-hot encode categoricals
        cats_present = [c for c in self.categorical_columns if c in df.columns]
        df = pd.get_dummies(df, columns=cats_present, drop_first=True)

        # Ensure all columns are numeric (bool → int for model)
        for col in df.columns:
            if df[col].dtype == "bool" and col != self.target_column:
                df[col] = df[col].astype(int)

        return df

    def save(self, path: str) -> None:
        """Persist feature engineer to disk."""
        joblib.dump(self, path)
        logger.info("FeatureEngineer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> FeatureEngineer:
        """Load a persisted feature engineer."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected FeatureEngineer, got {type(obj)}")
        logger.info("FeatureEngineer loaded from %s", path)
        return obj
