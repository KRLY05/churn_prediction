"""Data loading with schema validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "CustomerSatisfactionScore", "MonthlyCharges",
    "TotalCharges", "Churn",
]


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV.

    Args:
        path: Path to CSV file. Defaults to config value.

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If expected columns are missing.
    """
    path = Path(path or settings.data.raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)

    # Validate schema
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df
