"""Feature schema definitions and validation."""

from __future__ import annotations

from src.config import settings

# Feature groups loaded from config
NUMERIC_FEATURES: list[str] = list(settings.features.numeric)
BINARY_FEATURES: list[str] = list(settings.features.binary)
CATEGORICAL_FEATURES: list[str] = list(settings.features.categorical)

ALL_INPUT_FEATURES: list[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Valid values for categorical features (for input validation in serving)
CATEGORICAL_VALUES: dict[str, list[str]] = {
    "gender": ["Male", "Female"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
