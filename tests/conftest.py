"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
    """Minimal raw DataFrame matching the CSV schema."""
    return pd.DataFrame({
        "customerID": ["1234-ABCDE", "5678-FGHIJ"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [12, 0],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["One year", "Month-to-month"],
        "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": ["Mailed check", "Electronic check"],
        "CustomerSatisfactionScore": [None, 5.0],
        "MonthlyCharges": [50.0, 90.0],
        "TotalCharges": ["600.0", " "],
        "Churn": ["No", "Yes"],
    })
