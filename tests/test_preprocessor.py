"""Tests for the data preprocessor."""

from src.data.preprocessor import Preprocessor


def test_fit_transform_basic(sample_raw_df):
    """Preprocessor should clean data without errors."""
    pp = Preprocessor()
    result = pp.fit_transform(sample_raw_df)
    assert len(result) == 2
    assert "customerID" not in result.columns
    assert "CustomerSatisfactionScore" not in result.columns


def test_total_charges_conversion(sample_raw_df):
    """TotalCharges should be numeric; empty strings become 0."""
    pp = Preprocessor()
    result = pp.fit_transform(sample_raw_df)
    assert result["TotalCharges"].dtype == float
    assert result["TotalCharges"].iloc[1] == 0.0


def test_boolean_columns(sample_raw_df):
    """Boolean columns should be True/False."""
    pp = Preprocessor()
    result = pp.fit_transform(sample_raw_df)
    for col in ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        assert result[col].dtype == bool, f"{col} should be bool"


def test_churn_target(sample_raw_df):
    """Churn should be mapped to bool."""
    pp = Preprocessor()
    result = pp.fit_transform(sample_raw_df)
    assert result["Churn"].dtype == bool
    assert not result["Churn"].iloc[0]
    assert result["Churn"].iloc[1]


def test_inference_mode(sample_raw_df):
    """In inference mode, Churn column should not be processed."""
    pp = Preprocessor()
    pp.fit(sample_raw_df)
    df_no_churn = sample_raw_df.drop(columns=["Churn"])
    result = pp.transform(df_no_churn, is_training=False)
    assert "Churn" not in result.columns
