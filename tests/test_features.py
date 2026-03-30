"""Tests for feature engineering and store."""

import pandas as pd
from src.data.preprocessor import Preprocessor
from src.features.engineer import FeatureEngineer
from src.features.store import FeatureStore


def test_feature_engineer_fit_transform(sample_raw_df):
    """FeatureEngineer should one-hot encode and produce numeric columns."""
    pp = Preprocessor()
    df_clean = pp.fit_transform(sample_raw_df)
    fe = FeatureEngineer()
    result = fe.fit_transform(df_clean)
    assert len(result) == 2
    assert "AvgMonthlyCharge" in result.columns
    # Original categorical columns should be gone
    assert "gender" not in result.columns
    assert "Contract" not in result.columns


def test_feature_engineer_column_alignment(sample_raw_df):
    """Transform should align columns even if some categories are missing."""
    pp = Preprocessor()
    df_clean = pp.fit_transform(sample_raw_df)
    fe = FeatureEngineer()
    fe.fit(df_clean)

    # Transform with only first row (may miss some categories)
    result = fe.transform(df_clean.iloc[:1])
    assert set(fe.fitted_columns).issubset(set(result.columns) | {"Churn"})


def test_feature_store_roundtrip(tmp_path, sample_raw_df):
    """Feature store should save and load features correctly."""
    pp = Preprocessor()
    df_clean = pp.fit_transform(sample_raw_df)
    fe = FeatureEngineer()
    df_feat = fe.fit_transform(df_clean)

    store = FeatureStore(base_path=tmp_path / "test_store")
    version = store.save_features(df_feat, description="test")
    loaded = store.load_features(version)
    assert len(loaded) == len(df_feat)
    assert list(loaded.columns) == list(df_feat.columns)


def test_feature_store_versioning(tmp_path):
    """Feature store should auto-increment versions."""
    store = FeatureStore(base_path=tmp_path / "test_store")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    v1 = store.save_features(df, description="first")
    v2 = store.save_features(df, description="second")
    assert v1 == "v001"
    assert v2 == "v002"
    versions = store.list_versions()
    assert len(versions) == 2
