"""Tests for model training and evaluation."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.models.evaluate import evaluate_model, find_optimal_threshold


@pytest.fixture
def trained_model_and_data():
    """Create a simple trained model for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:80], y[:80])
    return model, X[80:], y[80:]


def test_evaluate_model(trained_model_and_data):
    """evaluate_model should return expected metric keys."""
    model, X_test, y_test = trained_model_and_data
    metrics = evaluate_model(model, X_test, y_test, threshold=0.5, log_to_mlflow=False)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_find_optimal_threshold(trained_model_and_data):
    """find_optimal_threshold should return a float between 0.1 and 0.9."""
    model, X_test, y_test = trained_model_and_data
    threshold = find_optimal_threshold(model, X_test, y_test, metric="f1")
    assert 0.1 <= threshold <= 0.9
