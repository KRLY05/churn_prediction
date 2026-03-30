"""End-to-end training pipeline: load -> preprocess -> features -> train -> evaluate -> register."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from sklearn.model_selection import train_test_split
from src.config import settings
from src.data.loader import load_raw_data
from src.data.preprocessor import Preprocessor
from src.features.engineer import FeatureEngineer
from src.features.store import FeatureStore
from src.models.evaluate import evaluate_model, find_optimal_threshold
from src.models.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run(config_path: str | None = None) -> None:
    """Execute the full training pipeline."""
    results = run_with_callback()
    logger.info("Training complete. Results: %s", results)


def run_with_callback(
    data_path: str | None = None,
    callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute the full training pipeline with progress callbacks.

    Args:
        data_path: Path to raw CSV data. Defaults to config value.
        callback: Optional function called with progress messages.

    Returns:
        Dictionary with run_id, feature_store_version, threshold, and metrics.
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if callback:
            callback(msg)

    _log("=" * 60)
    _log("TRAINING PIPELINE START")
    _log("=" * 60)

    Path("models").mkdir(exist_ok=True)

    # 1. Load raw data
    _log("Step 1/7: Loading raw data...")
    df = load_raw_data(data_path)

    # 2. Preprocess
    _log("Step 2/7: Preprocessing...")
    preprocessor = Preprocessor()
    df_clean = preprocessor.fit_transform(df, is_training=True)

    # 3. Feature engineering
    _log("Step 3/7: Engineering features...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.fit_transform(df_clean)

    # 4. Save to feature store
    _log("Step 4/7: Saving to feature store...")
    store = FeatureStore()
    version = store.save_features(df_features, description="Training pipeline run")
    _log(f"Feature store version: {version}")

    # 5. Split
    _log("Step 5/7: Splitting train/test...")
    target = settings.data.target_column
    X = df_features.drop(columns=[target])
    y = df_features[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.data.test_size,
        random_state=settings.data.random_state,
        stratify=y,
    )
    _log(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 6. Train
    _log("Step 6/7: Training model...")
    model, run_id = train_model(X_train, y_train)

    # 7. Evaluate
    _log("Step 7/7: Evaluating...")
    import mlflow
    with mlflow.start_run(run_id=run_id):
        optimal_threshold = find_optimal_threshold(model, X_test, y_test, metric="f1")
        metrics = evaluate_model(model, X_test, y_test, threshold=optimal_threshold)
        mlflow.log_param("optimal_threshold", optimal_threshold)

    # Save preprocessor and feature engineer for serving
    preprocessor.save("models/preprocessor.joblib")
    feature_engineer.save("models/feature_engineer.joblib")

    _log("=" * 60)
    _log("TRAINING PIPELINE COMPLETE")
    _log(f"MLflow Run ID: {run_id}")
    _log(f"Feature Store Version: {version}")
    _log(f"Optimal Threshold: {optimal_threshold:.2f}")
    _log(f"F1 Score: {metrics['f1']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
    _log("=" * 60)

    return {
        "run_id": run_id,
        "feature_store_version": version,
        "threshold": optimal_threshold,
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }


if __name__ == "__main__":
    run()

