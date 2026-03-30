"""Reusable inference pipeline for single and batch predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import shap
from src.config import settings
from src.data.preprocessor import Preprocessor
from src.features.engineer import FeatureEngineer
from src.models.train import load_production_model

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for a single prediction."""
    churn_probability: float
    churn_prediction: bool
    risk_level: str  # Low, Medium, High
    threshold: float


@dataclass
class ExplainedPrediction:
    """Prediction result with SHAP-based explanations."""
    result: PredictionResult
    feature_contributions: list[dict[str, Any]] = field(default_factory=list)
    base_value: float = 0.0


class InferencePipeline:
    """Loads model artifacts and runs predictions."""

    def __init__(
        self,
        preprocessor_path: str = "models/preprocessor.joblib",
        engineer_path: str = "models/feature_engineer.joblib",
        threshold: float | None = None,
    ) -> None:
        self.preprocessor = Preprocessor.load(preprocessor_path)
        self.feature_engineer = FeatureEngineer.load(engineer_path)
        self.model = load_production_model()
        self.threshold = threshold or settings.model.threshold
        self._explainer: shap.TreeExplainer | None = None

    @property
    def explainer(self) -> shap.TreeExplainer:
        """Lazy-init SHAP TreeExplainer (expensive, cache it)."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into features and strip the target column."""
        df_processed = self.preprocessor.transform(df.copy(), is_training=False)
        df_features = self.feature_engineer.transform(df_processed)

        # Always drop target if it leaked through (FeatureEngineer keeps it if present)
        target = settings.data.target_column
        if target in df_features.columns:
            df_features = df_features.drop(columns=[target])

        return df_features

    def predict_single(self, customer_data: dict[str, Any]) -> PredictionResult:
        """Predict churn for a single customer."""
        df = pd.DataFrame([customer_data])
        df_feat = self._get_features(df)

        proba = self.model.predict_proba(df_feat)[:, 1][0]
        prediction = proba >= self.threshold

        if proba < 0.3:
            risk = "Low"
        elif proba < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        return PredictionResult(
            churn_probability=float(proba),
            churn_prediction=bool(prediction),
            risk_level=risk,
            threshold=self.threshold,
        )

    def predict_single_with_explanation(
        self, customer_data: dict[str, Any],
    ) -> ExplainedPrediction:
        """Predict churn with SHAP feature-importance explanations."""
        df = pd.DataFrame([customer_data])
        df_feat = self._get_features(df)

        proba = self.model.predict_proba(df_feat)[:, 1][0]
        prediction = proba >= self.threshold

        if proba < 0.3:
            risk = "Low"
        elif proba < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        result = PredictionResult(
            churn_probability=float(proba),
            churn_prediction=bool(prediction),
            risk_level=risk,
            threshold=self.threshold,
        )

        # SHAP explanations (class 1 = churn)
        # Pass df_feat which matches training features perfectly now
        shap_values = self.explainer.shap_values(df_feat)

        # Format depends on SHAP version & model type. Extract class 1, first row.
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        base = self.explainer.expected_value
        if hasattr(base, "__len__") and len(base) > 1:
            base = base[1]

        contributions = []
        for feat_name, shap_val, feat_val in zip(
            df_feat.columns, sv, df_feat.iloc[0]
        ):
            contributions.append({
                "feature": feat_name,
                "shap_value": float(shap_val),
                "feature_value": float(feat_val),
            })
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return ExplainedPrediction(
            result=result,
            feature_contributions=contributions,
            base_value=float(base),
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict churn for a batch of customers."""
        df_features = self._get_features(df)
        probas = self.model.predict_proba(df_features)[:, 1]

        result = df.copy()
        result["churn_probability"] = probas
        result["churn_prediction"] = probas >= self.threshold
        result["risk_level"] = pd.cut(
            probas, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
        )
        return result
