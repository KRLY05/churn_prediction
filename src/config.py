"""Centralized configuration management using Pydantic and YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


class DataConfig(BaseModel):
    raw_path: str = "data/raw/Telco-Customer-Churn.csv"
    target_column: str = "Churn"
    id_column: str = "customerID"
    drop_columns: list[str] = Field(default_factory=lambda: ["customerID", "CustomerSatisfactionScore"])
    test_size: float = 0.2
    random_state: int = 42


class FeatureConfig(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    binary: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    type: str = "RandomForest"
    params: dict[str, Any] = Field(default_factory=dict)
    threshold: float = 0.30


class MLflowConfig(BaseModel):
    experiment_name: str = "churn-prediction"
    tracking_uri: str = "mlruns"
    model_name: str = "churn-classifier"


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860


class FeatureStoreConfig(BaseModel):
    path: str = "feature_store"


class Settings(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG) -> Settings:
        """Load settings from a YAML config file."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


# Singleton-ish: import and use directly
settings = Settings.from_yaml()
