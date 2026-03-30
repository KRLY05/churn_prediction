"""Lightweight Parquet-based feature store.

Provides versioned, immutable snapshots of engineered features
with metadata tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


class FeatureStore:
    """Simple versioned feature store backed by Parquet files.

    Structure:
        feature_store/
        ├── v001/
        │   ├── features.parquet
        │   └── metadata.json
        ├── v002/
        │   ├── features.parquet
        │   └── metadata.json
        └── ...
    """

    def __init__(self, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path or settings.feature_store.path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_features(
        self,
        df: pd.DataFrame,
        version: str | None = None,
        description: str = "",
    ) -> str:
        """Save a feature set as an immutable versioned snapshot.

        Args:
            df: Feature DataFrame to save.
            version: Version label (e.g., "v001"). Auto-increments if None.
            description: Human-readable description of this snapshot.

        Returns:
            The version string used.
        """
        if version is None:
            version = self._next_version()

        version_dir = self.base_path / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save features
        parquet_path = version_dir / "features.parquet"
        df.to_parquet(parquet_path, index=False)

        # Save metadata
        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved feature set %s: %d rows, %d cols", version, len(df), len(df.columns))
        return version

    def load_features(self, version: str = "latest") -> pd.DataFrame:
        """Load a feature set by version.

        Args:
            version: Version label or "latest" for the most recent.

        Returns:
            Feature DataFrame.
        """
        if version == "latest":
            version = self._latest_version()

        parquet_path = self.base_path / version / "features.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature set not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        logger.info("Loaded feature set %s: %d rows, %d cols", version, len(df), len(df.columns))
        return df

    def list_versions(self) -> list[dict]:
        """List all available feature set versions with metadata."""
        versions = []
        for version_dir in sorted(self.base_path.iterdir()):
            if version_dir.is_dir():
                meta_path = version_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        versions.append(json.load(f))
        return versions

    def get_metadata(self, version: str = "latest") -> dict:
        """Get metadata for a specific version."""
        if version == "latest":
            version = self._latest_version()

        meta_path = self.base_path / version / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found for version: {version}")

        with open(meta_path) as f:
            return json.load(f)

    def _next_version(self) -> str:
        """Generate the next version label."""
        existing = sorted(
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith("v")
        )
        if not existing:
            return "v001"
        last_num = int(existing[-1][1:])
        return f"v{last_num + 1:03d}"

    def _latest_version(self) -> str:
        """Get the latest version label."""
        existing = sorted(
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and d.name.startswith("v")
        )
        if not existing:
            raise FileNotFoundError("No feature sets found in the store.")
        return existing[-1]
