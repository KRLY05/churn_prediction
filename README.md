# 🔮 Churn Prediction — MLOps Platform

Production-grade customer churn prediction system built with **MLflow**, **Gradio**, and a lightweight **Parquet feature store**.

## Architecture

```
Raw CSV → Preprocessor → Feature Engineer → Feature Store (Parquet)
                                                    ↓
                                              Training Pipeline
                                                    ↓
                                            MLflow (Tracking + Registry)
                                                    ↓
                                              Gradio App (Inference)
```

## Quick Start

```bash
# Install dependencies
make install_pyenv      # or: pip install -r requirements.txt

# Train the model
make train

# Launch the prediction app
make serve              # → http://localhost:7860

# View experiment tracking
make mlflow             # → http://localhost:5000
```

## Project Structure

```
├── src/
│   ├── config.py              # Pydantic configuration management
│   ├── data/
│   │   ├── loader.py          # CSV loading with schema validation
│   │   └── preprocessor.py    # Data cleaning (fit/transform API)
│   ├── features/
│   │   ├── schema.py          # Feature definitions & valid values
│   │   ├── engineer.py        # One-hot encoding (fit/transform API)
│   │   └── store.py           # Versioned Parquet feature store
│   ├── models/
│   │   ├── train.py           # RandomForest + MLflow tracking
│   │   └── evaluate.py        # Metrics, threshold optimization
│   └── serving/
│       └── app.py             # Gradio inference UI
├── pipelines/
│   ├── train_pipeline.py      # End-to-end training orchestration
│   └── inference_pipeline.py  # Batch/single prediction pipeline
├── tests/                     # pytest test suite (11 tests)
├── configs/config.yaml        # Hyperparameters, paths, thresholds
├── docker/                    # Dockerfile + docker-compose
└── .github/workflows/         # CI/CD (lint, test, deploy to HF Spaces)
```

## Makefile Targets

| Target | Description |
| --- | --- |
| `make train` | Run full training pipeline |
| `make serve` | Start Gradio prediction app |
| `make mlflow` | Start MLflow tracking UI |
| `make test` | Run pytest suite |
| `make lint` | Run ruff linter |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start MLflow + Gradio via Docker Compose |

## Training Pipeline

The pipeline (`python -m pipelines.train_pipeline`) executes 7 steps:

1. **Load** raw CSV with schema validation
2. **Preprocess**: type conversions, boolean mapping, column cleanup
3. **Engineer features**: one-hot encoding, derived features (AvgMonthlyCharge)
4. **Save** to versioned feature store
5. **Split** train/test (80/20, stratified)
6. **Train** RandomForest with MLflow tracking
7. **Evaluate** with threshold optimization, register model

## Configuration

All settings in `configs/config.yaml`:

```yaml
model:
  type: "RandomForest"
  params:
    n_estimators: 100
    class_weight: "balanced"
  threshold: 0.30          # auto-optimized during training
```

## Deployment

### Hugging Face Spaces
The CD pipeline (`.github/workflows/cd.yml`) deploys to HF Spaces on push to `main`.
Set the `HF_TOKEN` secret in your GitHub repo settings.

### Docker
```bash
make docker-up   # Starts MLflow (port 5000) + Gradio (port 7860)
```

## Dataset

**Telco Customer Churn** — 7,043 customers, 22 features, ~26.5% churn rate.
See `data_description.txt` for column details.
