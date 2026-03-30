
# 🔮 Churn Prediction — MLOps Platform

Customer churn prediction system built with **MLflow**, **Gradio**, and a lightweight **Parquet feature store**.

# Disclaimer

This project was mostly vibe-coded, meaning most of the logic was generated through iterative AI prompting, high-level architectural guidance, and manual verification and debugging.

## Marketing Manager Dashboard
4-tab Gradio web application designed for non-technical marketing managers:
- **📂 Import Data**: Easily load and inspect your latest CSV batch of customers.
- **📊 Batch Predict**: Automatically scores all loaded customers and identifies high-risk clients ready for intervention campaigns.
- **🏋️ Model Training**: Retrain the model on fresh data natively from the browser, whilst seamlessly tracking performance metrics using the local MLflow server.
- **🔍 Single Prediction**: Obtain real-time, SHAP-explained individual decisions. Visually break down exactly *why* a customer is expected to churn (or stay).

## Architecture

```
Raw CSV → Preprocessor → Feature Engineer → Feature Store (Parquet)
                                                    ↓
                                              Training Pipeline
                                                    ↓
                                            MLflow (Tracking + Registry)
                                                    ↓
                                              Gradio App (Dashboard)
```

## Quick Start (Docker)

The fastest and most robust way to launch the full MLOps backend alongside the UI:

```bash
# Starts MLflow (port 5000) + Gradio App (port 7860)
make docker-up
```
Once it's running:
- **Gradio Marketing Dashboard**: http://localhost:7860
- **MLflow Tracking UI**: http://localhost:5000

## Quick Start (Local Setup)

```bash
# Install dependencies into a Pyenv virtualenv
make install_pyenv

# Start MLflow tracking UI in the background
make mlflow             # → http://localhost:5000

# Launch the prediction app dashboard
make serve              # → http://localhost:7860
```

## Makefile Targets

| Target | Description |
| --- | --- |
| `make train` | Run full training pipeline manually via CLI (`python -m pipelines.train_pipeline`) |
| `make serve` | Start Gradio prediction dashboard locally |
| `make mlflow` | Start MLflow tracking UI locally |
| `make test` | Run pytest suite |
| `make lint` | Run ruff linter and auto-formatter |
| `make docker-up` | Start MLflow + Gradio safely via Docker Compose |

## Project Structure

```
├── src/
│   ├── config.py              # Central Pydantic config management
│   ├── data/
│   │   ├── loader.py          # CSV loading with robust schema validation
│   │   └── preprocessor.py    # Data cleaning (sklearn fit/transform API)
│   ├── features/
│   │   ├── engineer.py        # OHE, derived features (fit/transform API)
│   │   └── store.py           # Immutable versioned Parquet feature store
│   ├── models/
│   │   ├── train.py           # RandomForest training & MLflow logging
│   │   └── evaluate.py        # F1/ROC-AUC metrics, threshold optimization
│   └── serving/
│       └── app.py             # 4-Tab Gradio Dashboard Interface
├── pipelines/
│   ├── train_pipeline.py      # Script running loader → engineer → train
│   └── inference_pipeline.py  # Inference with SHAP TreeExplainer support
├── tests/                     # 11-test Pytest validation suite
├── configs/config.yaml        # Easily tweakable model hyperparams & thresholds
├── docker/                    # Docker integrations (MLflow + Gradio services)
└── .github/workflows/         # CI/CD pipelines (Hugging Face deployments)
```

## Configuration

All central hyperparameter and feature settings are defined in `configs/config.yaml`. For example:
```yaml
model:
  type: "RandomForest"
  params:
    n_estimators: 100
    class_weight: "balanced"
  threshold: 0.30          # auto-optimized during training
```

## Dataset Details
**Telco Customer Churn** — 7,043 customers, 22 features, ~26.5% churn rate.
Please see `data_description.txt` at the root for individual column details.
