PYTHON_VERSION ?= 3.14.0
PROJECT_NAME ?= churn_prediction

install_pyenv:
	pyenv install --skip-existing $(PYTHON_VERSION)
	pyenv virtualenv $(PYTHON_VERSION) $(PROJECT_NAME)-$(PYTHON_VERSION) || true
	pyenv local $(PROJECT_NAME)-$(PYTHON_VERSION)
	pip install -r requirements.txt

train:
	python -m pipelines.train_pipeline

serve:
	python -m src.serving.app

mlflow:
	mlflow ui --backend-store-uri mlruns --port 5000

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ pipelines/

docker-build:
	docker build -f docker/Dockerfile -t churn-predictor .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

.PHONY: install_pyenv train serve mlflow test lint docker-build docker-up