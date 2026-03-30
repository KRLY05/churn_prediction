PYTHON_VERSION ?= 3.14.0
PROJECT_NAME ?= churn_prediction

install_pyenv:
	pyenv install --skip-existing $(PYTHON_VERSION)
	pyenv virtualenv $(PYTHON_VERSION) $(PROJECT_NAME)-$(PYTHON_VERSION) || true
	pyenv local $(PROJECT_NAME)-$(PYTHON_VERSION)
	pip install -r requirements.txt