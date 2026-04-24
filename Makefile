.PHONY: install fetch train train-quantile evaluate shap app mlflow compare test lint format clean all

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

fetch:
	python scripts/fetch_data.py --season 2024 2025

train:
	python scripts/train_model.py

train-quantile:
	python scripts/train_quantile.py

evaluate:
	python scripts/evaluate_model.py

shap:
	PYTHONPATH=src python -c "from f1_predictor.explainability.shap_explainer import F1Explainer; e = F1Explainer(); e.plot_beeswarm(); e.plot_bar(); print('SHAP plots saved to plots/')"

app:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

compare:
	python scripts/compare_runs.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

clean:
	rm -rf data/raw/* data/processed/* models/* plots/* mlruns/*
	find . -type d -name __pycache__ -exec rm -rf {} +

all: fetch train train-quantile evaluate app
