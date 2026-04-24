# 🏎️ gridlock-ml — F1 Race Position Predictor

> End-to-end ML system: data pipeline → feature engineering → XGBoost training → SHAP explainability → quantile regression confidence intervals → 6-tab Streamlit app → SageMaker serverless deployment. Built with real F1 race data from OpenF1 API.

### 🔗 Live Demo & Deployment

| Platform | Link | Status |
|----------|------|--------|
| **🌐 Live App** | [gridlock-ml-f1-enthusiasts.streamlit.app](https://gridlock-ml-f1-enthusiasts.streamlit.app/) | ✅ Live |
| **☁️ SageMaker API** | `f1-predictor-serverless` (us-east-1) | ✅ Live |
| **📦 GitHub** | [Duanshi-B-Shah/gridlock-ml](https://github.com/Duanshi-B-Shah/gridlock-ml) | ✅ Public |

---

## Overview

| Component | Description |
|-----------|-------------|
| **Model** | XGBoost Regressor (scikit-learn API) + quantile regression for confidence intervals |
| **Data** | OpenF1 API — positions, weather, stints, qualifying, driver/team metadata |
| **Features** | 23 engineered features across 8 categories |
| **Explainability** | SHAP TreeExplainer — global beeswarm + per-prediction waterfall charts |
| **Confidence** | Quantile regression (10th/50th/90th percentile) → "P3–P7" ranges |
| **Tracking** | MLflow experiment tracking, model versioning, artifact logging |
| **UI** | Streamlit app with 6 interactive tabs |
| **Baseline** | Grid position = finish position (model must beat this) |

## Quick Start

```bash
# 1. Clone and set up
cd f1-race-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Fetch multi-season data from OpenF1 API
python scripts/fetch_data.py --season 2024 2025

# 4. Train the main model
python scripts/train_model.py

# 5. Train quantile models (for confidence intervals)
python scripts/train_quantile.py

# 6. Evaluate
python scripts/evaluate_model.py

# 7. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

## Project Structure

```
f1-race-predictor/
├── src/f1_predictor/           # Core library
│   ├── config.py               # Pydantic settings
│   ├── data/                   # API client, schemas
│   │   ├── fetcher.py          # OpenF1 API (positions, weather, stints, qualifying)
│   │   └── schemas.py          # Pydantic models
│   ├── features/               # Feature engineering (23 features)
│   │   └── engineering.py
│   ├── training/               # Training & evaluation
│   │   ├── train.py            # XGBoost + TimeSeriesSplit CV + MLflow
│   │   └── evaluate.py         # Metrics, plots, MLflow logging
│   ├── inference/              # Model loading & prediction
│   │   └── predict.py
│   ├── explainability/         # SHAP + confidence intervals
│   │   ├── shap_explainer.py   # TreeExplainer, waterfall, beeswarm
│   │   └── confidence.py       # Quantile regression (10th/50th/90th)
│   ├── tracking/               # MLflow integration
│   │   └── mlflow_tracker.py
│   └── utils/
│       └── logging.py
├── app/
│   └── streamlit_app.py        # 6-tab interactive demo
├── scripts/                    # CLI entry points
│   ├── fetch_data.py           # --season 2024 2025 --driver Verstappen
│   ├── train_model.py          # --no-tune --experiment --register
│   ├── train_quantile.py       # Quantile regression models
│   ├── evaluate_model.py       # Metrics + plots + MLflow
│   └── compare_runs.py         # Compare MLflow experiment runs
├── tests/                      # Unit & integration tests (40+ test cases)
├── data/                       # Raw & processed data (gitignored)
├── models/                     # Serialized models (gitignored)
├── mlruns/                     # MLflow tracking data (gitignored)
├── plots/                      # Evaluation & SHAP plots (gitignored)
├── notebooks/                  # EDA notebook
├── docs/                       # Architecture documentation
├── Dockerfile                  # Container deployment
├── Makefile                    # All build commands
└── pyproject.toml              # Project metadata & tool config
```

## Features (23 Total)

| Category | Features | Source |
|----------|----------|--------|
| **Grid** | `grid_position` | OpenF1 position data |
| **Qualifying** | `quali_position`, `grid_quali_delta` | OpenF1 qualifying session |
| **Driver Form** | `rolling_avg_finish_short/long`, `rolling_avg_points`, `position_delta_trend` | Computed from race history |
| **Circuit** | `circuit_avg_finish`, `circuit_race_count` | Grouped by driver + circuit |
| **Team** | `team_season_avg_finish`, `team_points_per_race` | Grouped by team + season |
| **Reliability** | `dnf_rate_season`, `dnf_rate_circuit` | DNF flags |
| **Weather** | `air_temperature`, `track_temperature`, `humidity`, `wind_speed`, `is_wet_race` | OpenF1 weather API |
| **Strategy** | `n_pit_stops` | OpenF1 stints API |
| **Circuit Type** | `is_street_circuit` | Derived (Monaco, Jeddah, Singapore, etc.) |
| **Teammate** | `teammate_delta_rolling` | Rolling avg vs teammate finish |

## Streamlit App — 6 Tabs

| Tab | What It Does |
|-----|-------------|
| **🏎️ What is F1?** | Beginner-friendly intro — explains F1, key concepts, and how the model works |
| **🎛️ Predict** | 23 sidebar sliders → real-time position prediction |
| **🔍 Explainability** | SHAP waterfall (why this prediction?) + beeswarm (global) + confidence intervals |
| **🏁 Race Sim** | Pick a race → predict all 20 drivers. Or auto-fill features from real data |
| **👤 Drivers** | Multi-select drivers, filter by circuit, position trend charts |
| **📋 Batch** | Full dataset predictions, MAE comparison vs baseline |

## Explainability & Confidence

### SHAP
- **Global**: Beeswarm plot showing which features matter most across all predictions
- **Per-prediction**: Waterfall chart — "Verstappen predicted P2 because grid_position=1 pushed down, but is_wet_race=1 pushed up"
- **Top contributors**: Top 5 features driving each prediction

### Confidence Intervals
- 3 XGBoost quantile models (10th, 50th, 90th percentile)
- Output: "Predicted P5 (range: P3–P7)" — 80% confidence interval
- Train: `python scripts/train_quantile.py`

## MLOps — MLflow Integration

Every training run logs hyperparams, metrics, and artifacts automatically.

```bash
python scripts/train_model.py                    # Default: tracking + tuning
python scripts/train_model.py --no-track         # Disable MLflow
python scripts/train_model.py --register          # Register in Model Registry
python scripts/compare_runs.py --top-n 5         # Compare runs
make mlflow                                       # Launch MLflow UI
```

## Makefile Commands

```bash
make install          # Install all dependencies
make fetch            # Fetch data (2024 + 2025 seasons)
make train            # Train XGBoost model
make train-quantile   # Train quantile regression models
make evaluate         # Evaluate + generate plots
make shap             # Generate SHAP plots
make app              # Launch Streamlit app
make mlflow           # Launch MLflow UI
make compare          # Compare experiment runs
make test             # Run all tests
make lint             # Ruff linter
make format           # Ruff formatter
make clean            # Remove generated artifacts
make all              # fetch → train → train-quantile → evaluate → app
```

## Deployment

### Streamlit Cloud (Live)
Currently deployed at [gridlock-ml-f1-enthusiasts.streamlit.app](https://gridlock-ml-f1-enthusiasts.streamlit.app/). Auto-deploys on push to `main`.

### SageMaker Serverless Endpoint (Live)
REST API for programmatic access. Scales to zero when idle (~$1-5/mo).
```bash
python infra/sagemaker/deploy.py --bucket YOUR-BUCKET
python infra/sagemaker/test_endpoint.py  # Test it
```

### Docker
```bash
docker build -t gridlock-ml . && docker run -p 8501:8501 gridlock-ml
```

### App Runner
Full deployment script at `infra/apprunner/deploy.sh` — builds Docker, pushes to ECR, creates App Runner service.

## License

MIT
