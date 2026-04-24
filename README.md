# 🏎️ gridlock-ml — F1 Race Position Predictor

> End-to-end ML system: data pipeline → feature engineering → XGBoost training → SHAP explainability → quantile regression confidence intervals → 6-tab Streamlit app → SageMaker serverless deployment. Built with real F1 race data from OpenF1 API.

> inspired from my first project of F1 - "f1-race-predictor"

### 🔗 Live Demo & Deployment

| Platform | Link | Status |
|----------|------|--------|
| **🌐 Live App** | [gridlock-ml-f1-enthusiasts.streamlit.app](https://gridlock-ml-f1-enthusiasts.streamlit.app/) | ✅ Live |
| **☁️ SageMaker API** | `f1-predictor-serverless` (us-east-1) | ✅ Live |
| **📦 GitHub** | [Duanshi-B-Shah/gridlock-ml](https://github.com/Duanshi-B-Shah/gridlock-ml) | ✅ Public |

---

## Overview

**ML system that predicts where an F1 driver will finish a race. It uses XGBoost with 23 engineered features from the OpenF1 API — things like qualifying position, weather, pit strategy, and driver form. It includes SHAP explainability so I can explain why the model predicted P2, confidence intervals so I can say 'P2, likely P1–P4', and it's deployed live on Streamlit Cloud and SageMaker.**

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

## Walk-Through by Module

### 1. Data Pipeline (`src/f1_predictor/data/fetcher.py`)

Pulls race data from **5 OpenF1 API endpoints** — positions, drivers, weather, stints, and qualifying.

- **Grid position** is derived from the first position entry per driver (the API returns time-series position data, not a grid column directly)
- **Qualifying positions** come from a separate qualifying session — found via `meeting_key`, then the last position per driver is taken as the final classification
- **Weather** is aggregated from minute-by-minute readings to session-level means + a wet/dry binary flag
- **Rate limiting**: 2s delays between API calls, 3s between races, 5 retry attempts with 10–50s linear backoff on HTTP 429
- **Graceful degradation**: If any endpoint returns no data for a session, features fall back to sensible defaults instead of crashing

### 2. Feature Engineering (`src/f1_predictor/features/engineering.py`)

Transforms raw race results into **23 model-ready features** across 8 categories:

| Category | What It Captures | Why It Matters |
|----------|-----------------|----------------|
| **Grid + Qualifying** | Starting position, qualifying pace, grid penalties | Where you start strongly predicts where you finish |
| **Driver Form** | Rolling avg finish (3 & 5 races), points trend, positions gained/lost | A driver on a hot streak performs differently |
| **Circuit History** | Driver's avg finish at this track, number of prior races | Some drivers excel at specific circuits (Hamilton at Silverstone) |
| **Team Performance** | Team season avg finish, points per race | In F1, the car matters more than the driver — team performance is critical |
| **Reliability** | DNF rate (season + circuit) | Unreliable cars are more likely to retire mid-race |
| **Weather** | Air/track temp, humidity, wind, wet/dry flag | Rain causes massive position swings — high-signal feature |
| **Strategy** | Number of pit stops | More stops = more risk of position loss in the pit lane |
| **Circuit Type + Teammate** | Street circuit flag, rolling teammate delta | Street circuits are less predictable; teammate comparison shows car vs driver performance |

**Critical design**: Every rolling feature uses `.shift(1)` — the model **never** sees the current race when computing features. This prevents data leakage.

### 3. Model Training (`src/f1_predictor/training/train.py`)

- **Algorithm**: XGBoost Regressor — best-in-class for small tabular datasets
- **Why not deep learning?** Only ~400 training rows (20 drivers × ~20 races). XGBoost empirically wins at this scale.
- **Cross-validation**: `TimeSeriesSplit(n_splits=5)` — training set always precedes validation set chronologically
- **Hyperparameter tuning**: `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- **Baseline**: "Grid position = finishing position" — the model achieves **~35% lower MAE** than this baseline
- **Output**: Predictions clipped to [1, 20] and rounded to integers (P5, not P4.9)

### 4. SHAP Explainability (`src/f1_predictor/explainability/shap_explainer.py`)

Uses `shap.TreeExplainer` for **exact** (not approximate) feature contribution values:

- **Beeswarm** (global): Which features matter most across all predictions
- **Waterfall** (per-prediction): Decompose a single prediction — "grid_position=1 pushed prediction down by 3.2 positions, but is_wet_race=1 pushed it up by 2.1"
- **Top contributors**: The 5 features that drove a specific prediction the most, with direction and magnitude

### 5. Confidence Intervals (`src/f1_predictor/explainability/confidence.py`)

Instead of just "P5", the system outputs **"P5 (range: P3–P7)"** — an 80% confidence interval.

Three separate XGBoost models trained with `objective="reg:quantileerror"`:
- **10th percentile** → optimistic bound (P3)
- **50th percentile** → median prediction (P5)
- **90th percentile** → conservative bound (P7)

### 6. MLflow Tracking (`src/f1_predictor/tracking/mlflow_tracker.py`)

Every training run automatically logs:
- **Parameters**: All hyperparams, feature config, data stats, tuning method
- **Metrics**: MAE, RMSE, R², baseline comparison, CV fold scores
- **Artifacts**: Trained model, SHAP plots, evaluation plots
- **Model Registry**: Optional model registration with `--register` flag for staging/production promotion

### 7. Streamlit App (`app/streamlit_app.py`)

700+ lines, 6 interactive tabs:

| Tab | Demo Move |
|-----|-----------|
| **🏎️ What is F1?** | Onboards someone who's never watched F1 — explains the sport, key terms, and how the model works |
| **🎛️ Predict** | Adjust 23 sliders → see the predicted position change in real-time |
| **🔍 Explainability** | Pick a driver + circuit → SHAP waterfall shows *why* the model predicted that position |
| **🏁 Race Sim** | Select any race → predict all 20 drivers. Auto-fill mode populates features from real historical data |
| **👤 Drivers** | Multi-select drivers, filter by circuit, position trend charts, predicted vs actual scatter |
| **📋 Batch** | Full dataset MAE vs baseline comparison |

### 8. Deployment (`infra/`)

| Target | How | Cost |
|--------|-----|------|
| **Streamlit Cloud** | Auto-deploys from GitHub on push to `main` | Free |
| **SageMaker Serverless** | `python infra/sagemaker/deploy.py --bucket BUCKET` — REST API, scales to zero | ~$1-5/mo |
| **Docker** | `docker build -t gridlock-ml . && docker run -p 8501:8501 gridlock-ml` | — |
| **App Runner** | `./infra/apprunner/deploy.sh` — builds Docker, pushes ECR, creates service | ~$10-25/mo |

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
