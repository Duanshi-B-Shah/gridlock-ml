# Architecture — F1 Race Position Predictor

## System Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    OpenF1 API (2023+)                     │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│ /position│ /drivers │ /weather │ /stints  │ /sessions    │
│ (grid +  │ (name,   │ (temp,   │ (pits,   │ (qualifying  │
│  finish) │  team)   │  rain)   │  tires)  │  positions)  │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────┬───────┘
     │          │          │          │            │
     ▼          ▼          ▼          ▼            ▼
┌──────────────────────────────────────────────────────────┐
│                  Data Fetcher (fetcher.py)                │
│  fetch_positions · fetch_drivers · fetch_weather         │
│  fetch_stints · fetch_qualifying_positions               │
└─────────────────────────┬────────────────────────────────┘
                          │ Raw CSVs → data/raw/
                          ▼
┌──────────────────────────────────────────────────────────┐
│              Feature Engineering (engineering.py)         │
│  23 features across 8 categories                         │
│  All rolling features use .shift(1) — no leakage         │
└─────────────────────────┬────────────────────────────────┘
                          │ features.csv → data/processed/
                          ▼
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌───────────┐  ┌────────────┐  ┌─────────────┐
    │  Train    │  │  Evaluate  │  │  Streamlit  │
    │ XGBoost   │  │ MAE, RMSE  │  │  6-tab App  │
    │ train.py  │  │evaluate.py │  │ streamlit   │
    └─────┬─────┘  └────────────┘  │ _app.py     │
          │                        └──────┬──────┘
          ▼                               │
    ┌───────────┐                         │
    │  Model    │◀────────────────────────┘
    │ .joblib   │   (loads for inference)
    │ models/   │
    └─────┬─────┘
          │
          ▼
    ┌───────────────────────────────────────┐
    │           Explainability              │
    │  SHAP TreeExplainer (waterfall,       │
    │  beeswarm) + Quantile Regression     │
    │  (P10/P50/P90 confidence intervals)  │
    └───────────────────────────────────────┘
          │
          ▼
    ┌───────────────────────────────────────┐
    │         MLflow Tracking               │
    │  Params · Metrics · Artifacts · Model │
    │  Registry · Run Comparison            │
    └───────────────────────────────────────┘
```

## Feature Groups (23 Features)

| Group | Features | Source | Leakage Prevention |
|-------|----------|--------|--------------------|
| **Grid** | `grid_position` | First position entry in race session | Session-level |
| **Qualifying** | `quali_position`, `grid_quali_delta` | Qualifying session final positions | Pre-race data |
| **Driver Form** | `rolling_avg_finish_short/long`, `rolling_avg_points`, `position_delta_trend` | Computed from prior races | `.shift(1)` rolling |
| **Circuit** | `circuit_avg_finish`, `circuit_race_count` | Driver's history at this circuit | `.shift(1)` expanding |
| **Team** | `team_season_avg_finish`, `team_points_per_race` | Team's season performance | `.shift(1)` expanding |
| **Reliability** | `dnf_rate_season`, `dnf_rate_circuit` | DNF flags | `.shift(1)` expanding |
| **Weather** | `air_temperature`, `track_temperature`, `humidity`, `wind_speed`, `is_wet_race` | OpenF1 /weather endpoint | Session-level (no leakage) |
| **Strategy** | `n_pit_stops` | OpenF1 /stints endpoint | Session-level |
| **Circuit Type** | `is_street_circuit` | Derived from circuit name | Static mapping |
| **Teammate** | `teammate_delta_rolling` | Finish position vs teammate | `.shift(1)` rolling |

## Data Leakage Prevention

All rolling/expanding features use `.shift(1)` to ensure we only look at **past** data:
- Rolling averages exclude the current race
- Circuit history excludes the current race at that circuit
- Teammate delta uses only prior teammate comparisons
- `TimeSeriesSplit` for cross-validation preserves temporal ordering
- Weather and strategy features are session-level (available before/during race)

## Models

### Primary Model: XGBoost Regressor
- **Objective**: `reg:squarederror`
- **CV**: `TimeSeriesSplit(n_splits=5)`
- **Tuning**: `GridSearchCV` over n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Baseline**: Grid position = finishing position (model must beat this)
- **Output**: Integer positions (P1–P20), clipped and rounded

### Quantile Regression Models
- 3 separate XGBoost models with `objective="reg:quantileerror"`
- Quantiles: 10th percentile (optimistic), 50th (median), 90th (conservative)
- Output: "Predicted P5 (range: P3–P7)" — 80% confidence interval

## Explainability

### SHAP (SHapley Additive exPlanations)
- Uses `shap.TreeExplainer` — exact and fast for tree-based models
- **Global view**: Beeswarm plot showing feature impact distribution across all predictions
- **Per-prediction**: Waterfall chart decomposing a single prediction into feature contributions
- **Top contributors**: Top N features driving each prediction, with direction and magnitude

### Confidence Intervals
- Quantile regression produces prediction ranges, not just point estimates
- Useful for communicating uncertainty: "We're 80% confident this driver finishes P3–P7"
- Lower bound (P10) = optimistic scenario, upper bound (P90) = conservative scenario

## MLflow Integration

| What's Tracked | Details |
|---------------|---------|
| **Parameters** | All XGBoost hyperparams, feature config, data stats, tuning method |
| **Metrics** | MAE, RMSE, R², baseline comparison, CV fold scores, improvement % |
| **Artifacts** | Trained model, SHAP plots, evaluation plots, feature CSV |
| **Tags** | Project name, model type, task type, run type |
| **Registry** | Optional model registration with `--register` flag |
