"""Streamlit demo UI for the F1 Race Position Predictor."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from f1_predictor.config import settings
from f1_predictor.features.engineering import FEATURE_COLUMNS
from f1_predictor.inference.predict import load_model, predict

# -- Page Config --
st.set_page_config(
    page_title="F1 Race Position Predictor",
    page_icon="🏎️",
    layout="wide",
)

# -- Styling --
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E10600;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-header">🏎️ F1 Race Position Predictor</p>', unsafe_allow_html=True
)
st.caption("Predict finishing positions using XGBoost — powered by OpenF1 data")

# -- Load Model --
try:
    model = load_model()
    st.success("Model loaded successfully", icon="✅")
except FileNotFoundError:
    st.error("No trained model found. Run `python scripts/train_model.py` first.")
    st.stop()

# ═══════════════════════════════════════════════════
# TAB LAYOUT: Manual Prediction | Driver Explorer | Batch Predictions
# ═══════════════════════════════════════════════════

tab_intro, tab_predict, tab_explain, tab_race_sim, tab_driver, tab_batch = st.tabs(
    ["🏎️ What is F1?", "🎛️ Predict", "🔍 Explainability", "🏁 Race Sim", "👤 Drivers", "📋 Batch"]
)

# ─────────────────────────────────────────────────
# TAB 0: What is F1? (Intro for Beginners)
# ─────────────────────────────────────────────────

with tab_intro:
    st.markdown("""
    ## 🏎️ Welcome! Let's Talk Formula 1

    Never watched an F1 race? No problem. Here's everything you need to know
    to understand this app — explained like you're 5 (but with cool data science).

    ---

    ### 🤔 What is Formula 1?

    Formula 1 is the **fastest** and most prestigious motor racing championship in the world.
    Think of it as the Champions League of racing — 20 of the best drivers on the planet,
    driving cars that cost **$150 million each**, at speeds over **350 km/h (220 mph)**.

    There are about **24 races per year** (called Grand Prix) across iconic circuits worldwide —
    Monaco's narrow streets, Monza's high-speed straights, Silverstone's legendary corners.

    ---

    ### 🏁 How Does a Race Weekend Work?

    Each race weekend has 3 key sessions:

    | Session | What Happens | Why It Matters |
    |---------|-------------|----------------|
    | **Practice (FP1-FP3)** | Drivers test car setup, tire wear, and strategy | Teams gather data |
    | **Qualifying** | Drivers do hot laps to determine **starting grid** | Fastest lap = Pole Position (P1 on the grid) |
    | **Race** | 50-70 laps of wheel-to-wheel racing | Points awarded to top 10 finishers |

    ---

    ### 📊 The Points System

    | Position | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11+ |
    |----------|----|----|----|----|----|----|----|----|----|----|------|
    | **Points** | 25 | 18 | 15 | 12 | 10 | 8 | 6 | 4 | 2 | 1 | 0 |

    The driver with the most points at the end of the season wins the
    **World Drivers' Championship** 🏆

    ---

    ### 🧩 Key Concepts This App Uses

    Here are the F1 terms you'll see in this app:

    | Term | What It Means | Example |
    |------|--------------|---------|
    | **Grid Position** | Where a driver starts the race (from qualifying) | Verstappen starts P1 (pole) |
    | **Finishing Position** | Where they actually finish | Hamilton finishes P3 |
    | **DNF** | "Did Not Finish" — car broke down or crashed | Engine failure = DNF |
    | **Pit Stop** | Driver comes in to change tires mid-race (takes ~2.5 seconds!) | Usually 1-2 per race |
    | **Wet Race** 🌧️ | Rain during the race — chaos ensues, underdogs shine | Brazil 2024 was legendary |
    | **Street Circuit** | Racing on actual city streets (tight, bumpy, no margin for error) | Monaco, Singapore, Las Vegas |
    | **Teammate** | The other driver in the same team (same car, direct comparison) | Verstappen vs Pérez |

    ---

    ### 🤖 What Does This App Do?

    We built a **machine learning model** that predicts where a driver will finish a race.
    Here's the simple version:

    ```
    📥 Input: Grid position + driver form + weather + circuit + team strength
         ↓
    🧠 XGBoost Model (learns patterns from 100s of past races)
         ↓
    📤 Output: "Verstappen will finish P2" (with confidence: P1–P3)
    ```

    #### The 23 Features We Use:
    """)

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("""
        **🏁 Race Setup**
        - Grid position (where they start)
        - Qualifying position
        - Grid penalty indicator
        - Street circuit flag
        - Number of pit stops
        """)

    with col_f2:
        st.markdown("""
        **📈 Driver & Team Form**
        - Rolling avg finish (3 & 5 races)
        - Rolling avg points
        - Position gain/loss trend
        - Team season avg finish
        - Team points per race
        - Teammate comparison
        - DNF rate (season + circuit)
        """)

    with col_f3:
        st.markdown("""
        **🌤️ Conditions**
        - Air temperature
        - Track temperature
        - Humidity
        - Wind speed
        - Wet/dry race flag
        - Circuit history (avg finish, race count)
        """)

    st.markdown("""
    ---

    ### 🎯 How Good Is It?

    We compare our model against a simple **baseline**: *"The driver finishes where they start"*
    (grid position = finishing position). Our model beats this baseline because:

    - 🌧️ It knows **rain** causes chaos (underdogs can win!)
    - 📉 It knows some drivers are on a **hot streak** or cold streak
    - 🏙️ It knows **street circuits** are unpredictable
    - 🏎️ It knows which **teams** have the fastest car right now
    - 🔧 It knows which drivers are **unreliable** (high DNF rate)

    ---

    ### 🗺️ App Guide — Where to Go Next

    | Tab | What You Can Do |
    |-----|----------------|
    | **🎛️ Predict** | Slide the feature values and see the predicted position change in real-time |
    | **🔍 Explainability** | See *why* the model makes a prediction (SHAP waterfall charts) |
    | **🏁 Race Sim** | Pick a real race → see predicted results for ALL 20 drivers |
    | **👤 Drivers** | Explore a specific driver's race history and predictions |
    | **📋 Batch** | View model accuracy across the full dataset |

    ---

    *Ready? Head to the **🎛️ Predict** tab and start playing!* 🏎️💨
    """)


# ─────────────────────────────────────────────────
# TAB 1: Manual Prediction (original sidebar-driven)
# ─────────────────────────────────────────────────

with tab_predict:
    # -- Sidebar: Input Features --
    st.sidebar.header("🎛️ Input Features")

    grid_position = st.sidebar.slider("Grid Position", 1, 20, 5)

    st.sidebar.subheader("Driver Form")
    rolling_avg_finish_short = st.sidebar.slider(
        f"Rolling Avg Finish ({settings.rolling_window_short} races)", 1.0, 20.0, 6.0, 0.5
    )
    rolling_avg_finish_long = st.sidebar.slider(
        f"Rolling Avg Finish ({settings.rolling_window_long} races)", 1.0, 20.0, 7.0, 0.5
    )
    rolling_avg_points = st.sidebar.slider("Rolling Avg Points", 0.0, 25.0, 8.0, 0.5)
    position_delta_trend = st.sidebar.slider("Position Delta Trend", -10.0, 10.0, 0.5, 0.5)

    st.sidebar.subheader("Circuit History")
    circuit_avg_finish = st.sidebar.slider("Circuit Avg Finish", 1.0, 20.0, 8.0, 0.5)
    circuit_race_count = st.sidebar.slider("Races at Circuit", 0, 10, 3)

    st.sidebar.subheader("Team Performance")
    team_season_avg_finish = st.sidebar.slider("Team Season Avg Finish", 1.0, 20.0, 7.0, 0.5)
    team_points_per_race = st.sidebar.slider("Team Points/Race", 0.0, 25.0, 10.0, 0.5)

    st.sidebar.subheader("Reliability")
    dnf_rate_season = st.sidebar.slider("DNF Rate (Season)", 0.0, 1.0, 0.1, 0.05)
    dnf_rate_circuit = st.sidebar.slider("DNF Rate (Circuit)", 0.0, 1.0, 0.1, 0.05)

    st.sidebar.subheader("Qualifying")
    quali_position = st.sidebar.slider("Qualifying Position", 1, 20, 5)
    grid_quali_delta = grid_position - quali_position

    st.sidebar.subheader("Weather")
    air_temperature = st.sidebar.slider("Air Temperature (°C)", 10.0, 45.0, 25.0, 0.5)
    track_temperature = st.sidebar.slider("Track Temperature (°C)", 15.0, 60.0, 40.0, 0.5)
    is_wet_race = st.sidebar.selectbox("Wet Race?", [0, 1], format_func=lambda x: "Yes 🌧️" if x else "No ☀️")
    humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 55.0, 1.0)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0, 0.5)

    st.sidebar.subheader("Strategy & Circuit")
    n_pit_stops = st.sidebar.slider("Pit Stops", 0, 4, 1)
    is_street_circuit = st.sidebar.selectbox("Street Circuit?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    teammate_delta_rolling = st.sidebar.slider("Teammate Delta (rolling)", -10.0, 10.0, 0.0, 0.5)

    # -- Prediction --
    features = {
        "grid_position": float(grid_position),
        "rolling_avg_finish_short": rolling_avg_finish_short,
        "rolling_avg_finish_long": rolling_avg_finish_long,
        "rolling_avg_points": rolling_avg_points,
        "position_delta_trend": position_delta_trend,
        "circuit_avg_finish": circuit_avg_finish,
        "circuit_race_count": circuit_race_count,
        "team_season_avg_finish": team_season_avg_finish,
        "team_points_per_race": team_points_per_race,
        "dnf_rate_season": dnf_rate_season,
        "dnf_rate_circuit": dnf_rate_circuit,
        # Phase 1
        "quali_position": float(quali_position),
        "grid_quali_delta": float(grid_quali_delta),
        "air_temperature": air_temperature,
        "track_temperature": track_temperature,
        "is_wet_race": is_wet_race,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "n_pit_stops": n_pit_stops,
        "is_street_circuit": is_street_circuit,
        "teammate_delta_rolling": teammate_delta_rolling,
    }

    prediction = predict(features)[0]

    # -- Main Display --
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏁 Predicted Finish", f"P{prediction}")
    with col2:
        delta = grid_position - prediction
        st.metric("📊 Grid Position", f"P{grid_position}", delta=f"{delta:+d} positions")
    with col3:
        st.metric("📈 Positions Gained", f"{delta:+d}")

    st.divider()

    # -- Feature Importance --
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": importances,
            }
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            feat_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#16213e", "#E10600"],
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Feature Values")
        feat_df = pd.DataFrame([features]).T
        feat_df.columns = ["Value"]
        feat_df.index.name = "Feature"
        st.dataframe(feat_df, use_container_width=True)


# ─────────────────────────────────────────────────
# TAB 2: Explainability (SHAP + Confidence Intervals)
# ─────────────────────────────────────────────────

with tab_explain:
    st.subheader("🔍 Model Explainability")
    st.caption("Understand *why* the model predicts a specific position using SHAP values")

    features_path = settings.project_root / settings.data_processed_dir / "features.csv"

    if not features_path.exists():
        st.info("No processed data found. Run `python scripts/fetch_data.py` first.")
    else:
        explain_data = pd.read_csv(features_path)

        # ── SHAP Section ──
        st.markdown("### 🌊 SHAP Analysis")

        @st.cache_resource
        def get_explainer():
            from f1_predictor.explainability.shap_explainer import F1Explainer
            return F1Explainer(model=model)

        try:
            explainer = get_explainer()

            shap_mode = st.radio(
                "View",
                ["🌐 Global Importance", "🎯 Per-Prediction Explanation"],
                horizontal=True,
            )

            if shap_mode == "🌐 Global Importance":
                st.markdown(
                    "Shows how each feature impacts predictions across the entire dataset. "
                    "Each dot is one prediction — color = feature value, position = SHAP impact."
                )
                fig_beeswarm = explainer.get_beeswarm_figure(max_display=20)
                st.pyplot(fig_beeswarm)
                plt.close(fig_beeswarm)

            else:
                st.markdown(
                    "Explains *why* a specific prediction was made. "
                    "The waterfall shows how each feature pushes the prediction higher or lower."
                )

                explain_source = st.radio(
                    "Explain from",
                    ["Dataset row", "Current sidebar features"],
                    horizontal=True,
                )

                if explain_source == "Dataset row":
                    # Pick a driver/race to explain
                    if "full_name" in explain_data.columns:
                        driver_names = sorted(explain_data["full_name"].dropna().unique().tolist())
                        sel_driver = st.selectbox("Driver", driver_names, key="shap_driver")
                        driver_rows = explain_data[explain_data["full_name"] == sel_driver]

                        if "circuit_short_name" in driver_rows.columns:
                            circuits = driver_rows["circuit_short_name"].dropna().unique().tolist()
                            sel_circuit = st.selectbox("Circuit", circuits, key="shap_circuit")
                            row = driver_rows[driver_rows["circuit_short_name"] == sel_circuit].iloc[-1:]
                        else:
                            row = driver_rows.iloc[-1:]

                        if not row.empty:
                            idx = row.index[0]
                            actual = int(row["finishing_position"].iloc[0])
                            pred_val = predict(row[FEATURE_COLUMNS])[0]
                            st.info(f"**{sel_driver}** at **{sel_circuit}** — Actual: P{actual}, Predicted: P{pred_val}")

                            fig_wf = explainer.get_waterfall_figure(idx=idx)
                            st.pyplot(fig_wf)
                            plt.close(fig_wf)

                            # Top contributors
                            st.markdown("**Top 5 Contributing Features:**")
                            contributors = explainer.get_top_contributors(idx, n=5)
                            for fname, shap_val, fval in contributors:
                                direction = "⬆️ pushes higher" if shap_val > 0 else "⬇️ pushes lower"
                                st.write(f"- **{fname}** = {fval:.1f} → SHAP {shap_val:+.2f} ({direction})")
                    else:
                        st.warning("No driver names in dataset — select by index")
                        idx = st.number_input("Row index", 0, len(explain_data) - 1, 0)
                        fig_wf = explainer.get_waterfall_figure(idx=idx)
                        st.pyplot(fig_wf)
                        plt.close(fig_wf)

                else:
                    # Use current sidebar features
                    st.info(f"Explaining the current sidebar prediction: **P{prediction}**")
                    fig_wf = explainer.get_waterfall_figure(features=features)
                    st.pyplot(fig_wf)
                    plt.close(fig_wf)

        except Exception as e:
            st.error(f"SHAP failed: {e}")
            st.info("Make sure `shap` is installed: `pip install shap>=0.44.0`")

        st.divider()

        # ── Confidence Intervals Section ──
        st.markdown("### 📊 Confidence Intervals")
        st.caption("Prediction ranges from quantile regression (10th–90th percentile)")

        quantile_path = settings.project_root / settings.models_dir / "quantile_models.joblib"

        if not quantile_path.exists():
            st.info(
                "Quantile models not trained yet. Run:\n\n"
                "```bash\npython scripts/train_quantile.py\n```"
            )
        else:
            from f1_predictor.explainability.confidence import QuantilePredictor

            qp = QuantilePredictor.load()
            ci = qp.predict_single(features)

            col_ci1, col_ci2, col_ci3 = st.columns(3)
            with col_ci1:
                st.metric("⬇️ Optimistic (P10)", f"P{ci['lower']}")
            with col_ci2:
                st.metric("🎯 Median", f"P{ci['median']}")
            with col_ci3:
                st.metric("⬆️ Conservative (P90)", f"P{ci['upper']}")

            st.markdown(
                f"**Prediction range: P{ci['lower']}–P{ci['upper']}** "
                f"(80% confidence interval)"
            )


# ─────────────────────────────────────────────────
# TAB 3: Race Simulator
# ─────────────────────────────────────────────────

with tab_race_sim:
    st.subheader("🏁 Race Simulator")
    st.caption("Select a circuit to predict the full 20-driver race result — or pick a driver to auto-fill features from real data")

    features_path = settings.project_root / settings.data_processed_dir / "features.csv"

    if not features_path.exists():
        st.info("No processed data found. Run `python scripts/fetch_data.py` first.")
    else:
        sim_data = pd.read_csv(features_path)

        sim_mode = st.radio(
            "Mode",
            ["🏁 Full Race Simulation", "🎯 Auto-Fill Driver Prediction"],
            horizontal=True,
            key="sim_mode",
        )

        if sim_mode == "🏁 Full Race Simulation":
            st.markdown("#### Predict all drivers for a specific race")

            # Pick a session (circuit + date)
            if "circuit_short_name" in sim_data.columns and "session_key" in sim_data.columns:
                # Build race labels
                race_options = (
                    sim_data.groupby(["session_key", "circuit_short_name"])
                    .size()
                    .reset_index()
                    .rename(columns={0: "drivers"})
                )
                if "year" in sim_data.columns:
                    race_years = sim_data.groupby("session_key")["year"].first()
                    race_options = race_options.merge(
                        race_years, left_on="session_key", right_index=True, how="left"
                    )
                    race_options["label"] = (
                        race_options["circuit_short_name"]
                        + " "
                        + race_options["year"].astype(str)
                        + " ("
                        + race_options["drivers"].astype(str)
                        + " drivers)"
                    )
                else:
                    race_options["label"] = (
                        race_options["circuit_short_name"]
                        + " ("
                        + race_options["drivers"].astype(str)
                        + " drivers)"
                    )

                selected_race = st.selectbox(
                    "Select Race",
                    race_options["label"].tolist(),
                    key="sim_race",
                )

                # Get session_key for selected race
                sel_idx = race_options[race_options["label"] == selected_race].index[0]
                sel_sk = race_options.loc[sel_idx, "session_key"]

                race_rows = sim_data[sim_data["session_key"] == sel_sk].copy()

                if race_rows.empty:
                    st.warning("No data for selected race.")
                else:
                    # Predict
                    X_race = race_rows[FEATURE_COLUMNS]
                    preds = predict(X_race)
                    race_rows["predicted_position"] = preds

                    # Sort by predicted position
                    race_rows = race_rows.sort_values("predicted_position")

                    # Display columns
                    name_col = "full_name" if "full_name" in race_rows.columns else "driver_number"

                    st.markdown("#### 🏆 Predicted Race Result")

                    # Build result table
                    result_cols = [name_col, "team_name", "grid_position", "predicted_position"]
                    if "finishing_position" in race_rows.columns:
                        race_rows["actual"] = race_rows["finishing_position"].astype(int)
                        race_rows["delta"] = race_rows["actual"] - race_rows["predicted_position"]
                        result_cols += ["actual", "delta"]

                    display_df = race_rows[
                        [c for c in result_cols if c in race_rows.columns]
                    ].reset_index(drop=True)
                    display_df.index = display_df.index + 1  # 1-based
                    display_df.index.name = "Pos"

                    st.dataframe(display_df, use_container_width=True, height=500)

                    # Biggest movers chart
                    if "actual" in race_rows.columns:
                        st.markdown("#### 📊 Predicted vs Actual")
                        chart_df = race_rows[[name_col, "predicted_position", "actual"]].copy()
                        chart_df = chart_df.melt(
                            id_vars=[name_col],
                            value_vars=["predicted_position", "actual"],
                            var_name="Type",
                            value_name="Position",
                        )
                        chart_df["Type"] = chart_df["Type"].map(
                            {"predicted_position": "Predicted", "actual": "Actual"}
                        )

                        fig_sim = px.bar(
                            chart_df,
                            x=name_col,
                            y="Position",
                            color="Type",
                            barmode="group",
                            color_discrete_map={"Predicted": "#E10600", "Actual": "#16213e"},
                        )
                        fig_sim.update_yaxes(autorange="reversed")
                        fig_sim.update_layout(height=500, xaxis_tickangle=-45)
                        st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.warning("Dataset missing circuit or session data.")

        else:
            # ── Auto-Fill Driver Prediction ──
            st.markdown("#### Pick a driver + circuit to auto-fill features from historical data")

            name_col = "full_name" if "full_name" in sim_data.columns else "driver_number"
            all_drivers = sorted(sim_data[name_col].dropna().unique().tolist())

            sel_driver = st.selectbox("Driver", all_drivers, key="autofill_driver")

            driver_rows = sim_data[sim_data[name_col] == sel_driver]

            if "circuit_short_name" in driver_rows.columns:
                circuits = sorted(driver_rows["circuit_short_name"].dropna().unique().tolist())
                all_circuits = sorted(sim_data["circuit_short_name"].dropna().unique().tolist())
                sel_circuit = st.selectbox(
                    "Circuit",
                    all_circuits,
                    index=0,
                    key="autofill_circuit",
                )

                # Get latest data for this driver at this circuit (or latest overall)
                circuit_rows = driver_rows[driver_rows["circuit_short_name"] == sel_circuit]
                if not circuit_rows.empty:
                    latest = circuit_rows.iloc[-1]
                    source_label = f"{sel_driver} at {sel_circuit}"
                else:
                    latest = driver_rows.iloc[-1]
                    source_label = f"{sel_driver} (latest race — no data at {sel_circuit})"

                st.info(f"Features auto-filled from: **{source_label}**")

                # Extract features and predict
                auto_features = {col: float(latest[col]) for col in FEATURE_COLUMNS if col in latest.index}

                auto_pred = predict(auto_features)[0]

                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.metric("🏁 Predicted Finish", f"P{auto_pred}")
                with col_a2:
                    grid = int(auto_features.get("grid_position", 0))
                    st.metric("📊 Grid Position", f"P{grid}")
                with col_a3:
                    if "finishing_position" in latest.index:
                        actual = int(latest["finishing_position"])
                        st.metric("✅ Actual Result", f"P{actual}")

                # Show the auto-filled feature values
                with st.expander("📋 Auto-filled feature values"):
                    feat_auto_df = pd.DataFrame([auto_features]).T
                    feat_auto_df.columns = ["Value"]
                    feat_auto_df.index.name = "Feature"
                    st.dataframe(feat_auto_df, use_container_width=True)
            else:
                st.warning("No circuit data available for auto-fill.")


# ─────────────────────────────────────────────────
# TAB 2: Driver Explorer
# ─────────────────────────────────────────────────

with tab_driver:
    st.subheader("🏎️ Driver Explorer")
    st.caption("Filter the dataset by driver to see their race history and predictions")

    features_path = settings.project_root / settings.data_processed_dir / "features.csv"

    if not features_path.exists():
        st.info("No processed data found. Run `python scripts/fetch_data.py` first.")
    else:
        data = pd.read_csv(features_path)

        # -- Driver filter --
        driver_col = "full_name" if "full_name" in data.columns else None

        if driver_col:
            all_drivers = sorted(data[driver_col].dropna().unique().tolist())
            selected_drivers = st.multiselect(
                "Select Driver(s)",
                options=all_drivers,
                default=all_drivers[:3] if len(all_drivers) >= 3 else all_drivers,
                help="Pick one or more drivers to explore",
            )

            if selected_drivers:
                driver_data = data[data[driver_col].isin(selected_drivers)].copy()
            else:
                driver_data = data.copy()
        else:
            # Fallback: filter by driver_number
            if "driver_number" in data.columns:
                all_numbers = sorted(data["driver_number"].unique().tolist())
                selected_numbers = st.multiselect(
                    "Select Driver Number(s)",
                    options=all_numbers,
                    default=all_numbers[:3] if len(all_numbers) >= 3 else all_numbers,
                )
                driver_data = data[data["driver_number"].isin(selected_numbers)].copy()
            else:
                driver_data = data.copy()

        if driver_data.empty:
            st.warning("No data for selected driver(s).")
        else:
            # -- Circuit filter (optional) --
            if "circuit_short_name" in driver_data.columns:
                circuits = ["All Circuits"] + sorted(
                    driver_data["circuit_short_name"].dropna().unique().tolist()
                )
                selected_circuit = st.selectbox("Filter by Circuit", circuits)
                if selected_circuit != "All Circuits":
                    driver_data = driver_data[
                        driver_data["circuit_short_name"] == selected_circuit
                    ]

            # -- Run predictions --
            X = driver_data[FEATURE_COLUMNS]
            preds = predict(X)
            driver_data["predicted_position"] = preds
            driver_data["error"] = (
                driver_data["finishing_position"].astype(int) - preds
            )

            # -- Summary metrics per driver --
            st.markdown("### 📊 Driver Summary")

            if driver_col and driver_col in driver_data.columns:
                group_col = driver_col
            else:
                group_col = "driver_number"

            summary = (
                driver_data.groupby(group_col)
                .agg(
                    races=("finishing_position", "count"),
                    avg_grid=("grid_position", "mean"),
                    avg_finish=("finishing_position", "mean"),
                    avg_predicted=("predicted_position", "mean"),
                    avg_error=("error", "mean"),
                    best_finish=("finishing_position", "min"),
                    dnf_rate=("dnf_rate_season", "mean"),
                )
                .round(2)
                .sort_values("avg_finish")
            )
            st.dataframe(summary, use_container_width=True)

            # -- Race-by-race detail --
            st.markdown("### 📋 Race-by-Race Results")

            display_cols = [
                c
                for c in [
                    driver_col or "driver_number",
                    "team_name",
                    "circuit_short_name",
                    "grid_position",
                    "finishing_position",
                    "predicted_position",
                    "error",
                    "rolling_avg_finish_short",
                    "rolling_avg_points",
                    "dnf_rate_season",
                ]
                if c in driver_data.columns
            ]
            st.dataframe(
                driver_data[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=400,
            )

            # -- Finish position over races chart --
            st.markdown("### 📈 Position Trend")

            if driver_col and driver_col in driver_data.columns:
                chart_data = driver_data.copy()
                chart_data["race_index"] = chart_data.groupby(driver_col).cumcount() + 1

                fig_trend = px.line(
                    chart_data,
                    x="race_index",
                    y="finishing_position",
                    color=driver_col,
                    markers=True,
                    title="Finishing Position Over Races",
                    labels={
                        "race_index": "Race #",
                        "finishing_position": "Finish Position",
                    },
                )
                fig_trend.update_yaxes(autorange="reversed")  # P1 at top
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)

                # Predicted vs actual scatter per driver
                fig_scatter = px.scatter(
                    chart_data,
                    x="finishing_position",
                    y="predicted_position",
                    color=driver_col,
                    title="Predicted vs Actual (by Driver)",
                    labels={
                        "finishing_position": "Actual Position",
                        "predicted_position": "Predicted Position",
                    },
                )
                fig_scatter.add_shape(
                    type="line",
                    x0=0, y0=0, x1=20, y1=20,
                    line=dict(dash="dash", color="gray"),
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────────────────────────────
# TAB 3: Batch Predictions
# ─────────────────────────────────────────────────

with tab_batch:
    st.subheader("📋 Batch Predictions")

    features_path = settings.project_root / settings.data_processed_dir / "features.csv"

    if not features_path.exists():
        st.info(
            "No processed data found. Run `python scripts/fetch_data.py` to fetch race data."
        )
    else:
        data = pd.read_csv(features_path)

        X = data[FEATURE_COLUMNS]
        preds = predict(X)
        data["predicted_position"] = preds
        data["prediction_error"] = data["finishing_position"].astype(int) - preds

        # Summary stats
        from sklearn.metrics import mean_absolute_error

        mae = mean_absolute_error(data["finishing_position"], preds)
        baseline_mae = mean_absolute_error(
            data["finishing_position"], data["grid_position"]
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model MAE", f"{mae:.2f}")
        with col2:
            st.metric("Baseline MAE", f"{baseline_mae:.2f}")
        with col3:
            improvement = (baseline_mae - mae) / baseline_mae * 100
            st.metric("Improvement", f"{improvement:.1f}%")

        # Full table
        display_cols = [
            c
            for c in [
                "full_name",
                "team_name",
                "circuit_short_name",
                "grid_position",
                "finishing_position",
                "predicted_position",
                "prediction_error",
            ]
            if c in data.columns
        ]

        st.dataframe(
            data[display_cols],
            use_container_width=True,
            height=500,
        )

# -- Footer --
st.divider()
st.caption("Built with XGBoost + Streamlit | Data from OpenF1 API")
