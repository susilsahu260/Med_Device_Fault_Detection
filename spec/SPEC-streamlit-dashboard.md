# Spec: Ventilator Fault Detection Streamlit Dashboard — V1

## Objective

Build a clinical-grade, resilient Streamlit monitoring dashboard (`app.py` and `src/dashboard_client.py`) for ventilator predictive equipment maintenance. It connects to the FastAPI backend (`http://127.0.0.1:8000`) for real-time inference and history, with automatic, transparent fallback to labeled `[DEMO DATA]` sampled from `synthetic_ventilator_equipment_dataset.csv` whenever the backend is offline, unreachable, or errors.

## Tech Stack

- **Dashboard Framework**: Streamlit (≥ 1.35)
- **Visualizations**: Plotly Express & Graph Objects (clinical-grade, responsive, dark/light clinical palettes)
- **HTTP Client**: `requests` or `httpx` (timeout-guarded, session-based)
- **Backend**: FastAPI REST API (`src/api.py`, `src/schemas.py`)
- **Testing**: `pytest`, `unittest.mock`, `streamlit.testing.v1.AppTest`

## Architecture & Data Flow

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit (app.py)                   │
│   Fleet Status │ Live Charts │ Fault Center │ History  │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────▼─────────────┐
               │ src/dashboard_client.py  │
               └──────┬────────────┬──────┘
                      │            │
          (Online)    │            │  (Offline / Error)
                      ▼            ▼
               FastAPI Backend   Fallback Demo Provider
               (127.0.0.1:8000)  (CSV / Artifacts)
```

## Commands

```bash
# Launch dashboard
streamlit run app.py --server.port 8501 --server.headless true

# Run dashboard tests
pytest tests/test_dashboard.py -v

# Run full test suite
pytest -v
```

## Project Structure

```
src/
  dashboard_client.py     ← API client with health, predict, history & resilient demo fallbacks
app.py                    ← Clinical-grade Streamlit application
tests/
  test_dashboard.py       ← Pytest test suite with mocked API and Streamlit AppTest
```

## Required Sections & Features

1. **Sidebar Controls**:
   - Backend URL configuration (default `http://127.0.0.1:8000`)
   - Connection status indicator (`[LIVE API]` vs `[DEMO MODE]`)
   - Manual Refresh / Reconnect button
   - Device selector dropdown

2. **Fleet Status & Overview**:
   - KPI metrics cards: Monitored Devices, API Liveness, Active Fault Rate, Model PR-AUC (`0.9978`), F1 (`0.9939`).
   - Fleet-wide status breakdown.

3. **Live Sensor Charts (Plotly)**:
   - Synchronized time-series charts for:
     - Airway & Inlet Pressures (`airway_pressure_cmH2O`, `inlet_pressure_kPa`) with safe clinical bands [10-35 cmH2O]
     - Airflow (`airflow_L_min`) & Oxygen (`oxygen_concentration_pct`)
     - Motor Current (`motor_current_A`), Blower Speed (`blower_speed_rpm`), Vibration (`vibration_mm_s`)
     - Internal Temperature (`internal_temperature_C`) & Voltage (`supply_voltage_V`)
   - Clear `[LIVE API]` vs `[DEMO DATA]` badge on charts.

4. **Fault Detection & Alert Center**:
   - Sample sequence selector (Normal episode vs Fault episode from dataset) or CSV upload.
   - Run Prediction button sending sequential 20-step window.
   - Interactive Plotly gauge for fault probability `[0.0 - 1.0]`.
   - Alert banner: Green "Normal Operation" vs Red/Amber "CRITICAL FAULT DETECTED: [Action Required]".

5. **Model Analytics & Diagnostics**:
   - Actual test set metrics: Accuracy 0.9959, Precision 1.000, Recall 0.9878, F1 0.9939, ROC-AUC 0.9985, PR-AUC 0.9978, Cohen's Kappa 0.9908.
   - Interactive Plotly Confusion Matrix (`[[1815, 0], [11, 889]]`).
   - Image viewers for saved artifacts (`artifacts/roc_curve.png`, `artifacts/precision_recall_curve.png`, `artifacts/learning_curves.png`).

6. **Maintenance & Prediction History**:
   - Table of predictions fetched from `GET /predictions/history` (or persisted/demo rows).
   - Filter by device ID, search table, and CSV download button.

## Boundaries & Constraints

- **Never** crash the Streamlit dashboard on backend network errors (`ConnectError`, `HTTPError`, `ReadTimeout`, `422`, `500`).
- **Always** visually distinguish `[LIVE API]` data from `[DEMO DATA]`.
- **Always** validate incoming or sampled sequence shapes (`window_size=20`, `num_features=16`).
