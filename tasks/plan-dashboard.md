# Plan: Streamlit Dashboard Implementation

## Steps

1. **Dashboard Client (`src/dashboard_client.py`)**:
   - Implement `VentilatorDashboardClient`:
     * `check_health(base_url)` -> Health status or offline fallback dict with `demo_mode=True`.
     * `predict(base_url, device_id, readings)` -> calls `POST /predict`, falls back to local heuristic/demo prediction on error.
     * `get_history(base_url, device_id, limit)` -> calls `GET /predictions/history`, falls back to sampled history on error.
     * `load_demo_data()` -> loads `synthetic_ventilator_equipment_dataset.csv`, caches it, extracts device list, and provides sample normal/fault windows.
     * `get_model_evaluation_metrics()` -> loads `artifacts/evaluation_metrics.json`.

2. **Streamlit UI Application (`app.py`)**:
   - Clinical design with page config, sidebar status, custom CSS styling for high-contrast medical telemetry.
   - Tab 1: Device Overview & Fleet Status
   - Tab 2: Live Telemetry Charts (Plotly interactive charts with normal bands)
   - Tab 3: Fault Detection & Alert Center (Interactive Gauge, episode selector, inference runner)
   - Tab 4: Model Diagnostics & Performance (Metrics table, interactive confusion matrix, training plots)
   - Tab 5: Maintenance & Audit Logs (History table with CSV export)

3. **Automated Pytest Suite (`tests/test_dashboard.py`)**:
   - Test client under online mode (mocked 200).
   - Test client under offline mode (connection error / timeout / 422 fallback).
   - Test `load_demo_data()` returns valid windows.
   - Test Streamlit `AppTest` rendering `app.py` end-to-end without unhandled exceptions.
