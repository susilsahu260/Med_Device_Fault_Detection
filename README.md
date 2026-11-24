# PulmoGuard ICU - Intelligent Ventilator Telemetry & Fault Prediction

End-to-end predictive maintenance and real-time clinical telemetry surveillance for 100 ICU ventilators (VENT-X100, VENT-X200, VENT-M500) across 5 fault categories (sensor_drift, overheating, airflow_restriction, blower_fault, power_supply_fault).

## Architecture & Performance
- Model: 20-step sliding-window LSTM (best_lstm_model.keras) across 14 sensor features (PR-AUC: 0.9978, Test F1-Score: 0.9939, Decision Threshold: 0.50).
- Backend (src/api.py): FastAPI server on port 8000 (/health, /predict, /predictions/history) with Pydantic schema validation (src/schemas.py) and SQLite persistence (artifacts/predictions.db).
- Dashboard (app.py): Streamlit UI on port 8501 with Fleet Overview, Live Telemetry, Fault Alert Center (live CSV upload), Model Diagnostics, and Maintenance History.

## Local Setup & Run
1. Backend: .\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
2. Dashboard: .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
