# Spec: Ventilator Fault Detection — FastAPI Backend — V1

## Objective

Expose the trained LSTM fault-detection model and fitted preprocessing
artifacts as a production-ready REST API so downstream dashboards, edge
devices, or clinical operator consoles can submit raw sensor windows and
receive structured fault-probability predictions without any ML dependencies
on the client side.

**Users:** Streamlit dashboard (same host), clinical device monitoring
integrations, automated testing harnesses.

**Success:** A `uvicorn`-hosted server loads artifacts once at startup, serves
three endpoints (`/health`, `/POST /predict`, `GET /predictions/history`),
validates inputs strictly with Pydantic v2, persists every prediction to
SQLite, and all 4+ `pytest` API tests pass.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Web framework | FastAPI 0.111+ | Async-first, Pydantic v2 native, OpenAPI auto-docs |
| Validation | Pydantic v2 `BaseModel` | Field-level validators, `model_validator`, strict mode |
| ASGI server | Uvicorn[standard] | Production-grade, asyncio |
| HTTP test client | HTTPX + `TestClient` | Lifespan-aware starlette test client |
| Persistence | SQLite (`aiosqlite` or stdlib `sqlite3`) | Zero-ops, file-based, queryable |
| ML runtime | TensorFlow/Keras (already installed in .venv) | Loads `best_lstm_model.keras` |
| Preprocessing | `src/preprocessing.py::VentilatorPreprocessor` | Reuse fitted scaler/encoder |

---

## Commands

```
# Install
.\.venv\Scripts\pip.exe install "fastapi[standard]" uvicorn[standard] httpx aiosqlite

# Run server
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

# API tests only
.\.venv\Scripts\pytest.exe tests/test_api.py -v

# Full suite
.\.venv\Scripts\pytest.exe -v
```

---

## Project Structure

```
src/
  api.py          ← FastAPI app: lifespan, routes, SQLite persistence
  schemas.py      ← Pydantic v2 request/response models + physical-bound validators
  preprocessing.py (existing)
artifacts/
  best_lstm_model.keras
  scaler.joblib
  encoder.joblib
  metadata.json
  predictions.db  ← Created at first startup
tests/
  test_api.py     ← Pytest API tests (TestClient, lifespan-aware)
```

---

## Code Style

```python
# Physical-bound validator example (Pydantic v2)
from pydantic import BaseModel, field_validator, model_validator
from typing import Annotated
import math

class SensorReading(BaseModel):
    operating_hours: float
    inlet_pressure_kPa: float
    # ...

    @model_validator(mode="after")
    def check_no_nan_inf(self) -> "SensorReading":
        for field, val in self.model_dump().items():
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(f"{field} must not be NaN or Inf")
        return self

# Lifespan pattern (FastAPI ≥ 0.93)
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load model + artifacts into app.state
    yield
    # shutdown: release resources
```

---

## API Contract

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "best_lstm_model",
  "window_size": 20,
  "num_features": 16,
  "feature_names": ["operating_hours", "..."],
  "timestamp": "2026-11-24T13:00:00Z"
}
```

### `POST /predict`

**Request body:**
```json
{
  "device_id": "DEV-001",
  "readings": [
    {
      "operating_hours": 1500.0,
      "inlet_pressure_kPa": 10.5,
      "airway_pressure_cmH2O": 20.0,
      "airflow_L_min": 60.0,
      "oxygen_concentration_pct": 95.0,
      "internal_temperature_C": 38.5,
      "supply_voltage_V": 220.0,
      "motor_current_A": 3.5,
      "blower_speed_rpm": 3000.0,
      "battery_voltage_V": 12.6,
      "vibration_mm_s": 1.2,
      "filter_pressure_drop_kPa": 0.5,
      "ambient_humidity_pct": 55.0,
      "device_model": "VENT-X100"
    }
    // ... window_size readings total
  ]
}
```

**Response (200 OK):**
```json
{
  "device_id": "DEV-001",
  "fault_probability": 0.012,
  "predicted_label": "Normal",
  "threshold": 0.5,
  "confidence": 0.988,
  "window_size_received": 20,
  "prediction_id": "uuid-...",
  "timestamp": "2026-11-24T13:00:01Z"
}
```

**Validation errors → 422 Unprocessable Entity** for:
- Wrong number of readings (not exactly `window_size`)
- Missing required sensor fields
- NaN / Inf values
- Out-of-physical-bounds values (e.g. negative pressure, O₂ > 100%)

### `GET /predictions/history`

Query params: `device_id` (optional), `limit` (default 50, max 500)

```json
{
  "count": 3,
  "predictions": [
    {
      "prediction_id": "uuid-...",
      "device_id": "DEV-001",
      "fault_probability": 0.012,
      "predicted_label": "Normal",
      "timestamp": "..."
    }
  ]
}
```

---

## Testing Strategy

- Framework: `pytest` with `starlette.testclient.TestClient`
- Use `with TestClient(app) as client:` so lifespan events fire
- Tests live in `tests/test_api.py`
- Tests are **medium** size (spawn real lifespan, real model load, SQLite in-memory or temp file)
- Minimum 10 test functions covering:
  1. Health endpoint returns 200 and `model_loaded: true`
  2. Health schema has `window_size`, `num_features`, `feature_names`
  3. POST /predict with valid normal window → 200, probability in [0,1]
  4. POST /predict with valid fault-like window → 200, probability in [0,1]
  5. POST /predict with wrong number of readings → 422
  6. POST /predict with missing field → 422
  7. POST /predict with NaN value → 422
  8. POST /predict with out-of-bounds value → 422
  9. GET /predictions/history returns persisted record after POST
  10. GET /predictions/history with device_id filter
  11. GET /predictions/history with limit param

---

## Boundaries

- **Always:** Use `app.state` for model/artifact storage (never module-level globals mutable after startup); return structured JSON errors; validate before inference.
- **Ask first:** Switching to PostgreSQL, adding auth/JWT, adding async TF inference.
- **Never:** Load model on every request; expose raw traceback in 500 responses; skip validation; fabricate predictions.

---

## Success Criteria

- [ ] `GET /health` returns 200 with `model_loaded: true` and correct `window_size`/`num_features`
- [ ] `POST /predict` with a valid 20-reading window returns `fault_probability` in [0, 1] and `predicted_label` in {"Normal", "Fault"}
- [ ] Invalid inputs return HTTP 422 with structured error detail
- [ ] Every prediction is persisted to SQLite `artifacts/predictions.db`
- [ ] `GET /predictions/history` returns persisted records, filterable by `device_id` and `limit`
- [ ] All `pytest tests/test_api.py` tests pass
- [ ] Full `pytest` suite (all existing 21+ tests + new API tests) passes
- [ ] Server starts cleanly with `uvicorn src.api:app --port 8000`

---

## Out of Scope (Deferred — V2)

- JWT/OAuth2 authentication and rate limiting
- Async TF inference with `asyncio.run_in_executor`
- Prometheus metrics endpoint (`/metrics`)
- Batch CSV upload endpoint
- WebSocket real-time streaming
- Docker / containerization
- PostgreSQL migration (expand/contract)
- Alerting webhook fan-out on fault detection

---

## Open Questions

_None — all requirements are concrete and grounded in existing artifacts._
