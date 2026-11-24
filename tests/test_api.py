"""Pytest API tests for the ventilator fault-detection FastAPI backend.

All tests use ``with TestClient(app) as client:`` so FastAPI's lifespan
(startup / shutdown) events execute correctly, loading the real model and
preprocessing artifacts from ``artifacts/``.

Test categories
---------------
1. Health endpoint — liveness and schema correctness
2. POST /predict — valid windows (normal & fault-like)
3. POST /predict — validation failures → HTTP 422
4. GET /predictions/history — persistence and filtering
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

# Point the API at a temporary DB so test runs don't contaminate the real one
_TMP_DB = tempfile.mktemp(suffix=".db", prefix="test_predictions_")
os.environ["PREDICTIONS_DB"] = _TMP_DB

from src.api import app  # noqa: E402  (import after env-var is set)

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

WINDOW_SIZE = 20  # must match metadata.json

# A single valid, physically realistic "normal" sensor reading
NORMAL_READING: Dict[str, Any] = {
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
    "device_model": "VENT-X100",
}

# A single reading engineered to look fault-like (high temp, vibration spike)
FAULT_READING: Dict[str, Any] = {
    "operating_hours": 50000.0,
    "inlet_pressure_kPa": 45.0,
    "airway_pressure_cmH2O": 100.0,
    "airflow_L_min": 5.0,
    "oxygen_concentration_pct": 20.0,
    "internal_temperature_C": 115.0,
    "supply_voltage_V": 480.0,
    "motor_current_A": 18.0,
    "blower_speed_rpm": 14500.0,
    "battery_voltage_V": 48.0,
    "vibration_mm_s": 45.0,
    "filter_pressure_drop_kPa": 18.0,
    "ambient_humidity_pct": 95.0,
    "device_model": "VENT-M500",
}


def _window(reading: Dict[str, Any], size: int = WINDOW_SIZE) -> List[Dict[str, Any]]:
    """Repeat a single reading ``size`` times to form a window."""
    return [reading.copy() for _ in range(size)]


def _predict_payload(reading: Dict[str, Any], device_id: str = "TEST-001", size: int = WINDOW_SIZE) -> Dict[str, Any]:
    return {"device_id": device_id, "readings": _window(reading, size)}


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health must respond with HTTP 200."""
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self):
        """model_loaded must be True when artifacts are present."""
        with TestClient(app) as client:
            resp = client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True

    def test_health_schema_fields(self):
        """Health response must include window_size, num_features, feature_names."""
        with TestClient(app) as client:
            resp = client.get("/health")
        data = resp.json()
        assert "window_size" in data, "window_size missing from health response"
        assert "num_features" in data, "num_features missing from health response"
        assert "feature_names" in data, "feature_names missing from health response"
        assert isinstance(data["feature_names"], list)
        assert len(data["feature_names"]) > 0

    def test_health_window_size_matches_metadata(self):
        """window_size in health response must equal the value in metadata.json."""
        import json
        with open("artifacts/metadata.json") as fh:
            meta = json.load(fh)
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.json()["window_size"] == meta["window_size"]

    def test_health_status_field(self):
        """status field must be 'ok' when model is loaded."""
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. POST /predict — valid windows
# ---------------------------------------------------------------------------

class TestPredictValid:
    def test_predict_normal_window_200(self):
        """POST /predict with a valid normal-pattern window must return 200."""
        with TestClient(app) as client:
            resp = client.post("/predict", json=_predict_payload(NORMAL_READING))
        assert resp.status_code == 200

    def test_predict_fault_window_200(self):
        """POST /predict with a fault-pattern window must return 200."""
        with TestClient(app) as client:
            resp = client.post("/predict", json=_predict_payload(FAULT_READING, device_id="FAULT-DEV"))
        assert resp.status_code == 200

    def test_predict_probability_in_range(self):
        """fault_probability must be a float in [0, 1]."""
        with TestClient(app) as client:
            resp = client.post("/predict", json=_predict_payload(NORMAL_READING))
        prob = resp.json()["fault_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_label_is_valid(self):
        """predicted_label must be exactly 'Normal' or 'Fault'."""
        with TestClient(app) as client:
            resp = client.post("/predict", json=_predict_payload(NORMAL_READING))
        label = resp.json()["predicted_label"]
        assert label in {"Normal", "Fault"}

    def test_predict_response_schema(self):
        """Response must contain all required fields with correct types."""
        with TestClient(app) as client:
            resp = client.post("/predict", json=_predict_payload(NORMAL_READING))
        data = resp.json()
        required = {
            "prediction_id", "device_id", "fault_probability",
            "predicted_label", "threshold", "confidence",
            "window_size_received", "timestamp",
        }
        assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"

    def test_predict_device_id_echoed(self):
        """device_id in response must match the request."""
        payload = _predict_payload(NORMAL_READING, device_id="ECHO-99")
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.json()["device_id"] == "ECHO-99"

    def test_predict_window_size_received(self):
        """window_size_received must equal the number of readings sent."""
        with TestClient(app) as client:
            resp = client.post("/predict", json=_predict_payload(NORMAL_READING, size=WINDOW_SIZE))
        assert resp.json()["window_size_received"] == WINDOW_SIZE

    def test_predict_confidence_in_range(self):
        """confidence must be a float in [0, 1]."""
        with TestClient(app) as client:
            resp = client.post("/predict", json=_predict_payload(NORMAL_READING))
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 3. POST /predict — validation failures → 422
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_wrong_window_length_too_short(self):
        """Sending fewer than window_size readings must return 422."""
        payload = _predict_payload(NORMAL_READING, size=5)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_wrong_window_length_too_long(self):
        """Sending more than window_size readings must return 422."""
        payload = _predict_payload(NORMAL_READING, size=WINDOW_SIZE + 5)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_required_field(self):
        """A reading missing a required sensor field must return 422."""
        bad_reading = NORMAL_READING.copy()
        del bad_reading["airflow_L_min"]
        payload = _predict_payload(bad_reading)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_nan_value_rejected(self):
        """A NaN value sent as a string 'NaN' in any sensor field must return 422.

        Standard JSON does not allow bare NaN; we send it as the string "NaN"
        which Pydantic v2's float coercion will reject with a validation error.
        """
        bad_reading = NORMAL_READING.copy()
        bad_reading["internal_temperature_C"] = "NaN"  # string NaN → Pydantic rejects
        payload = _predict_payload(bad_reading)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_inf_value_rejected(self):
        """An extremely large value that is out of bounds must return 422.

        Standard JSON does not support Infinity.  We use a value that exceeds
        the physical upper bound for motor_current_A (20 A) to verify the
        bounds validator fires and returns 422.
        """
        bad_reading = NORMAL_READING.copy()
        bad_reading["motor_current_A"] = 999_999.0  # Way beyond [0, 20] → 422
        payload = _predict_payload(bad_reading)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_pressure_rejected(self):
        """A negative inlet_pressure_kPa (physically impossible) must return 422."""
        bad_reading = NORMAL_READING.copy()
        bad_reading["inlet_pressure_kPa"] = -5.0
        payload = _predict_payload(bad_reading)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_oxygen_above_100_rejected(self):
        """oxygen_concentration_pct > 100 must return 422."""
        bad_reading = NORMAL_READING.copy()
        bad_reading["oxygen_concentration_pct"] = 105.0
        payload = _predict_payload(bad_reading)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_oxygen_below_min_rejected(self):
        """oxygen_concentration_pct < 15 (physical minimum) must return 422."""
        bad_reading = NORMAL_READING.copy()
        bad_reading["oxygen_concentration_pct"] = 5.0
        payload = _predict_payload(bad_reading)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_unknown_device_model_rejected(self):
        """An unknown device_model must return 422."""
        bad_reading = NORMAL_READING.copy()
        bad_reading["device_model"] = "UNKNOWN-9999"
        payload = _predict_payload(bad_reading)
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_empty_device_id_rejected(self):
        """An empty device_id string must return 422."""
        payload = _predict_payload(NORMAL_READING)
        payload["device_id"] = ""
        with TestClient(app) as client:
            resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_readings_key_rejected(self):
        """A payload with no 'readings' key must return 422."""
        with TestClient(app) as client:
            resp = client.post("/predict", json={"device_id": "X"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. GET /predictions/history — persistence & filtering
# ---------------------------------------------------------------------------

class TestPredictionsHistory:
    """All tests in this class share a single TestClient session so records
    inserted in one test are visible in later tests."""

    def test_history_returns_200(self):
        """GET /predictions/history must return HTTP 200."""
        with TestClient(app) as client:
            resp = client.get("/predictions/history")
        assert resp.status_code == 200

    def test_history_schema(self):
        """History response must have 'count' and 'predictions' keys."""
        with TestClient(app) as client:
            resp = client.get("/predictions/history")
        data = resp.json()
        assert "count" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)

    def test_prediction_persisted_after_post(self):
        """A prediction POSTed must appear in /predictions/history."""
        device = "PERSIST-TEST-01"
        payload = _predict_payload(NORMAL_READING, device_id=device)
        with TestClient(app) as client:
            post_resp = client.post("/predict", json=payload)
            assert post_resp.status_code == 200
            prediction_id = post_resp.json()["prediction_id"]

            hist_resp = client.get(f"/predictions/history?device_id={device}")
        assert hist_resp.status_code == 200
        ids = [p["prediction_id"] for p in hist_resp.json()["predictions"]]
        assert prediction_id in ids, "POSTed prediction not found in history"

    def test_history_device_id_filter(self):
        """device_id query param must filter results to that device only."""
        device_a = "DEVICE-AAA"
        device_b = "DEVICE-BBB"
        with TestClient(app) as client:
            client.post("/predict", json=_predict_payload(NORMAL_READING, device_id=device_a))
            client.post("/predict", json=_predict_payload(NORMAL_READING, device_id=device_b))

            resp = client.get(f"/predictions/history?device_id={device_a}&limit=100")
        data = resp.json()
        for record in data["predictions"]:
            assert record["device_id"] == device_a, (
                f"Expected device_id={device_a}, got {record['device_id']}"
            )

    def test_history_limit_param(self):
        """limit query param must cap the number of records returned."""
        with TestClient(app) as client:
            # Insert several records
            for i in range(5):
                client.post("/predict", json=_predict_payload(NORMAL_READING, device_id=f"LIMIT-{i}"))

            resp = client.get("/predictions/history?limit=2")
        data = resp.json()
        assert len(data["predictions"]) <= 2

    def test_history_count_matches_predictions_length(self):
        """'count' field must equal len(predictions)."""
        with TestClient(app) as client:
            resp = client.get("/predictions/history?limit=10")
        data = resp.json()
        assert data["count"] == len(data["predictions"])

    def test_history_record_fields(self):
        """Each history record must contain required fields."""
        with TestClient(app) as client:
            client.post("/predict", json=_predict_payload(NORMAL_READING, device_id="SCHEMA-CHECK"))
            resp = client.get("/predictions/history?device_id=SCHEMA-CHECK&limit=1")
        records = resp.json()["predictions"]
        assert len(records) >= 1
        required = {"prediction_id", "device_id", "fault_probability", "predicted_label", "timestamp"}
        assert required.issubset(records[0].keys())
