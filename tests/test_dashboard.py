"""Unit and integration tests for the Streamlit dashboard and its client.

Tests cover:
1. Dashboard client online behaviour (mocked FastAPI 200 OK).
2. Dashboard client offline & error handling with transparent [DEMO DATA] fallback.
3. Dataset loading and 20-step window extraction.
4. Model evaluation metrics retrieval.
5. Streamlit AppTest smoke test confirming app.py renders end-to-end without unhandled exceptions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.dashboard_client import VentilatorDashboardClient


# ---------------------------------------------------------------------------
# 1. Dashboard Client Tests - Online Mode (Mocked 200 OK)
# ---------------------------------------------------------------------------
class TestDashboardClientOnline:
    @patch("requests.get")
    def test_check_health_online(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "ok",
            "model_loaded": True,
            "model_name": "best_lstm_model",
            "window_size": 20,
            "num_features": 16,
            "feature_names": ["sensor1", "sensor2"],
        }
        mock_get.return_value = mock_resp

        client = VentilatorDashboardClient(base_url="http://test-server:8000")
        health = client.check_health()

        assert health["is_online"] is True
        assert health["source"] == "[LIVE API]"
        assert health["model_name"] == "best_lstm_model"
        assert health["window_size"] == 20

    @patch("requests.post")
    def test_predict_online(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "prediction_id": "test-uuid",
            "device_id": "VENT-001",
            "fault_probability": 0.025,
            "predicted_label": "Normal",
            "threshold": 0.5,
            "confidence": 0.95,
            "window_size_received": 20,
        }
        mock_post.return_value = mock_resp

        client = VentilatorDashboardClient(base_url="http://test-server:8000")
        sample_readings = [{"inlet_pressure_kPa": 10.0}] * 20
        res = client.predict(device_id="VENT-001", readings=sample_readings)

        assert res["source"] == "[LIVE API]"
        assert res["predicted_label"] == "Normal"
        assert res["fault_probability"] == 0.025

    @patch("requests.get")
    def test_get_history_online(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "count": 1,
            "predictions": [
                {
                    "prediction_id": "p1",
                    "device_id": "VENT-001",
                    "fault_probability": 0.01,
                    "predicted_label": "Normal",
                }
            ],
        }
        mock_get.return_value = mock_resp

        client = VentilatorDashboardClient(base_url="http://test-server:8000")
        history = client.get_history(device_id="VENT-001", limit=10)

        assert history["source"] == "[LIVE API]"
        assert history["count"] == 1
        assert len(history["predictions"]) == 1


# ---------------------------------------------------------------------------
# 2. Dashboard Client Tests - Offline & Resilient Demo Fallback
# ---------------------------------------------------------------------------
class TestDashboardClientOfflineFallback:
    @patch("requests.get", side_effect=requests.ConnectionError("Connection refused"))
    def test_check_health_connection_error_fallback(self, mock_get):
        client = VentilatorDashboardClient(base_url="http://offline-server:8000")
        health = client.check_health()

        # Must not raise an exception; must return demo fallback
        assert health["is_online"] is False
        assert health["source"] == "[DEMO DATA]"
        assert health["status"] == "offline"
        assert health["window_size"] == 20

    @patch("requests.post", side_effect=requests.Timeout("Read timed out"))
    def test_predict_timeout_fallback(self, mock_post):
        client = VentilatorDashboardClient(base_url="http://offline-server:8000")
        readings = [
            {
                "internal_temperature_C": 75.0,  # Overheating pattern
                "vibration_mm_s": 25.0,
                "airway_pressure_cmH2O": 50.0,
                "blower_speed_rpm": 12000.0,
            }
        ] * 20
        res = client.predict(device_id="DEV-FALLBACK", readings=readings)

        assert res["source"] == "[DEMO DATA]"
        assert res["predicted_label"] == "Fault"
        assert res["fault_probability"] >= 0.5
        assert 0.0 <= res["confidence"] <= 1.0

    @patch("requests.post")
    def test_predict_server_500_fallback(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        client = VentilatorDashboardClient(base_url="http://broken-server:8000")
        normal_readings = [{"internal_temperature_C": 35.0, "vibration_mm_s": 1.0}] * 20
        res = client.predict(device_id="DEV-500", readings=normal_readings)

        assert res["source"] == "[DEMO DATA]"
        assert res["predicted_label"] == "Normal"
        assert res["fault_probability"] < 0.5

    @patch("requests.get", side_effect=requests.ConnectionError("Offline"))
    def test_get_history_offline_fallback(self, mock_get):
        client = VentilatorDashboardClient(base_url="http://offline-server:8000")
        history = client.get_history(device_id="VENT-001", limit=5)

        assert history["source"] == "[DEMO DATA]"
        assert history["count"] > 0
        assert "predictions" in history
        assert len(history["predictions"]) <= 5


# ---------------------------------------------------------------------------
# 3. Dataset & Metrics Retrieval Tests
# ---------------------------------------------------------------------------
class TestDatasetAndMetrics:
    def test_load_dataset(self):
        client = VentilatorDashboardClient()
        df = client.load_dataset()
        assert not df.empty
        assert "device_id" in df.columns
        assert "inlet_pressure_kPa" in df.columns
        assert "airway_pressure_cmH2O" in df.columns

    def test_get_sample_episode_window_length(self):
        client = VentilatorDashboardClient()
        normal_win = client.get_sample_episode_window(fault_type="normal", window_size=20)
        fault_win = client.get_sample_episode_window(fault_type="fault", window_size=20)

        assert len(normal_win) == 20
        assert len(fault_win) == 20
        assert isinstance(normal_win[0], dict)
        assert "device_model" in normal_win[0]

    def test_get_evaluation_metrics(self):
        client = VentilatorDashboardClient()
        metrics = client.get_evaluation_metrics()

        required_keys = [
            "accuracy", "precision", "recall", "f1_score",
            "roc_auc", "pr_auc", "cohen_kappa", "confusion_matrix"
        ]
        for k in required_keys:
            assert k in metrics, f"Missing metric key: {k}"

        assert metrics["accuracy"] >= 0.99
        assert metrics["pr_auc"] >= 0.99


# ---------------------------------------------------------------------------
# 4. Streamlit AppTest Smoke Test
# ---------------------------------------------------------------------------
class TestStreamlitAppRender:
    def test_app_renders_without_exceptions(self):
        """Verify that app.py compiles, initializes, and renders without unhandled errors."""
        from pathlib import Path
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()

        # Ensure no unhandled exceptions in the Streamlit runtime
        assert not at.exception, f"Streamlit raised an exception: {at.exception}"
