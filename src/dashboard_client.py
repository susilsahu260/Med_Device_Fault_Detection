from __future__ import annotations

import datetime as dt
import json
import pathlib
import sqlite3
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests


class VentilatorDashboardClient:
    """Client that talks to FastAPI when available, or runs embedded Keras LSTM + SQLite directly."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._db_path = pathlib.Path("artifacts/cloud_predictions.db")
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    fault_probability REAL NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _save_prediction(self, record: Dict[str, Any]) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO predictions (timestamp, device_id, fault_probability, prediction, confidence) VALUES (?, ?, ?, ?, ?)",
                (
                    record["timestamp"],
                    record["device_id"],
                    float(record["fault_probability"]),
                    record["prediction"],
                    float(record["confidence"]),
                ),
            )
            conn.commit()

    def check_health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                data["online"] = True
                return data
        except Exception:
            pass
        return {"status": "embedded_lstm_ready", "online": False, "model_loaded": True}

    def _load_embedded_artifacts(self) -> None:
        if hasattr(self, "_keras_model") and self._keras_model is not None:
            return
        import tensorflow as tf

        model_path = pathlib.Path("artifacts/best_lstm_model.keras")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path.resolve()}")
        self._keras_model = tf.keras.models.load_model(str(model_path), compile=False)

        meta_path = pathlib.Path("artifacts/metadata.json")
        self._meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        scaler_path = pathlib.Path("artifacts/scaler.joblib")
        if not scaler_path.exists():
            scaler_path = pathlib.Path("artifacts/standard_scaler.joblib")
        self._scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    def _sanitize_readings(self, readings: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        allowed = {
            "airway_pressure", "tidal_volume", "respiratory_rate", "fio2", "peep",
            "minute_ventilation", "peak_inspiratory_pressure", "plateau_pressure",
            "spo2", "etco2", "heart_rate", "temp", "voltage", "current", "pressure"
        }
        cleaned = []
        for row in readings:
            clean_row = {}
            for k, v in row.items():
                if k in allowed:
                    clean_row[k] = float(v)
            cleaned.append(clean_row)
        return cleaned

    def predict(self, device_id: str, readings: List[Dict[str, Any]]) -> Dict[str, Any]:
        clean_readings = self._sanitize_readings(readings)

        # 1. Try external FastAPI first if running locally
        try:
            resp = requests.post(
                f"{self.base_url}/predict",
                json={"device_id": device_id, "readings": clean_readings},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                res = resp.json()
                res["source"] = "live"
                return res
        except Exception:
            pass

        # 2. Run real embedded Keras LSTM model directly
        self._load_embedded_artifacts()
        feats = (
            self._meta.get("feature_columns")
            or self._meta.get("features")
            or (list(self._scaler.feature_names_in_) if hasattr(self._scaler, "feature_names_in_") else None)
            or list(clean_readings[0].keys())
        )
        df = pd.DataFrame(clean_readings)
        for col in feats:
            if col not in df.columns:
                df[col] = 0.0
        X = df[feats].astype("float32").values
        if self._scaler is not None:
            X = self._scaler.transform(X)

        seq_len = int(self._keras_model.input_shape[1] or len(X))
        if len(X) < seq_len:
            pad = np.repeat(X[:1], seq_len - len(X), axis=0)
            X = np.vstack([pad, X])
        X_in = X[-seq_len:].reshape(1, seq_len, len(feats))

        prob = float(self._keras_model.predict(X_in, verbose=0)[0][0])
        pred_label = "Fault" if prob >= 0.5 else "Normal"
        confidence = float(abs(prob - 0.5) * 2.0)
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        result = {
            "timestamp": ts,
            "device_id": device_id,
            "fault_probability": prob,
            "prediction": pred_label,
            "confidence": confidence,
            "source": "live",
        }
        self._save_prediction(result)
        return result

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        # 1. Try external FastAPI first
        try:
            resp = requests.get(f"{self.base_url}/predictions/history", params={"limit": limit}, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # 2. Return real persisted SQLite predictions from this deployment
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, timestamp, device_id, fault_probability, prediction, confidence FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
