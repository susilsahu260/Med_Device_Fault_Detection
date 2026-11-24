"""FastAPI backend for the ventilator fault-detection model.

Endpoints
---------
GET  /health                  → Service liveness + model readiness
POST /predict                 → Run inference on a validated sensor window
GET  /predictions/history     → Query persisted prediction records

Architecture
------------
* Lifespan context manager loads the Keras model, fitted scaler/encoder, and
  metadata.json exactly once at startup into ``app.state``.
* SQLite (stdlib ``sqlite3``, synchronous, tiny connection pool) stores every
  prediction so the history endpoint can answer without reloading files.
* All heavy validation happens in ``src.schemas`` via Pydantic v2 — the route
  handlers receive fully validated objects.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Ensure project root on sys.path when run as a script
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # headless — must come before any pyplot import

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionHistoryResponse,
    PredictionRecord,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — allow override via environment variable for testing
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "best_lstm_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
ENCODER_PATH = ARTIFACTS_DIR / "encoder.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DB_PATH = Path(os.environ.get("PREDICTIONS_DB", str(ARTIFACTS_DIR / "predictions.db")))

FAULT_THRESHOLD = float(os.environ.get("FAULT_THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_db_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create the predictions table if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id      TEXT PRIMARY KEY,
                device_id          TEXT NOT NULL,
                fault_probability  REAL NOT NULL,
                predicted_label    TEXT NOT NULL,
                threshold          REAL NOT NULL,
                window_size_received INTEGER NOT NULL,
                timestamp          TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _insert_prediction(record: PredictResponse) -> None:
    """Persist a prediction response to SQLite."""
    with _get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (prediction_id, device_id, fault_probability, predicted_label,
                 threshold, window_size_received, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.prediction_id,
                record.device_id,
                record.fault_probability,
                record.predicted_label,
                record.threshold,
                record.window_size_received,
                record.timestamp.isoformat(),
            ),
        )
        conn.commit()


def _query_predictions(
    device_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Return prediction rows from SQLite, newest first."""
    with _get_db_connection() as conn:
        if device_id:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE device_id = ? ORDER BY timestamp DESC LIMIT ?",
                (device_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Lifespan — load artifacts once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load the ML model and preprocessing artifacts at startup."""
    logger.info("=== Ventilator API: startup ===")

    # Initialise SQLite schema
    _init_db()
    logger.info("SQLite predictions store ready: %s", DB_PATH)

    # Load metadata
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as fh:
            metadata: dict = json.load(fh)
        app.state.metadata = metadata
        app.state.window_size = int(metadata["window_size"])
        app.state.num_features = int(metadata["num_total_features"])
        app.state.feature_names = list(metadata["encoded_feature_names"])
        logger.info(
            "Metadata loaded — window_size=%d  num_features=%d",
            app.state.window_size,
            app.state.num_features,
        )
    else:
        logger.warning("metadata.json not found at %s", METADATA_PATH)
        app.state.metadata = {}
        app.state.window_size = 20
        app.state.num_features = 16
        app.state.feature_names = []

    # Load fitted scaler
    if SCALER_PATH.exists():
        app.state.scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded: %s", SCALER_PATH)
    else:
        app.state.scaler = None
        logger.warning("scaler.joblib not found — inference will fail")

    # Load fitted encoder
    if ENCODER_PATH.exists():
        app.state.encoder = joblib.load(ENCODER_PATH)
        logger.info("Encoder loaded: %s", ENCODER_PATH)
    else:
        app.state.encoder = None
        logger.warning("encoder.joblib not found — inference will fail")

    # Load Keras model (import TF lazily to avoid import-time overhead)
    if MODEL_PATH.exists():
        try:
            import tensorflow as tf  # noqa: PLC0415
            app.state.model = tf.keras.models.load_model(str(MODEL_PATH))
            app.state.model_loaded = True
            app.state.model_name = MODEL_PATH.stem
            logger.info("Model loaded: %s  (%d parameters)", MODEL_PATH, app.state.model.count_params())
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            app.state.model = None
            app.state.model_loaded = False
            app.state.model_name = "unavailable"
    else:
        logger.warning("Model file not found: %s", MODEL_PATH)
        app.state.model = None
        app.state.model_loaded = False
        app.state.model_name = "unavailable"

    yield  # ← application runs

    # Shutdown — nothing heavy to release for SQLite (stdlib) or joblib objects
    logger.info("=== Ventilator API: shutdown ===")
    if app.state.model is not None:
        del app.state.model


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ventilator Fault Detection API",
    description=(
        "REST API for real-time ventilator equipment fault detection "
        "powered by a trained LSTM model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "operating_hours",
    "inlet_pressure_kPa",
    "airway_pressure_cmH2O",
    "airflow_L_min",
    "oxygen_concentration_pct",
    "internal_temperature_C",
    "supply_voltage_V",
    "motor_current_A",
    "blower_speed_rpm",
    "battery_voltage_V",
    "vibration_mm_s",
    "filter_pressure_drop_kPa",
    "ambient_humidity_pct",
]
CATEGORICAL_COL = "device_model"


def _preprocess_window(request: PredictRequest, app_state) -> np.ndarray:
    """Convert validated readings into a scaled/encoded (1, window_size, features) tensor.

    Returns
    -------
    np.ndarray
        Shape ``(1, window_size, num_features)`` ready for model.predict().

    Raises
    ------
    RuntimeError
        If the scaler or encoder is not loaded.
    """
    if app_state.scaler is None or app_state.encoder is None:
        raise RuntimeError("Preprocessing artifacts (scaler/encoder) are not loaded.")

    # Build a DataFrame from the validated readings
    rows = [r.model_dump() for r in request.readings]
    df = pd.DataFrame(rows)

    # --- Numeric scaling ---
    numeric_data = df[NUMERIC_COLS].values.astype(float)
    numeric_scaled = app_state.scaler.transform(numeric_data)

    # --- Categorical encoding ---
    cat_data = df[[CATEGORICAL_COL]]
    try:
        cat_encoded = app_state.encoder.transform(cat_data)
    except Exception:
        # Unseen category — encoder may raise; return zeros for OHE columns
        n_cat_cols = len(app_state.feature_names) - len(NUMERIC_COLS)
        cat_encoded = np.zeros((len(df), n_cat_cols), dtype=float)

    # Concatenate → (window_size, num_features)
    window = np.concatenate([numeric_scaled, cat_encoded], axis=1).astype(np.float32)

    # Expand batch dim → (1, window_size, num_features)
    return window[np.newaxis, :, :]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """Return service liveness and model readiness status."""
    return HealthResponse(
        status="ok" if app.state.model_loaded else "degraded",
        model_loaded=app.state.model_loaded,
        model_name=app.state.model_name,
        window_size=app.state.window_size,
        num_features=app.state.num_features,
        feature_names=app.state.feature_names,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Run fault-detection inference on a validated sensor window.

    * Validates that the number of readings equals ``window_size``.
    * Scales/encodes the readings using the fitted preprocessing artifacts.
    * Runs the LSTM model and returns the fault probability + predicted label.
    * Persists the prediction record to SQLite.
    """
    # Window-size check (runtime — depends on loaded metadata)
    window_size: int = app.state.window_size
    if len(request.readings) != window_size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected exactly {window_size} readings (window_size), "
                f"got {len(request.readings)}."
            ),
        )

    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is degraded.",
        )

    # Preprocess
    try:
        x = _preprocess_window(request, app.state)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Inference
    prob: float = float(app.state.model.predict(x, verbose=0).ravel()[0])
    prob = float(np.clip(prob, 0.0, 1.0))

    label = "Fault" if prob >= FAULT_THRESHOLD else "Normal"
    confidence = float(1.0 - abs(prob - FAULT_THRESHOLD) * 2)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    response = PredictResponse(
        prediction_id=str(uuid.uuid4()),
        device_id=request.device_id,
        fault_probability=round(prob, 6),
        predicted_label=label,
        threshold=FAULT_THRESHOLD,
        confidence=round(confidence, 6),
        window_size_received=len(request.readings),
        timestamp=datetime.now(timezone.utc),
    )

    # Persist
    _insert_prediction(response)
    logger.info(
        "Prediction — device=%s  prob=%.4f  label=%s",
        request.device_id, prob, label,
    )
    return response


@app.get("/predictions/history", response_model=PredictionHistoryResponse, tags=["History"])
async def predictions_history(
    device_id: Optional[str] = Query(default=None, description="Filter by device ID"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum records to return"),
) -> PredictionHistoryResponse:
    """Return past prediction records from the SQLite persistence store."""
    rows = _query_predictions(device_id=device_id, limit=limit)
    records = [PredictionRecord(**row) for row in rows]
    return PredictionHistoryResponse(count=len(records), predictions=records)


# ---------------------------------------------------------------------------
# Custom 422 handler — ensure consistent JSON structure
# ---------------------------------------------------------------------------

from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse as _JSONResponse  # noqa: E402


def _serialise_errors(errors: list) -> list:
    """Convert Pydantic v2 error dicts to plain JSON-safe dicts.

    Pydantic v2 error dicts may contain Python objects (e.g. ``ValueError``
    instances in the ``ctx`` sub-dict) that are not JSON-serialisable.
    This helper converts every value to its ``str()`` representation.
    """
    safe: list = []
    for err in errors:
        safe_err: dict = {}
        for k, v in err.items():
            if k == "ctx" and isinstance(v, dict):
                safe_err[k] = {ck: str(cv) for ck, cv in v.items()}
            elif isinstance(v, (list, tuple)):
                safe_err[k] = [str(i) for i in v]
            else:
                try:
                    import json as _json
                    _json.dumps(v)  # test serialisability
                    safe_err[k] = v
                except (TypeError, ValueError):
                    safe_err[k] = str(v)
        safe.append(safe_err)
    return safe


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):  # type: ignore[override]
    return _JSONResponse(
        status_code=422,
        content={
            "detail": _serialise_errors(exc.errors()),
            "body": None,
        },
    )
