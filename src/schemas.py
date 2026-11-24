"""Pydantic v2 request/response schemas for the ventilator fault-detection API.

Physical sensor bounds are sourced directly from artifacts/metadata.json and
enforced at the field-validator level, so invalid readings produce HTTP 422
before any ML inference is attempted.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Physical sensor bounds (must stay in sync with metadata.json / preprocessing.py)
# ---------------------------------------------------------------------------
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "operating_hours": (0.0, 100_000.0),
    "inlet_pressure_kPa": (0.0, 50.0),
    "airway_pressure_cmH2O": (0.0, 120.0),
    "airflow_L_min": (0.0, 200.0),
    "oxygen_concentration_pct": (15.0, 100.0),
    "internal_temperature_C": (-20.0, 120.0),
    "supply_voltage_V": (0.0, 500.0),
    "motor_current_A": (0.0, 20.0),
    "blower_speed_rpm": (0.0, 15_000.0),
    "battery_voltage_V": (0.0, 50.0),
    "vibration_mm_s": (0.0, 50.0),
    "filter_pressure_drop_kPa": (0.0, 20.0),
    "ambient_humidity_pct": (0.0, 100.0),
}

VALID_DEVICE_MODELS: frozenset[str] = frozenset(
    {"VENT-M500", "VENT-X100", "VENT-X200"}
)


# ---------------------------------------------------------------------------
# Single time-step sensor reading
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    """One 5-minute sensor snapshot from a ventilator device.

    Every numeric field is validated against the physical sensor bounds
    documented in ``artifacts/metadata.json``.  NaN and ±Inf are rejected
    for every numeric field.
    """

    model_config = {"extra": "forbid"}  # reject unknown fields

    # Numeric sensors
    operating_hours: float = Field(..., description="Cumulative operating hours [0, 100 000]")
    inlet_pressure_kPa: float = Field(..., description="Inlet pressure in kPa [0, 50]")
    airway_pressure_cmH2O: float = Field(..., description="Airway pressure in cmH₂O [0, 120]")
    airflow_L_min: float = Field(..., description="Airflow in L/min [0, 200]")
    oxygen_concentration_pct: float = Field(..., description="O₂ concentration % [15, 100]")
    internal_temperature_C: float = Field(..., description="Internal temperature °C [-20, 120]")
    supply_voltage_V: float = Field(..., description="Supply voltage V [0, 500]")
    motor_current_A: float = Field(..., description="Motor current A [0, 20]")
    blower_speed_rpm: float = Field(..., description="Blower speed RPM [0, 15 000]")
    battery_voltage_V: float = Field(..., description="Battery voltage V [0, 50]")
    vibration_mm_s: float = Field(..., description="Vibration mm/s [0, 50]")
    filter_pressure_drop_kPa: float = Field(..., description="Filter pressure drop kPa [0, 20]")
    ambient_humidity_pct: float = Field(..., description="Ambient humidity % [0, 100]")

    # Categorical sensor
    device_model: str = Field(..., description="Device model identifier")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator(
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
        mode="before",
    )
    @classmethod
    def reject_nan_inf(cls, v: object) -> object:
        """Reject NaN and Inf before any other validation."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("sensor value must not be NaN or Inf")
        return v

    @model_validator(mode="after")
    def check_physical_bounds(self) -> "SensorReading":
        """Validate every numeric sensor value against its physical operating range."""
        for field_name, (lo, hi) in PHYSICAL_BOUNDS.items():
            value: float = getattr(self, field_name)
            if not (lo <= value <= hi):
                raise ValueError(
                    f"'{field_name}' value {value} is outside physical bounds [{lo}, {hi}]"
                )
        return self

    @field_validator("device_model", mode="after")
    @classmethod
    def validate_device_model(cls, v: str) -> str:
        if v not in VALID_DEVICE_MODELS:
            raise ValueError(
                f"Unknown device_model '{v}'. Valid values: {sorted(VALID_DEVICE_MODELS)}"
            )
        return v


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Predict endpoint request: exactly ``window_size`` consecutive sensor readings."""

    model_config = {"extra": "forbid"}

    device_id: str = Field(..., min_length=1, description="Unique device identifier")
    readings: List[SensorReading] = Field(
        ...,
        min_length=1,
        description="Ordered sensor readings (must equal the model's window_size)",
    )

    @model_validator(mode="after")
    def readings_must_not_be_empty(self) -> "PredictRequest":
        # Min-length=1 handles the empty case; window_size check is done in
        # the route handler after loading metadata (window_size is runtime data).
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Prediction result returned by POST /predict."""

    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    fault_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_label: Literal["Normal", "Fault"]
    threshold: float = Field(default=0.5)
    confidence: float = Field(..., ge=0.0, le=1.0, description="1 − |p − threshold| × 2 when binary")
    window_size_received: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Health check response from GET /health."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    window_size: int
    num_features: int
    feature_names: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PredictionRecord(BaseModel):
    """One row in the predictions history store."""

    prediction_id: str
    device_id: str
    fault_probability: float
    predicted_label: str
    threshold: float
    window_size_received: int
    timestamp: str  # ISO-8601 string as stored in SQLite


class PredictionHistoryResponse(BaseModel):
    """Response from GET /predictions/history."""

    count: int
    predictions: List[PredictionRecord]
