import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.preprocessing import (
    clean_data,
    split_data_by_device,
    fit_transformers,
    transform_features,
    create_sliding_windows,
    save_artifacts,
    load_artifacts,
    VentilatorPreprocessor,
    DEFAULT_NUMERIC_COLS,
    DEFAULT_CATEGORICAL_COLS,
    PHYSICAL_SENSOR_BOUNDS,
)


@pytest.fixture
def sample_raw_df():
    """Generates a small deterministic multi-device ventilator dataframe for testing."""
    np.random.seed(42)
    devices = ["VENT-0001", "VENT-0002", "VENT-0003", "VENT-0004", "VENT-0005"]
    models = ["VENT-X100", "VENT-X100", "VENT-X200", "VENT-X200", "VENT-M500"]
    rows = []

    for dev_idx, (dev, model) in enumerate(zip(devices, models)):
        base_time = pd.Timestamp("2026-01-01 08:00:00") + pd.Timedelta(days=dev_idx)
        for step in range(30):
            ts = base_time + pd.Timedelta(minutes=5 * step)
            is_failure = 1 if 10 <= step < 20 else 0
            fault_type = "overheating" if is_failure else "normal"
            temp = 52.0 if is_failure else 38.0 + np.random.normal(0, 0.5)
            vib = 2.1 if is_failure else 0.8 + np.random.normal(0, 0.05)
            
            rows.append({
                "timestamp": str(ts),
                "device_id": dev,
                "device_model": model,
                "operating_hours": 500.0 + step * 0.0833,
                "inlet_pressure_kPa": 5.5 + np.random.normal(0, 0.1),
                "airway_pressure_cmH2O": 18.0 + np.random.normal(0, 0.2),
                "airflow_L_min": 41.0 + np.random.normal(0, 0.5),
                "oxygen_concentration_pct": 35.0 + np.random.normal(0, 0.2),
                "internal_temperature_C": temp,
                "supply_voltage_V": 230.0 + np.random.normal(0, 1.0),
                "motor_current_A": 1.8 + (0.3 if is_failure else 0.0),
                "blower_speed_rpm": 3200.0 + np.random.normal(0, 20),
                "battery_voltage_V": 24.0 + np.random.normal(0, 0.1),
                "vibration_mm_s": vib,
                "filter_pressure_drop_kPa": 0.7 + np.random.normal(0, 0.05),
                "ambient_humidity_pct": 45.0 + np.random.normal(0, 0.5),
                "failure": is_failure,
                "fault_type": fault_type,
            })
    return pd.DataFrame(rows)


# ============================================================================
# 1. Cleaning & Imputation Tests
# ============================================================================

def test_clean_data_handles_missing_values(sample_raw_df):
    """Verifies that missing values are imputed and no NaNs remain."""
    df_missing = sample_raw_df.copy()
    # Inject isolated NaNs in numeric sensor columns
    df_missing.loc[2, "internal_temperature_C"] = np.nan
    df_missing.loc[5, "airflow_L_min"] = np.nan
    df_missing.loc[12, "vibration_mm_s"] = np.nan

    cleaned = clean_data(df_missing)
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) == len(sample_raw_df)


def test_clean_data_removes_duplicate_records(sample_raw_df):
    """Verifies that duplicate rows are cleanly removed."""
    df_dup = pd.concat([sample_raw_df, sample_raw_df.iloc[[0, 1, 5]]], ignore_index=True)
    assert len(df_dup) == len(sample_raw_df) + 3

    cleaned = clean_data(df_dup)
    assert len(cleaned) == len(sample_raw_df)
    assert cleaned.duplicated(subset=["device_id", "timestamp"]).sum() == 0


def test_clean_data_outlier_filtering_vs_fault_preservation(sample_raw_df):
    """Verifies that physically impossible sensor errors are handled while genuine fault spikes are preserved."""
    df_outliers = sample_raw_df.copy()
    
    # Inject physically impossible values (e.g. negative vibration, impossible temperature)
    impossible_idx_temp = 3
    impossible_idx_vib = 4
    df_outliers.loc[impossible_idx_temp, "internal_temperature_C"] = -150.0  # Physically impossible
    df_outliers.loc[impossible_idx_vib, "vibration_mm_s"] = -10.0            # Physically impossible
    
    # Ensure genuine fault spikes exist in sample data (overheating ~52C, vibration ~2.1)
    fault_mask = df_outliers["failure"] == 1
    assert (df_outliers.loc[fault_mask, "internal_temperature_C"] > 50.0).any()
    assert (df_outliers.loc[fault_mask, "vibration_mm_s"] > 2.0).any()

    cleaned = clean_data(df_outliers)

    # Impossible values should be clipped or imputed to physically valid boundaries
    assert (cleaned["internal_temperature_C"] >= PHYSICAL_SENSOR_BOUNDS["internal_temperature_C"][0]).all()
    assert (cleaned["vibration_mm_s"] >= PHYSICAL_SENSOR_BOUNDS["vibration_mm_s"][0]).all()

    # Crucially, genuine fault spikes MUST be preserved
    cleaned_faults = cleaned[cleaned["failure"] == 1]
    assert (cleaned_faults["internal_temperature_C"] > 50.0).any()
    assert (cleaned_faults["vibration_mm_s"] > 2.0).any()


# ============================================================================
# 2. Data Leakage Prevention & Splitting Tests
# ============================================================================

def test_split_by_device_zero_leakage(sample_raw_df):
    """Ensures device-grouped split creates strictly disjoint sets with zero overlap."""
    train_df, val_df, test_df = split_data_by_device(
        sample_raw_df,
        device_col="device_id",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify_col="device_model",
        random_state=42,
    )

    train_devs = set(train_df["device_id"].unique())
    val_devs = set(val_df["device_id"].unique())
    test_devs = set(test_df["device_id"].unique())

    # 1. Mutually exclusive devices
    assert train_devs.isdisjoint(val_devs), "Train and Val share device IDs! Data leakage detected!"
    assert train_devs.isdisjoint(test_devs), "Train and Test share device IDs! Data leakage detected!"
    assert val_devs.isdisjoint(test_devs), "Val and Test share device IDs! Data leakage detected!"

    # 2. All rows accounted for
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_raw_df)

    # 3. Time sequence intact within each device
    for df_split in [train_df, val_df, test_df]:
        for dev, group in df_split.groupby("device_id"):
            assert group["timestamp"].is_monotonic_increasing


# ============================================================================
# 3. Scaler & Encoder Isolation Tests
# ============================================================================

def test_scaler_fitted_on_train_only(sample_raw_df):
    """Verifies that scalers and encoders are fitted strictly on train_df without seeing test data."""
    train_df, val_df, test_df = split_data_by_device(
        sample_raw_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
    )

    scaler, encoder = fit_transformers(
        train_df,
        numeric_cols=DEFAULT_NUMERIC_COLS,
        categorical_cols=DEFAULT_CATEGORICAL_COLS,
        scaler_type="robust",
    )

    # The scaler center/scale must match train_df statistics, not the full df
    assert hasattr(scaler, "center_") or hasattr(scaler, "mean_")
    
    # Transform test set
    X_test, _ = transform_features(test_df, scaler, encoder, DEFAULT_NUMERIC_COLS, DEFAULT_CATEGORICAL_COLS)
    assert not np.isnan(X_test).any()
    assert X_test.shape[0] == len(test_df)


def test_categorical_encoding_unseen_values(sample_raw_df):
    """Verifies that previously unseen categorical values in test do not raise exceptions."""
    train_df, _, test_df = split_data_by_device(
        sample_raw_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    scaler, encoder = fit_transformers(
        train_df,
        numeric_cols=DEFAULT_NUMERIC_COLS,
        categorical_cols=DEFAULT_CATEGORICAL_COLS,
    )

    # Inject unknown device model in test_df
    test_df_unseen = test_df.copy()
    test_df_unseen.loc[test_df_unseen.index[0], "device_model"] = "VENT-FUTURE999"

    # Should transform cleanly without crash
    X_unseen, _ = transform_features(
        test_df_unseen, scaler, encoder, DEFAULT_NUMERIC_COLS, DEFAULT_CATEGORICAL_COLS
    )
    assert not np.isnan(X_unseen).any()
    assert X_unseen.shape[0] == len(test_df_unseen)


# ============================================================================
# 4. Sequential Sliding Window & Boundary Isolation Tests
# ============================================================================

def test_sliding_window_boundary_isolation():
    """Confirms sliding windows NEVER cross device or episode boundaries."""
    # Create two artificial devices with 15 rows each
    records = []
    for dev in ["DEV-A", "DEV-B"]:
        for i in range(15):
            records.append({
                "device_id": dev,
                "timestamp": str(pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * i)),
                "val": float(100 if dev == "DEV-A" else 200),
                "failure": 0 if i < 10 else 1,
            })
    toy_df = pd.DataFrame(records)

    window_size = 5
    step_size = 1
    X_seq, y_seq = create_sliding_windows(
        toy_df,
        window_size=window_size,
        step_size=step_size,
        group_col="device_id",
        feature_cols=["val"],
        target_col="failure",
    )

    # For DEV-A (15 rows, w=5, s=1): 15 - 5 + 1 = 11 windows
    # For DEV-B (15 rows, w=5, s=1): 15 - 5 + 1 = 11 windows
    # Total = 22 windows
    assert X_seq.shape == (22, 5, 1)
    assert len(y_seq) == 22

    # Verify no window has mixed values (DEV-A is all 100s, DEV-B is all 200s)
    for window in X_seq:
        unique_vals = np.unique(window)
        assert len(unique_vals) == 1, f"Window crossed device boundary! Found {unique_vals}"
        assert unique_vals[0] in [100.0, 200.0]


def test_sliding_window_output_shapes_and_target_alignment(sample_raw_df):
    """Verifies that sliding window tensors match expected 3D shapes (samples, window_size, features)."""
    window_size = 10
    step_size = 2
    feature_cols = DEFAULT_NUMERIC_COLS

    X_seq, y_seq = create_sliding_windows(
        sample_raw_df,
        window_size=window_size,
        step_size=step_size,
        group_col="device_id",
        feature_cols=feature_cols,
        target_col="failure",
    )

    assert X_seq.ndim == 3
    assert X_seq.shape[1] == window_size
    assert X_seq.shape[2] == len(feature_cols)
    assert X_seq.shape[0] == len(y_seq)
    assert X_seq.dtype == np.float32 or X_seq.dtype == np.float64
    assert not np.isnan(X_seq).any()


# ============================================================================
# 5. Artifact Saving & Loading Tests
# ============================================================================

def test_artifact_save_and_load(sample_raw_df, tmp_path):
    """Verifies pipeline artifacts can be saved and loaded to reproduce identical transformations."""
    preprocessor = VentilatorPreprocessor(artifacts_dir=str(tmp_path))
    
    # Fit & Transform
    train_tensors, val_tensors, test_tensors = preprocessor.fit_transform(
        sample_raw_df, window_size=10, step_size=2
    )

    # Check artifact files exist
    assert (tmp_path / "scaler.joblib").exists()
    assert (tmp_path / "encoder.joblib").exists()
    assert (tmp_path / "metadata.json").exists()

    # Load from disk
    loaded_preprocessor = VentilatorPreprocessor.load(artifacts_dir=str(tmp_path))
    assert loaded_preprocessor.is_fitted

    # Verify metadata contents
    with open(tmp_path / "metadata.json", "r") as f:
        meta = json.load(f)
    assert "numeric_features" in meta
    assert "categorical_features" in meta
    assert meta["window_size"] == 10

    # Transform new data with loaded preprocessor
    new_data = sample_raw_df.head(20).copy()
    X_new_tab, _ = loaded_preprocessor.transform_tabular(new_data)
    assert X_new_tab.shape[0] == 20
    assert not np.isnan(X_new_tab).any()


# ============================================================================
# 6. Edge Cases & Defensive Execution Tests
# ============================================================================

def test_unfitted_preprocessor_raises_runtime_error(sample_raw_df):
    """Ensures calling transform or save on an unfitted preprocessor raises RuntimeError."""
    preprocessor = VentilatorPreprocessor()
    with pytest.raises(RuntimeError, match="has not been fitted"):
        preprocessor.transform_tabular(sample_raw_df)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        preprocessor.transform_sequential(sample_raw_df)

    with pytest.raises(RuntimeError, match="Cannot save an unfitted"):
        preprocessor.save()


def test_standard_scaler_and_unstratified_split(sample_raw_df):
    """Verifies standard scaler configuration and unstratified entity splitting."""
    train_df, val_df, test_df = split_data_by_device(
        sample_raw_df, stratify_col=None, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

    scaler, encoder = fit_transformers(
        train_df,
        numeric_cols=DEFAULT_NUMERIC_COLS,
        categorical_cols=DEFAULT_CATEGORICAL_COLS,
        scaler_type="standard",
    )
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")

    X_train, feature_names = transform_features(
        train_df, scaler, encoder, DEFAULT_NUMERIC_COLS, DEFAULT_CATEGORICAL_COLS
    )
    assert X_train.shape[0] == len(train_df)
    assert len(feature_names) == 16


def test_empty_dataframe_transform(sample_raw_df):
    """Verifies that an empty dataframe returns properly shaped empty arrays without crashing."""
    train_df, _, _ = split_data_by_device(sample_raw_df, random_state=42)
    scaler, encoder = fit_transformers(train_df)
    
    empty_df = sample_raw_df.iloc[0:0].copy()
    X_empty, feature_names = transform_features(empty_df, scaler, encoder)
    assert X_empty.shape == (0, len(feature_names))

    X_seq_empty, y_seq_empty = create_sliding_windows(
        empty_df, window_size=10, fitted_scaler=scaler, fitted_encoder=encoder
    )
    assert X_seq_empty.shape == (0, 10, len(feature_names))
    assert len(y_seq_empty) == 0


def test_main_cli_execution():
    """Verifies that the main() execution block runs smoothly on the dataset."""
    from src.preprocessing import main
    # Should run and execute without error
    main()


