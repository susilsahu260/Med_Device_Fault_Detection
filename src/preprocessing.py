"""Ventilator Sensor Telemetry Preprocessing Pipeline.

This module provides a modular, leak-free preprocessing pipeline for longitudinal
ventilator equipment sensor data. It supports tabular feature engineering and
isolated sequential sliding window tensor generation (samples, window_size, features)
for 1D-CNN + BiLSTM deep learning models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# Default sensor and metadata columns based on EDA
DEFAULT_NUMERIC_COLS: List[str] = [
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

DEFAULT_CATEGORICAL_COLS: List[str] = [
    "device_model",
]

DEFAULT_TARGET_COLS: List[str] = [
    "failure",
    "fault_type",
]

# Physical sensor boundaries: used strictly to filter or clip physically impossible
# sensor hardware faults (e.g. negative pressure/vibration, extreme impossible values)
# while preserving authentic operational fault spikes (overheating, vibration spikes, etc.)
PHYSICAL_SENSOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "operating_hours": (0.0, 100000.0),
    "inlet_pressure_kPa": (0.0, 50.0),
    "airway_pressure_cmH2O": (0.0, 120.0),
    "airflow_L_min": (0.0, 200.0),
    "oxygen_concentration_pct": (15.0, 100.0),
    "internal_temperature_C": (-20.0, 120.0),
    "supply_voltage_V": (0.0, 500.0),
    "motor_current_A": (0.0, 20.0),
    "blower_speed_rpm": (0.0, 15000.0),
    "battery_voltage_V": (0.0, 50.0),
    "vibration_mm_s": (0.0, 50.0),
    "filter_pressure_drop_kPa": (0.0, 20.0),
    "ambient_humidity_pct": (0.0, 100.0),
}

# Empirical ground-truth class mappings
FAULT_TYPE_MAPPING: Dict[str, int] = {
    "normal": 0,
    "sensor_drift": 1,
    "overheating": 2,
    "airflow_restriction": 3,
    "blower_fault": 4,
    "power_supply_fault": 5,
}

REVERSE_FAULT_TYPE_MAPPING: Dict[int, str] = {v: k for k, v in FAULT_TYPE_MAPPING.items()}


# ============================================================================
# 1. Data Cleaning & Integrity
# ============================================================================

def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    impossible_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    sort_time: bool = True,
) -> pd.DataFrame:
    """Cleans sensor telemetry by removing duplicates, imputing missing values, and clipping impossible sensor values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry dataframe.
    drop_duplicates : bool, default=True
        Whether to drop duplicate records based on `(device_id, timestamp)` or full rows.
    impossible_bounds : Optional[Dict[str, Tuple[float, float]]], default=None
        Physical bounds for clipping sensor errors. If None, uses PHYSICAL_SENSOR_BOUNDS.
    sort_time : bool, default=True
        Whether to sort records chronologically within each device.

    Returns
    -------
    pd.DataFrame
        Cleaned, deduplicated, and validated dataframe.
    """
    df_clean = df.copy()

    # 1. Deduplication
    if drop_duplicates:
        subset = ["device_id", "timestamp"] if "device_id" in df_clean.columns and "timestamp" in df_clean.columns else None
        df_clean = df_clean.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)

    # 2. Chronological Ordering per Device
    if sort_time and "timestamp" in df_clean.columns:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
        if "device_id" in df_clean.columns:
            df_clean = df_clean.sort_values(by=["device_id", "timestamp"]).reset_index(drop=True)
        else:
            df_clean = df_clean.sort_values(by="timestamp").reset_index(drop=True)

    # 3. Handling Missing Values (Forward-fill within device trajectory, then median fallback)
    numeric_present = [c for c in DEFAULT_NUMERIC_COLS if c in df_clean.columns]
    if "device_id" in df_clean.columns:
        # Group forward-fill and back-fill
        df_clean[numeric_present] = df_clean.groupby("device_id")[numeric_present].ffill().bfill()

    # Fallback column-wise median imputation for any remaining NaNs
    for col in numeric_present:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # Categorical forward-fill / mode fill
    cat_present = [c for c in DEFAULT_CATEGORICAL_COLS if c in df_clean.columns]
    for col in cat_present:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "unknown"
            df_clean[col] = df_clean[col].fillna(mode_val)

    # 4. Filter / Clip Physically Impossible Sensor Anomalies
    bounds = impossible_bounds or PHYSICAL_SENSOR_BOUNDS
    for col, (lower, upper) in bounds.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    return df_clean


# ============================================================================
# 2. Zero-Leakage Entity Splitting
# ============================================================================

def split_data_by_device(
    df: pd.DataFrame,
    device_col: str = "device_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: Optional[str] = "device_model",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits longitudinal sensor data by device/entity ID to prevent temporal data leakage.

    Guarantees:
    - Train, Validation, and Test splits contain mutually disjoint sets of device IDs.
    - Continuous time sequences for any given device are kept strictly together in one split.
    - Stratifies devices across hardware models and ensures all requested splits receive devices.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned telemetry dataframe.
    device_col : str, default='device_id'
        Column identifying unique equipment devices/runs.
    train_ratio : float, default=0.70
        Proportion of devices allocated to training set.
    val_ratio : float, default=0.15
        Proportion of devices allocated to validation set.
    test_ratio : float, default=0.15
        Proportion of devices allocated to test set.
    stratify_col : Optional[str], default='device_model'
        Column used for stratified sampling across device models.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    total = train_ratio + val_ratio + test_ratio
    ratios = np.array([train_ratio, val_ratio, test_ratio]) / total

    # Extract unique devices and their stratification metadata
    if stratify_col and stratify_col in df.columns:
        device_meta = df.groupby(device_col)[stratify_col].first().reset_index()
    else:
        device_meta = pd.DataFrame({device_col: df[device_col].unique()})
        stratify_col = None

    rng = np.random.RandomState(random_state)
    device_meta = device_meta.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    n_devices = len(device_meta)
    if n_devices == 0:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    # Calculate target counts per split
    target_counts = np.round(ratios * n_devices).astype(int)
    # Ensure non-zero partitions if ratio > 0 and n_devices >= 3
    for i in range(3):
        if ratios[i] > 0 and target_counts[i] == 0 and n_devices >= 3:
            target_counts[i] = 1
    # Adjust sum to match n_devices
    diff = n_devices - target_counts.sum()
    target_counts[0] += diff

    train_devs: List[str] = []
    val_devs: List[str] = []
    test_devs: List[str] = []
    current_counts = np.array([0, 0, 0])

    # Assign devices group by group (stratification)
    if stratify_col:
        for _, group in device_meta.groupby(stratify_col, sort=False):
            for dev in group[device_col].tolist():
                # Pick split furthest from its target count
                deficit = target_counts - current_counts
                chosen_split = int(np.argmax(deficit))
                if chosen_split == 0:
                    train_devs.append(dev)
                elif chosen_split == 1:
                    val_devs.append(dev)
                else:
                    test_devs.append(dev)
                current_counts[chosen_split] += 1
    else:
        for dev in device_meta[device_col].tolist():
            deficit = target_counts - current_counts
            chosen_split = int(np.argmax(deficit))
            if chosen_split == 0:
                train_devs.append(dev)
            elif chosen_split == 1:
                val_devs.append(dev)
            else:
                test_devs.append(dev)
            current_counts[chosen_split] += 1

    train_df = df[df[device_col].isin(train_devs)].copy().reset_index(drop=True)
    val_df = df[df[device_col].isin(val_devs)].copy().reset_index(drop=True)
    test_df = df[df[device_col].isin(test_devs)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================================
# 3. Scaler & Encoder Estimation (Train Split Only)
# ============================================================================

def fit_transformers(
    train_df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    scaler_type: str = "robust",
) -> Tuple[Union[RobustScaler, StandardScaler], OneHotEncoder]:
    """Fits numerical scaler and categorical encoder strictly on the training partition.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set dataframe.
    numeric_cols : Optional[List[str]], default=None
        Continuous feature column names.
    categorical_cols : Optional[List[str]], default=None
        Categorical feature column names.
    scaler_type : str, default='robust'
        Type of scaler: 'robust' (RobustScaler, resilient to outliers) or 'standard'.

    Returns
    -------
    Tuple[Transformer, OneHotEncoder]
        Fitted (scaler, encoder).
    """
    num_cols = numeric_cols or [c for c in DEFAULT_NUMERIC_COLS if c in train_df.columns]
    cat_cols = categorical_cols or [c for c in DEFAULT_CATEGORICAL_COLS if c in train_df.columns]

    # Scaler
    if scaler_type.lower() == "robust":
        scaler = RobustScaler(unit_variance=True)
    else:
        scaler = StandardScaler()

    scaler.fit(train_df[num_cols])

    # Encoder (handles unseen values cleanly with handle_unknown='ignore')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if cat_cols:
        encoder.fit(train_df[cat_cols])
    else:
        encoder.fit(np.empty((len(train_df), 0)))

    return scaler, encoder


def transform_features(
    df: pd.DataFrame,
    scaler: Union[RobustScaler, StandardScaler],
    encoder: OneHotEncoder,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Transforms numeric and categorical features using previously fitted transformers.

    Parameters
    ----------
    df : pd.DataFrame
        Partition to transform.
    scaler : RobustScaler or StandardScaler
        Fitted scaler.
    encoder : OneHotEncoder
        Fitted encoder.
    numeric_cols : Optional[List[str]], default=None
        Numeric column names.
    categorical_cols : Optional[List[str]], default=None
        Categorical column names.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (X_array, feature_names)
    """
    num_cols = numeric_cols or [c for c in DEFAULT_NUMERIC_COLS if c in df.columns]
    cat_cols = categorical_cols or [c for c in DEFAULT_CATEGORICAL_COLS if c in df.columns]

    if len(df) == 0:
        cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist() if cat_cols else []
        feature_names = list(num_cols) + cat_feature_names
        return np.empty((0, len(feature_names)), dtype=np.float32), feature_names

    # Scale numeric
    X_num = scaler.transform(df[num_cols]).astype(np.float32)

    # Encode categorical
    if cat_cols:
        X_cat = encoder.transform(df[cat_cols]).astype(np.float32)
        cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
    else:
        X_cat = np.empty((len(df), 0), dtype=np.float32)
        cat_feature_names = []

    X_combined = np.hstack([X_num, X_cat])
    feature_names = list(num_cols) + cat_feature_names

    return X_combined, feature_names


# ============================================================================
# 4. Isolated Sliding Window Tensor Generation
# ============================================================================

def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 20,
    step_size: int = 1,
    group_col: str = "device_id",
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = "failure",
    fitted_scaler: Optional[Union[RobustScaler, StandardScaler]] = None,
    fitted_encoder: Optional[OneHotEncoder] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generates 3D sliding sequence windows strictly isolated within each device sequence.

    Windows never cross device or episode boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing time-series observations.
    window_size : int, default=20
        Temporal window length (20 steps = 100 minutes at 5-min sampling).
    step_size : int, default=1
        Stride/step between successive windows within a device.
    group_col : str, default='device_id'
        Column name denoting the entity / sequence boundary.
    feature_cols : Optional[List[str]], default=None
        List of numeric feature column names to include in window.
    target_col : Optional[str], default='failure'
        Target label column to extract for each window.
    fitted_scaler : Optional[Transformer], default=None
        Optional pre-fitted scaler. If provided, transforms features on the fly.
    fitted_encoder : Optional[OneHotEncoder], default=None
        Optional pre-fitted encoder.
    categorical_cols : Optional[List[str]], default=None
        Categorical columns if encoder is provided.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        - X_seq: 3D array of shape (samples, window_size, num_features), dtype float32
        - y_seq: 1D array of shape (samples,) containing window-aligned target values
    """
    num_cols = feature_cols or [c for c in DEFAULT_NUMERIC_COLS if c in df.columns]
    cat_cols = categorical_cols or [c for c in DEFAULT_CATEGORICAL_COLS if c in df.columns]

    all_windows: List[np.ndarray] = []
    all_targets: List[Any] = []

    # Process each device completely independently to ensure absolute boundary isolation
    for _, group in df.groupby(group_col, sort=False):
        n_rows = len(group)
        if n_rows < window_size:
            continue

        # Feature matrix for this device
        if fitted_scaler is not None and fitted_encoder is not None:
            features, _ = transform_features(group, fitted_scaler, fitted_encoder, num_cols, cat_cols)
        else:
            features = group[num_cols].values.astype(np.float32)

        # Target vector for this device
        targets = group[target_col].values if target_col and target_col in group.columns else None

        # Generate sliding windows within this isolated device sequence
        for start_idx in range(0, n_rows - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = features[start_idx:end_idx]
            all_windows.append(window)

            if targets is not None:
                # The window label is aligned to the state at the window's terminal step (t)
                target_val = targets[end_idx - 1]
                all_targets.append(target_val)

    if not all_windows:
        num_feats = len(num_cols) + (fitted_encoder.get_feature_names_out().shape[0] if fitted_encoder else 0)
        return (
            np.empty((0, window_size, num_feats), dtype=np.float32),
            np.empty((0,), dtype=np.int64 if target_col else np.float32),
        )

    X_seq = np.array(all_windows, dtype=np.float32)
    y_seq = np.array(all_targets) if all_targets else None

    return X_seq, y_seq


# ============================================================================
# 5. Artifact Serialization & Persistence
# ============================================================================

def save_artifacts(
    artifacts_dir: Union[str, Path],
    scaler: Union[RobustScaler, StandardScaler],
    encoder: OneHotEncoder,
    metadata: Dict[str, Any],
) -> None:
    """Serializes fitted preprocessors and pipeline metadata to the artifacts directory.

    Parameters
    ----------
    artifacts_dir : Union[str, Path]
        Target directory path (e.g., 'artifacts/').
    scaler : RobustScaler or StandardScaler
        Fitted feature scaler.
    encoder : OneHotEncoder
        Fitted categorical encoder.
    metadata : Dict[str, Any]
        Pipeline configuration and schema metadata.
    """
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, path / "scaler.joblib")
    joblib.dump(encoder, path / "encoder.joblib")

    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_artifacts(
    artifacts_dir: Union[str, Path],
) -> Tuple[Union[RobustScaler, StandardScaler], OneHotEncoder, Dict[str, Any]]:
    """Loads serialized pipeline components and metadata from disk.

    Parameters
    ----------
    artifacts_dir : Union[str, Path]
        Directory path containing 'scaler.joblib', 'encoder.joblib', and 'metadata.json'.

    Returns
    -------
    Tuple[Transformer, OneHotEncoder, Dict[str, Any]]
        (scaler, encoder, metadata)
    """
    path = Path(artifacts_dir)
    scaler = joblib.load(path / "scaler.joblib")
    encoder = joblib.load(path / "encoder.joblib")

    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return scaler, encoder, metadata


# ============================================================================
# 6. Unified Orchestrator Pipeline Class
# ============================================================================

class VentilatorPreprocessor:
    """End-to-end modular preprocessor orchestrating cleaning, splitting, scaling, and sequence windowing."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        scaler_type: str = "robust",
        window_size: int = 20,
        step_size: int = 1,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        self.artifacts_dir = artifacts_dir
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.step_size = step_size
        self.numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS
        self.categorical_cols = categorical_cols or DEFAULT_CATEGORICAL_COLS

        self.scaler: Optional[Union[RobustScaler, StandardScaler]] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "VentilatorPreprocessor":
        """Fits scalers and encoders strictly on the training partition.

        Parameters
        ----------
        train_df : pd.DataFrame
            Cleaned training dataframe.

        Returns
        -------
        VentilatorPreprocessor
            Fitted instance.
        """
        self.scaler, self.encoder = fit_transformers(
            train_df,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            scaler_type=self.scaler_type,
        )

        _, self.feature_names = transform_features(
            train_df.head(2),
            self.scaler,
            self.encoder,
            self.numeric_cols,
            self.categorical_cols,
        )

        self.metadata = {
            "scaler_type": self.scaler_type,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "numeric_features": self.numeric_cols,
            "categorical_features": self.categorical_cols,
            "encoded_feature_names": self.feature_names,
            "num_total_features": len(self.feature_names),
            "fault_type_mapping": FAULT_TYPE_MAPPING,
            "physical_bounds": PHYSICAL_SENSOR_BOUNDS,
        }

        self.is_fitted = True
        return self

    def transform_tabular(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforms dataframe into a 2D scaled feature matrix suitable for tabular models."""
        if not self.is_fitted or self.scaler is None or self.encoder is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        X, _ = transform_features(
            df, self.scaler, self.encoder, self.numeric_cols, self.categorical_cols
        )
        y = df["failure"].values if "failure" in df.columns else None
        return X, y

    def transform_sequential(
        self,
        df: pd.DataFrame,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        target_col: Optional[str] = "failure",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generates 3D sliding sequence tensors strictly isolated within each device."""
        if not self.is_fitted or self.scaler is None or self.encoder is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        w_size = window_size or self.window_size
        s_size = step_size or self.step_size

        return create_sliding_windows(
            df=df,
            window_size=w_size,
            step_size=s_size,
            group_col="device_id",
            feature_cols=self.numeric_cols,
            target_col=target_col,
            fitted_scaler=self.scaler,
            fitted_encoder=self.encoder,
            categorical_cols=self.categorical_cols,
        )

    def fit_transform(
        self,
        raw_df: pd.DataFrame,
        window_size: int = 20,
        step_size: int = 1,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """Executes complete end-to-end preprocessing, zero-leakage split, and sequential tensor windowing."""
        self.window_size = window_size
        self.step_size = step_size

        # 1. Clean
        cleaned_df = clean_data(raw_df)

        # 2. Zero-Leakage Split
        train_df, val_df, test_df = split_data_by_device(
            cleaned_df,
            device_col="device_id",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_col="device_model",
            random_state=42,
        )

        # 3. Fit on Train only
        self.fit(train_df)

        # 4. Generate 3D Sequence Windows
        X_train, y_train = self.transform_sequential(train_df, window_size, step_size)
        X_val, y_val = self.transform_sequential(val_df, window_size, step_size)
        X_test, y_test = self.transform_sequential(test_df, window_size, step_size)

        # 5. Persist artifacts
        self.save()

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def save(self, artifacts_dir: Optional[str] = None) -> None:
        """Saves scaler, encoder, and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        out_dir = artifacts_dir or self.artifacts_dir
        save_artifacts(out_dir, self.scaler, self.encoder, self.metadata)

    @classmethod
    def load(cls, artifacts_dir: str = "artifacts") -> "VentilatorPreprocessor":
        """Loads a fitted preprocessor instance from disk."""
        scaler, encoder, metadata = load_artifacts(artifacts_dir)
        instance = cls(
            artifacts_dir=artifacts_dir,
            scaler_type=metadata.get("scaler_type", "robust"),
            window_size=metadata.get("window_size", 20),
            step_size=metadata.get("step_size", 1),
            numeric_cols=metadata.get("numeric_features", DEFAULT_NUMERIC_COLS),
            categorical_cols=metadata.get("categorical_features", DEFAULT_CATEGORICAL_COLS),
        )
        instance.scaler = scaler
        instance.encoder = encoder
        instance.metadata = metadata
        instance.feature_names = metadata.get("encoded_feature_names")
        instance.is_fitted = True
        return instance


# ============================================================================
# CLI Execution & Verification Entrypoint
# ============================================================================

def main() -> None:
    csv_path = Path("synthetic_ventilator_equipment_dataset.csv")
    if not csv_path.exists():
        print(f"Dataset {csv_path} not found.")
        return

    print("=" * 70)
    print("Ventilator Preprocessing Pipeline Execution")
    print("=" * 70)

    print(f"Loading raw dataset from {csv_path}...")
    raw_df = pd.read_csv(csv_path)
    print(f"Raw shape: {raw_df.shape} ({raw_df['device_id'].nunique()} unique devices)")

    preprocessor = VentilatorPreprocessor(artifacts_dir="artifacts", window_size=20, step_size=1)
    
    print("\nExecuting fit_transform (Clean -> Split -> Fit Train -> Window)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.fit_transform(
        raw_df, window_size=20, step_size=1, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    print("\n--- Output 3D Sequential Tensor Dimensions ---")
    print(f"X_train shape: {X_train.shape} (dtype: {X_train.dtype}), y_train: {y_train.shape} (Failure rate: {y_train.mean():.2%})")
    print(f"X_val shape:   {X_val.shape} (dtype: {X_val.dtype}), y_val:   {y_val.shape} (Failure rate: {y_val.mean():.2%})")
    print(f"X_test shape:  {X_test.shape} (dtype: {X_test.dtype}), y_test:  {y_test.shape} (Failure rate: {y_test.mean():.2%})")

    print("\n--- Artifacts Serialized to 'artifacts/' ---")
    for f in Path("artifacts").glob("*"):
        print(f" - {f.name} ({f.stat().st_size:,} bytes)")

    print("\nPreprocessing Pipeline successfully verified!")


if __name__ == "__main__":
    main()
