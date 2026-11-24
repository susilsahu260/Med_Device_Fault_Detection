# Technical Implementation Plan: Preprocessing Pipeline

## Architecture Overview
The preprocessing system transforms raw ventilator sensor telemetry into leakage-free tabular arrays and 3D sliding sequence tensors for deep learning (`1D-CNN + BiLSTM`) and tabular gradient boosting.

```
Raw CSV / DataFrame
      │
      ▼
1. Data Hygiene & Cleaning
   ├── Deduplication on (device_id, timestamp)
   ├── Physical Sanity Boundary Checks (impossible sensor errors only)
   └── Missing Value Imputation (forward-fill per device + median fallback)
      │
      ▼
2. Zero-Leakage Entity/Chronological Splitting
   ├── Stratified Group Split on `device_id` (by `device_model`)
   ├── Train Devices (~70%), Validation Devices (~15%), Test Devices (~15%)
   └── Ensure Train ∩ Val ∩ Test = ∅
      │
      ▼
3. Scaler & Encoder Estimation (Train Split Only)
   ├── Fit RobustScaler / StandardScaler on Train sensor columns
   ├── Fit OneHotEncoder (handle_unknown='ignore') on Train categorical columns (`device_model`)
   └── Serialize fit state to `artifacts/`
      │
      ▼
4. Feature Transformation & Assembly
   ├── Transform Train, Val, Test using fitted transformers
   ├── Produce 2D feature matrices: X_train, X_val, X_test
   └── Produce 1D target arrays: y_failure, y_fault_type
      │
      ▼
5. Isolated Sequential Sliding Window Generation
   ├── Iterate over each `device_id` independently
   ├── Form sliding windows: shape (window_samples, window_size, num_features)
   ├── Align window target with last-step state (or majority state)
   └── Stack device windows into 3D NumPy arrays: (N, T, D)
      │
      ▼
6. Artifact Persistence & Metadata Generation
   ├── `artifacts/scaler.joblib`
   ├── `artifacts/encoder.joblib`
   └── `artifacts/metadata.json` (feature order, bounds, classes, shapes)
```

## Component Breakdown & Dependencies
1. `clean_data`: Pure cleaning function. Depends on pandas.
2. `split_data_by_device`: Stratified group splitter. Depends on scikit-learn / pandas.
3. `fit_transformers`: Scaler and encoder fitter. Fits strictly on training split.
4. `transform_data`: Applies fitted scaler and encoder to any split.
5. `create_sliding_windows`: Groups by device, applies stride-based windowing.
6. `save_artifacts` & `load_artifacts`: Joblib / JSON persistence.
7. `VentilatorPreprocessor`: Unified end-to-end class tying all steps together.

## Risk & Mitigation
- **Risk:** Sliding windows crossing boundaries between devices or disjoint time jumps.
  - **Mitigation:** Strict `groupby('device_id')` loop where window generation is contained inside each device; assert that window length cannot exceed group length.
- **Risk:** Data leakage from standard deviation / mean of validation or test sets.
  - **Mitigation:** Explicit API design where `fit()` takes only `train_df`.
- **Risk:** Overly aggressive outlier clipping removing true failure signals.
  - **Mitigation:** Only filter values outside physically possible physics boundaries (e.g. negative pressure, vibration > 50 mm/s), verified by empirical max/min from the EDA report.
