# Spec: Ventilator Sensor Preprocessing Pipeline — V1

## Objective
Build a robust, modular, and leak-free preprocessing pipeline (`src/preprocessing.py`) for the ventilator equipment sensor telemetry dataset (`synthetic_ventilator_equipment_dataset.csv`). The pipeline ingests raw longitudinal equipment sensor records, performs data hygiene, enforces zero-leakage device-grouped splits, handles tabular feature encoding and scaling, and constructs 3D sequential sliding window tensors `(samples, window_size, features)` strictly isolated within individual equipment sequences for 1D-CNN + BiLSTM deep learning models. All transformation artifacts (scalers, encoders, column metadata) must be persisted to `artifacts/` for reproducible inference and dashboard integration.

## Tech Stack
- **Language:** Python 3.12+ (in project virtual environment `.venv`)
- **Core Libraries:** `pandas==3.1.1`, `numpy==2.4.6`, `scikit-learn==1.9.1`, `joblib==1.6.0`
- **Testing Framework:** `pytest==9.2.0`, `pytest-cov==7.1.0`
- **Downstream Consumers:** TensorFlow / Keras 3.x (1D-CNN + BiLSTM models), Streamlit dashboard

## Commands
- **Run Unit Tests:** `.\.venv\Scripts\pytest.exe tests/test_preprocessing.py -v`
- **Run with Coverage:** `.\.venv\Scripts\pytest.exe tests/test_preprocessing.py --cov=src -v`
- **Execute Pipeline CLI / Script:** `.\.venv\Scripts\python.exe src/preprocessing.py`

## Project Structure
```text
my_project/
├── synthetic_ventilator_equipment_dataset.csv  # Raw dataset (20,000 rows, 18 columns)
├── ventilator_data_analysis_report.md          # Grounding EDA report
├── spec/
│   └── SPEC-preprocessing.md                   # This specification contract
├── tasks/
│   ├── plan.md                                 # Technical architecture & dependency plan
│   └── todo.md                                 # TDD implementation tasks
├── src/
│   ├── __init__.py
│   └── preprocessing.py                        # Modular preprocessing module & pipeline class
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py                   # Comprehensive unit tests suite
└── artifacts/                                  # Serialized pipeline artifacts
    ├── scaler.joblib                           # Fitted scikit-learn Scaler (RobustScaler / StandardScaler)
    ├── encoder.joblib                          # Fitted OneHotEncoder for categorical metadata
    └── metadata.json                           # Feature names, tensor dimensions, target mappings, thresholds
```

## Code Style & Conventions
- **PEP 8 Compliance:** 4 spaces, snake_case functions, PascalCase classes, UPPER_SNAKE constants.
- **Type Annotations:** Full typing on all public functions (`pd.DataFrame`, `np.ndarray`, `Tuple`, `Dict`, `Optional`).
- **Comprehensive Docstrings:** Google/NumPy style docstrings explaining parameters, returns, and domain rationale.
- **Pure Functions & Orchestrator Pattern:** Each transformation step exists as an independent, testable function, coordinated by a unified `VentilatorPreprocessor` class.

Example Code Style:
```python
def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    impossible_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Cleans raw sensor dataframe by removing duplicates and physically impossible values.
    
    Preserves genuine fault spikes (e.g. overheating, vibration surges).
    """
```

## Testing Strategy
- **Framework:** `pytest` running inside the local Python virtual environment.
- **Test File:** `tests/test_preprocessing.py`.
- **Key Test Requirements (DAMP, Arrange-Act-Assert):**
  1. `test_handle_missing_values`: Validates imputation (median/forward-fill) for missing sensor data.
  2. `test_handle_duplicates`: Validates deduplication on `(device_id, timestamp)` or full rows.
  3. `test_outlier_filtering_bounds`: Confirms physically impossible sensor values are clipped or dropped while genuine fault spikes (e.g., $T > 50^\circ\text{C}$, $\text{vibration} > 2.0\text{ mm/s}$) are strictly preserved.
  4. `test_leakage_free_splitting`: Asserts that `train`, `val`, and `test` splits contain mutually exclusive sets of `device_id`s ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$).
  5. `test_categorical_encoding_unseen_values`: Confirms encoder handles unseen categories (e.g., new `device_model`) gracefully via `handle_unknown='ignore'`.
  6. `test_scaler_fitted_on_train_only`: Verifies scaler statistics ($\mu, \sigma$ or median/IQR) are computed purely from training devices, not validation/test devices.
  7. `test_sliding_window_boundary_isolation`: Confirms sliding window generation resets per sequence/device so no window blends data across distinct equipment or disjoint episodes.
  8. `test_sliding_window_output_shapes`: Confirms 3D array output `(samples, window_size, features)` and aligned target label arrays `(samples,)`.
  9. `test_artifact_persistence_and_reload`: Verifies `artifacts/` files can be saved, reloaded, and used to transform new incoming sensor telemetry identically.

## Boundaries
- **Always do:**
  - Split dataset by `device_id` (or sequence ID) **before** fitting scalers and encoders.
  - Reset sliding window generator across devices/episodes.
  - Save complete feature column ordering and scaling metadata to `artifacts/metadata.json`.
- **Ask first:**
  - Dropping any feature column entirely from downstream training.
  - Modifying the underlying raw CSV dataset on disk.
- **Never do:**
  - Never shuffle raw time-series rows randomly across train/val/test splits.
  - Never allow sliding windows to cross device or sequence boundaries.
  - Never clip genuine fault signatures (e.g. clipping high vibration or high temperature to normal range).
  - Never commit binary artifacts or temporary files to untracked git paths.

## Success Criteria
1. `src/preprocessing.py` implements a modular, self-contained pipeline with zero data leakage.
2. `tests/test_preprocessing.py` contains 10+ thorough unit tests covering all edge cases.
3. All tests pass cleanly (`pytest` returns exit code 0).
4. Running the preprocessing script on `synthetic_ventilator_equipment_dataset.csv` generates:
   - 3D sequence tensors of shape `(N, 20, num_features)` for train, val, and test.
   - Serialized artifacts in `artifacts/`: `scaler.joblib`, `encoder.joblib`, `metadata.json`.
   - Complete verification of shapes, absence of NaNs, and boundary isolation.

## Out of Scope (Deferred to V2)
- Online streaming Kafka / WebSocket consumer (V1 focuses on batch and windowed inference).
- Automated hyperparameter search for optimal window size (fixed at $w=20$ for V1 based on 20-step fault episodes discovered in EDA).
