# Tasks: Ventilator Sensor Preprocessing Pipeline

- [x] Task 1: Initialize Project Directories and Init Files
  - Acceptance: `src/`, `tests/`, `artifacts/` directories exist with `__init__.py`.
  - Verify: Directory check via python / terminal.
  - Files: `src/__init__.py`, `tests/__init__.py`.

- [x] Task 2: Write Unit Tests Suite (TDD - RED Phase)
  - Acceptance: Write comprehensive `tests/test_preprocessing.py` covering:
    - Missing value handling and imputation
    - Duplicate detection and removal
    - Physical boundary filtering vs. fault spike preservation
    - Device-grouped zero-leakage splitting
    - Unseen categorical value handling
    - Scaler fitting isolation to training set
    - Sequential sliding window boundary isolation across devices/episodes
    - 3D tensor output shapes `(samples, window_size, features)` and target alignment
    - Artifact serialization and re-loading for inference
  - Verify: Run `pytest tests/test_preprocessing.py -v` and observe failures (RED).
  - Files: `tests/test_preprocessing.py`.

- [x] Task 3: Implement Modular Preprocessing Functions & Pipeline Class (TDD - GREEN Phase)
  - Acceptance: Implement `src/preprocessing.py` with:
    - `clean_data`
    - `split_data_by_device`
    - `fit_transformers`
    - `transform_features`
    - `create_sliding_windows`
    - `save_artifacts` & `load_artifacts`
    - `VentilatorPreprocessor` class
  - Verify: Run `pytest tests/test_preprocessing.py -v` and confirm all tests pass (GREEN).
  - Files: `src/preprocessing.py`.

- [x] Task 4: Verify Real-World Execution on `synthetic_ventilator_equipment_dataset.csv`
  - Acceptance: Run `src/preprocessing.py` as main execution; produce:
    - `artifacts/scaler.joblib`
    - `artifacts/encoder.joblib`
    - `artifacts/metadata.json`
    - Verify 3D tensor shapes and summary logging.
  - Verify: Check files in `artifacts/` and run verification assertions.
  - Files: `src/preprocessing.py`, `artifacts/*`.

- [x] Task 5: Run Full Pytest Suite with Coverage & Final Refactor
  - Acceptance: 100% tests passing, clean code, no regressions (94% coverage achieved).
  - Verify: `pytest tests/test_preprocessing.py --cov=src -v`
  - Files: `src/preprocessing.py`, `tests/test_preprocessing.py`.
