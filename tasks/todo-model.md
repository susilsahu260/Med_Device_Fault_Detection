# Tasks: LSTM Model & Training/Evaluation Pipeline

- [ ] Task 1: Create Model Package Structure
  - Files: `src/models/__init__.py`, `src/models/factory.py`, `src/models/lstm_model.py`
  - Acceptance: Package imports cleanly; factory registry maps architecture strings to builder functions.

- [ ] Task 2: Write Unit Tests Suite (TDD - RED Phase)
  - Files: `tests/test_model.py`, `tests/test_pipeline.py`
  - Acceptance: Comprehensive tests covering input/output shapes, probability bounds, forward/backward gradient update, metrics computation, and pipeline dry-runs.
  - Verify: Run `pytest tests/test_model.py tests/test_pipeline.py` to confirm failure (RED).

- [ ] Task 3: Implement Modular LSTM Architecture & Factory (TDD - GREEN Phase)
  - Files: `src/models/lstm_model.py`, `src/models/factory.py`, `src/models/__init__.py`
  - Acceptance: Clean Keras model builder with Stacked/Bidirectional LSTM, BatchNormalization, Dropout, and focal/class-weight compatibility.
  - Verify: `test_model.py` passes cleanly.

- [ ] Task 4: Implement Training Pipeline (`src/train.py`)
  - Files: `src/train.py`
  - Acceptance: Deterministic data ingestion from preprocessor, class weight computation, callbacks, training loop, history persistence, learning curve plot.

- [ ] Task 5: Implement Evaluation Pipeline (`src/evaluate.py`)
  - Files: `src/evaluate.py`
  - Acceptance: Metrics computation (PR-AUC, ROC-AUC, Recall, Precision, Macro F1, Cohen's Kappa, Confusion Matrix), JSON export, and diagnostic plot saving.
  - Verify: `test_pipeline.py` passes cleanly.

- [ ] Task 6: Run Full Pytest Suite & Execute End-to-End Training
  - Acceptance: All tests pass (100% pass rate); train model on full dataset generating all required artifacts in `artifacts/`.
  - Verify: `pytest -v` and inspect `artifacts/best_lstm_model.keras`, `artifacts/evaluation_metrics.json`.
