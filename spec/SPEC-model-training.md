# Spec: Ventilator Sensor LSTM Model & Training/Evaluation Pipeline — V1

## Objective
Build a modular, extensible deep learning architecture and end-to-end training and evaluation pipeline for ventilator failure detection and diagnosis.
For this minor project milestone, implement specifically the **LSTM** (Long Short-Term Memory) recurrent architecture, while structuring the model registry/factory (`src/models/factory.py`) so future architectures (1D-CNN, GRU, CNN-LSTM, Autoencoder) can be registered seamlessly.
The training pipeline (`src/train.py`) must reuse the deterministic zero-leakage splits from `src/preprocessing.py`, handle class imbalance via class weighting, utilize standard callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`), and produce the serialized model `artifacts/best_lstm_model.keras`.
The evaluation pipeline (`src/evaluate.py`) computes medical-domain metrics (PR-AUC, ROC-AUC, Recall, Precision, Macro F1, Cohen's Kappa, Confusion Matrix) and saves evaluation metrics and diagnostic plots into `artifacts/`.

## Tech Stack
- **Framework:** TensorFlow 2.21.0 / Keras 3.15.1
- **Language & Runtime:** Python 3.11/3.12 (in `.venv`)
- **Supporting Libraries:** `numpy==2.4.6`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `joblib==1.6.0`
- **Testing:** `pytest==9.2.0`, `pytest-cov==7.1.0`

## Commands
- **Run Unit Tests:** `.\.venv\Scripts\pytest.exe tests/test_model.py tests/test_pipeline.py -v`
- **Run Full Test Suite:** `.\.venv\Scripts\pytest.exe -v`
- **Run Model Training CLI:** `.\.venv\Scripts\python.exe src/train.py --epochs 25 --batch-size 64`
- **Run Evaluation CLI:** `.\.venv\Scripts\python.exe src/evaluate.py --model-path artifacts/best_lstm_model.keras`

## Project Structure
```text
my_project/
├── spec/
│   ├── SPEC-preprocessing.md            # Shipped preprocessing spec
│   └── SPEC-model-training.md           # Model architecture & training spec
├── tasks/
│   ├── plan-model.md                    # Technical plan for model & pipeline
│   └── todo-model.md                    # TDD tasks for model & pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Shipped preprocessing pipeline
│   ├── train.py                         # Training pipeline & CLI
│   ├── evaluate.py                      # Evaluation pipeline & diagnostic plots
│   └── models/
│       ├── __init__.py                  # Exports factory and model builders
│       ├── factory.py                   # Extensible Model Factory pattern
│       └── lstm_model.py                # Stacked / Bidirectional LSTM architecture
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py            # Preprocessing test suite (13 passing tests)
│   ├── test_model.py                    # Model shape, forward pass, and training tests
│   └── test_pipeline.py                 # Pipeline training & evaluation metrics tests
└── artifacts/
    ├── best_lstm_model.keras            # Saved best model weights & architecture
    ├── training_history.json            # Epoch-by-epoch loss & metric curves
    ├── evaluation_metrics.json          # Test set evaluation metrics
    ├── learning_curves.png              # Loss & PR-AUC / Accuracy curves plot
    ├── confusion_matrix.png             # Test set confusion matrix visualization
    ├── precision_recall_curve.png       # Test set PR curve visualization
    ├── scaler.joblib                    # Preprocessor scaler artifact
    ├── encoder.joblib                   # Preprocessor encoder artifact
    └── metadata.json                    # Feature & tensor metadata
```

## Code Style & Architecture
- **Factory Pattern (`src/models/factory.py`):**
  A central registry maps architecture names (`"lstm"`, and placeholders for `"cnn"`, `"gru"`, `"cnn_lstm"`, `"autoencoder"`) to builder functions.
- **Model Builder Contract (`src/models/lstm_model.py`):**
  ```python
  def build_lstm_model(
      input_shape: Tuple[int, int] = (20, 16),
      num_classes: int = 1,
      units: List[int] = [64, 32],
      dropout_rate: float = 0.2,
      recurrent_dropout: float = 0.0,
      bidirectional: bool = False,
      learning_rate: float = 1e-3,
      task_type: str = "binary",
  ) -> tf.keras.Model: ...
  ```
- **Layer Stacking:**
  1. `Input(shape=input_shape)` -> `(batch, 20, 16)`
  2. `LSTM(units[0], return_sequences=True)` + `BatchNormalization` + `Dropout(dropout_rate)`
  3. `LSTM(units[1], return_sequences=False)` + `BatchNormalization` + `Dropout(dropout_rate)`
  4. `Dense(32, activation='relu')`
  5. `Dense(1, activation='sigmoid')` for binary classification (or `Dense(num_classes, activation='softmax')` for multiclass).
- **Class Imbalance Handling:**
  Uses scikit-learn's `compute_class_weight('balanced', classes=..., y=...)` passed directly to `model.fit(..., class_weight=...)`.

## Testing Strategy
- **Framework:** `pytest` in `.venv`.
- **Target Files:** `tests/test_model.py` and `tests/test_pipeline.py`.
- **Unit Test Coverage:**
  1. `test_lstm_model_input_output_shapes_binary`: Input shape `(B, 20, 16)` produces output `(B, 1)` with values bounded in $[0, 1]$.
  2. `test_lstm_model_input_output_shapes_multiclass`: Input shape `(B, 20, 16)` with 6 classes produces output `(B, 6)` where rows sum to $1.0 \pm 1\times 10^{-5}$.
  3. `test_lstm_forward_pass_and_backward_step`: Executes 1 mini-batch optimization step; asserts finite loss and gradient weight updates.
  4. `test_model_factory_registration`: `get_model("lstm")` returns a compiled model; unsupported names raise `ValueError`.
  5. `test_compute_evaluation_metrics_binary`: Verifies calculation of PR-AUC, ROC-AUC, F1, Recall, Precision, Cohen's Kappa, and Confusion Matrix against known mock values.
  6. `test_train_pipeline_dry_run`: Tests `train_model` function on small synthetic tensors with callbacks, verifying that history and checkpoint saving operate without error.
  7. `test_evaluate_pipeline_dry_run`: Tests `evaluate_model` function generating metrics dictionary and plot artifacts.

## Boundaries
- **Always do:**
  - Seed random number generators (`numpy`, `random`, `tf.random.set_seed(42)`) for deterministic behavior.
  - Compute class weights on the training split only.
  - Save checkpoints using Keras native `.keras` format (`artifacts/best_lstm_model.keras`).
  - Keep model factory extensible so other models can be registered without modifying the training runner.
- **Ask first:**
  - Changing the default window size ($w=20$) or feature set ($D=16$).
  - Introducing heavy external dependencies not currently in `requirements.txt`.
- **Never do:**
  - Never evaluate the model on training or validation data and report them as test metrics.
  - Never shuffle sequential batches during time-series evaluation.
  - Never suppress non-finite loss (NaN/Inf) warnings.

## Success Criteria
1. Modular LSTM model builder and model factory implemented in `src/models/`.
2. Clean `src/train.py` and `src/evaluate.py` pipelines with callbacks and metrics.
3. 100% of unit tests pass in `pytest`.
4. Successful training run on `synthetic_ventilator_equipment_dataset.csv`, generating `artifacts/best_lstm_model.keras`, `training_history.json`, `evaluation_metrics.json`, and diagnostic plots.
