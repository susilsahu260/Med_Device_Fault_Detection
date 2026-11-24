# Technical Implementation Plan: LSTM Model & Training/Evaluation Pipeline

## Architecture Overview
The model subsystem consists of an extensible Model Factory, an LSTM model implementation, a training harness with callbacks, and a rigorous evaluation module.

```
                    ┌────────────────────────┐
                    │    Model Factory       │
                    │ (src/models/factory.py)│
                    └───────────┬────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
        ┌───────────────────┐       ┌──────────────────────┐
        │  build_lstm_model │       │  [Future: CNN, GRU,  │
        │(models/lstm_model)│       │   Autoencoder, etc.] │
        └─────────┬─────────┘       └──────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ Training Pipeline (src/train.py)                       │
│ 1. Load data via VentilatorPreprocessor                │
│ 2. Compute balanced class weights on Train labels      │
│ 3. Build model from factory                            │
│ 4. Configure EarlyStopping, ReduceLROnPlateau,         │
│    ModelCheckpoint(artifacts/best_lstm_model.keras)    │
│ 5. Execute model.fit(X_train, y_train, validation_data)│
│ 6. Save history & learning curves to artifacts/        │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Evaluation Pipeline (src/evaluate.py)                  │
│ 1. Load artifacts/best_lstm_model.keras                │
│ 2. Predict on X_test                                   │
│ 3. Compute PR-AUC, ROC-AUC, Recall, Precision,         │
│    Macro F1, Cohen's Kappa, Confusion Matrix           │
│ 4. Save evaluation_metrics.json, confusion_matrix.png, │
│    precision_recall_curve.png, roc_curve.png           │
└────────────────────────────────────────────────────────┘
```

## Modular Extension Points
1. To add a new architecture (e.g. 1D-CNN or GRU):
   - Create `src/models/cnn_model.py`.
   - Register it with `@register_model("cnn")` in `src/models/factory.py`.
   - The training and evaluation pipelines require zero code modifications—just specify `--model cnn`.
