"""Training pipeline and experiment orchestration for ventilator sensor telemetry models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from src.evaluate import evaluate_model
from src.models.factory import get_model
from src.preprocessing import VentilatorPreprocessor


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Computes balanced inverse-frequency class weights for training under class imbalance.

    Parameters
    ----------
    y_train : np.ndarray
        Ground-truth training labels (e.g. 70% normal, 30% failure).

    Returns
    -------
    Dict[int, float]
        Dictionary mapping class integer to its balanced loss multiplier.
    """
    y_flat = np.asarray(y_train).ravel()
    classes = np.unique(y_flat)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_flat)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def plot_learning_curves(
    history_dict: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plots and saves loss and metric learning curves across epochs."""
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Curve
    axes[0].plot(epochs_range, history_dict["loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history_dict:
        axes[0].plot(epochs_range, history_dict["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary Crossentropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # PR-AUC or Accuracy Curve
    metric_key = "pr_auc" if "pr_auc" in history_dict else "accuracy"
    val_metric_key = f"val_{metric_key}"
    metric_label = "PR-AUC" if metric_key == "pr_auc" else "Accuracy"

    axes[1].plot(epochs_range, history_dict[metric_key], label=f"Train {metric_label}", color="#2ca02c", lw=2)
    if val_metric_key in history_dict:
        axes[1].plot(epochs_range, history_dict[val_metric_key], label=f"Val {metric_label}", color="#d62728", lw=2)
    axes[1].set_title(f"Training & Validation {metric_label}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel(metric_label, fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close(fig)


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    artifacts_dir: str = "artifacts",
    model_filename: str = "best_lstm_model.keras",
    patience_early_stopping: int = 8,
    patience_reduce_lr: int = 4,
    save_plots: bool = True,
) -> Tuple[Dict[str, List[float]], tf.keras.Model]:
    """Trains model with standard callbacks, class balancing, and learning rate scheduling.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled model instance.
    X_train : np.ndarray
        Training feature tensors (N, window_size, features).
    y_train : np.ndarray
        Training target vector.
    X_val : np.ndarray
        Validation feature tensors.
    y_val : np.ndarray
        Validation target vector.
    epochs : int, default=25
        Maximum training epochs.
    batch_size : int, default=64
        Batch size.
    artifacts_dir : str, default='artifacts'
        Artifacts directory to save best checkpoint and history.
    model_filename : str, default='best_lstm_model.keras'
        Name of saved model checkpoint.
    patience_early_stopping : int, default=8
        Patience epochs before early stopping.
    patience_reduce_lr : int, default=4
        Patience epochs before learning rate reduction.
    save_plots : bool, default=True
        Whether to generate learning curves plot.

    Returns
    -------
    Tuple[Dict[str, List[float]], tf.keras.Model]
        (history_dict, best_trained_model)
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = artifacts_path / model_filename

    # Compute balanced class weights
    class_weights = compute_class_weights(y_train)

    # Standard production callbacks
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Format history dictionary
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Save history JSON
    history_file = artifacts_path / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    # Save learning curves visualization
    if save_plots:
        plot_learning_curves(history_dict, save_path=artifacts_path / "learning_curves.png")

    # If ModelCheckpoint saved the best weights, reload them
    if checkpoint_filepath.exists():
        try:
            best_model = tf.keras.models.load_model(str(checkpoint_filepath))
            return history_dict, best_model
        except Exception:
            pass

    return history_dict, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ventilator Sensor Telemetry Models")
    parser.add_argument("--data-path", type=str, default="synthetic_ventilator_equipment_dataset.csv")
    parser.add_argument("--model", type=str, default="lstm", help="Model architecture (e.g. 'lstm')")
    parser.add_argument("--window-size", type=int, default=20, help="Window size (default: 20 steps / 100 min)")
    parser.add_argument("--step-size", type=int, default=1, help="Stride step size (default: 1)")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer initial learning rate")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    # Reproducibility seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    csv_path = Path(args.data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file {csv_path} does not exist.")

    print("=" * 70)
    print("Ventilator Equipment Sensor Deep Learning Training Pipeline")
    print(f"Architecture: {args.model.upper()} | Window: {args.window_size} | Epochs: {args.epochs}")
    print("=" * 70)

    # 1. Ingest & Preprocess using Preprocessor
    print("\n[Step 1/4] Preprocessing raw data with zero-leakage device grouping...")
    raw_df = pd.read_csv(csv_path)
    preprocessor = VentilatorPreprocessor(artifacts_dir=args.artifacts_dir, window_size=args.window_size, step_size=args.step_size)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.fit_transform(
        raw_df, window_size=args.window_size, step_size=args.step_size
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f" - Train Tensors: {X_train.shape}, Failure Rate: {y_train.mean():.2%}")
    print(f" - Val Tensors:   {X_val.shape}, Failure Rate: {y_val.mean():.2%}")
    print(f" - Test Tensors:  {X_test.shape}, Failure Rate: {y_test.mean():.2%}")

    # 2. Build Model from Factory
    print(f"\n[Step 2/4] Building {args.model.upper()} model via Model Factory (input_shape={input_shape})...")
    model = get_model(
        name=args.model,
        input_shape=input_shape,
        num_classes=1,
        units=[64, 32],
        dropout_rate=0.2,
        learning_rate=args.learning_rate,
        task_type="binary",
    )
    model.summary()

    # 3. Train Model
    print("\n[Step 3/4] Training model with callbacks and balanced class weights...")
    history, best_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        artifacts_dir=args.artifacts_dir,
        model_filename="best_lstm_model.keras",
    )

    # 4. Evaluate on Unseen Test Split
    print("\n[Step 4/4] Evaluating best model on held-out test split...")
    metrics = evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        artifacts_dir=args.artifacts_dir,
        save_plots=True,
    )

    print("\n" + "=" * 70)
    print("Test Set Diagnostic Evaluation Results")
    print("=" * 70)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:<20}: {v}")
    print(f"  confusion_matrix    : {metrics['confusion_matrix']}")
    print("=" * 70)
    print("Pipeline completed successfully! All model weights & plots saved to 'artifacts/'.")


if __name__ == "__main__":
    main()
