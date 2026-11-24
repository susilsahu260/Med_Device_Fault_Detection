import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

from src.evaluate import compute_classification_metrics, evaluate_model
from src.train import compute_class_weights, train_model
from src.models.lstm_model import build_lstm_model


def test_compute_classification_metrics_binary():
    """Verifies that evaluation metrics computation produces expected keys and valid ranges for binary tasks."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int32)
    y_prob = np.array([0.1, 0.2, 0.85, 0.9, 0.35, 0.75, 0.15, 0.4, 0.8, 0.65], dtype=np.float32)

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "macro_f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "cohen_kappa" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["recall"] == 1.0  # With threshold 0.5, all 1s are detected
    assert len(metrics["confusion_matrix"]) == 2  # 2x2 matrix


def test_compute_class_weights():
    """Verifies class weight computation on imbalanced training labels."""
    # 70% class 0, 30% class 1 (matches our ventilator dataset distribution)
    y_train = np.array([0] * 70 + [1] * 30)
    weights = compute_class_weights(y_train)

    assert 0 in weights and 1 in weights
    # Minority class 1 should receive higher weight than majority class 0
    assert weights[1] > weights[0]
    # Sum of weights balanced
    np.testing.assert_approx_equal(weights[0] * 70 + weights[1] * 30, 100.0, significant=2)


def test_train_pipeline_dry_run(tmp_path):
    """Executes a fast 2-epoch training pipeline dry run on synthetic sequence data with callbacks."""
    np.random.seed(42)
    tf.random.set_seed(42)

    n_samples = 40
    window_size = 10
    num_features = 4

    X_train = np.random.randn(n_samples, window_size, num_features).astype(np.float32)
    y_train = np.random.randint(0, 2, size=n_samples).astype(np.int32)

    X_val = np.random.randn(10, window_size, num_features).astype(np.float32)
    y_val = np.random.randint(0, 2, size=10).astype(np.int32)

    model = build_lstm_model(
        input_shape=(window_size, num_features),
        units=[8],
        dropout_rate=0.0,
        task_type="binary",
    )

    history, trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=2,
        batch_size=16,
        artifacts_dir=str(tmp_path),
        save_plots=True,
    )

    # Check history structure
    assert "loss" in history
    assert "val_loss" in history
    assert len(history["loss"]) == 2

    # Check checkpoint artifact was saved
    assert (tmp_path / "best_lstm_model.keras").exists() or (tmp_path / "latest_model.keras").exists()
    assert (tmp_path / "training_history.json").exists()


def test_evaluate_pipeline_dry_run(tmp_path):
    """Executes an evaluation run on test tensors and verifies saved reports and plots."""
    n_samples = 30
    window_size = 10
    num_features = 4

    X_test = np.random.randn(n_samples, window_size, num_features).astype(np.float32)
    y_test = np.random.randint(0, 2, size=n_samples).astype(np.int32)

    model = build_lstm_model(
        input_shape=(window_size, num_features),
        units=[8],
        task_type="binary",
    )

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        artifacts_dir=str(tmp_path),
        save_plots=True,
    )

    assert "accuracy" in metrics
    assert "pr_auc" in metrics
    assert (tmp_path / "evaluation_metrics.json").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "precision_recall_curve.png").exists()
    assert (tmp_path / "roc_curve.png").exists()
