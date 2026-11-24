"""Evaluation pipeline and diagnostic reporting for ventilator sensor telemetry models."""

from __future__ import annotations

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
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive domain-appropriate evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels (0 or 1).
    y_pred_prob : np.ndarray
        Predicted probabilities of failure (shape: (N,) or (N, 1)).
    threshold : float, default=0.5
        Decision threshold for converting probabilities to binary classes.

    Returns
    -------
    Dict[str, Any]
        Dictionary of scalar metrics and confusion matrix.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_pred_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # ROC-AUC & PR-AUC (handle edge case where test set only has 1 class)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "cohen_kappa": round(kappa, 4),
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plots and saves a styled confusion matrix heatmap."""
    labels = class_names or ["Normal (0)", "Failure (1)"]
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Ground-Truth", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plots and saves the Precision-Recall curve."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_pred_prob).ravel()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1f77b4", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title("Precision-Recall Curve (Medical Fault Detection)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_pred_prob).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#ff7f0e", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance Baseline")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([-0.02, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close(fig)


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    artifacts_dir: str = "artifacts",
    threshold: float = 0.5,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Evaluates a trained model on the held-out test split and writes reports/plots to artifacts/.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model instance.
    X_test : np.ndarray
        Test features tensor (samples, window_size, num_features).
    y_test : np.ndarray
        Ground-truth test labels.
    artifacts_dir : str, default='artifacts'
        Target directory to save metric JSON and diagnostic plots.
    threshold : float, default=0.5
        Decision threshold for binary failure detection.
    save_plots : bool, default=True
        Whether to generate and save visual plots (confusion matrix, PR curve, ROC curve).

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed test set metrics.
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # 1. Run Predictions
    y_prob = model.predict(X_test, batch_size=64, verbose=0)
    y_prob = np.asarray(y_prob).ravel()

    # 2. Compute Metrics
    metrics = compute_classification_metrics(y_test, y_prob, threshold=threshold)

    # 3. Save JSON Metrics Report
    metrics_file = artifacts_path / "evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 4. Generate Diagnostic Visualizations
    if save_plots:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            save_path=artifacts_path / "confusion_matrix.png",
        )
        plot_precision_recall_curve(
            y_test,
            y_prob,
            save_path=artifacts_path / "precision_recall_curve.png",
        )
        plot_roc_curve(
            y_test,
            y_prob,
            save_path=artifacts_path / "roc_curve.png",
        )

    return metrics
