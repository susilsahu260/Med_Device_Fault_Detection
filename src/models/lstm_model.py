"""Stacked and Bidirectional LSTM Architecture for Ventilator Sensor Telemetry."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import tensorflow as tf

from src.models.factory import register_model


@register_model("lstm")
def build_lstm_model(
    input_shape: Tuple[int, int] = (20, 16),
    num_classes: int = 1,
    units: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.0,
    bidirectional: bool = False,
    dense_units: int = 32,
    learning_rate: float = 1e-3,
    task_type: str = "binary",
) -> tf.keras.Model:
    """Builds and compiles a deep LSTM neural network for temporal sensor sequence classification.

    Parameters
    ----------
    input_shape : Tuple[int, int], default=(20, 16)
        Shape of sliding window input (window_size, num_features).
    num_classes : int, default=1
        Number of target classes (1 for binary failure detection).
    units : Optional[List[int]], default=None
        Number of hidden units per LSTM layer. Defaults to [64, 32].
    dropout_rate : float, default=0.2
        Dropout probability applied after recurrent and dense layers.
    recurrent_dropout : float, default=0.0
        Dropout probability on recurrent connections (keep 0.0 for cuDNN optimization).
    bidirectional : bool, default=False
        Whether to wrap LSTM layers in Bidirectional wrappers.
    dense_units : int, default=32
        Units in the dense bottleneck feature layer preceding classification.
    learning_rate : float, default=1e-3
        Adam optimizer learning rate.
    task_type : str, default='binary'
        Classification task type: 'binary' or 'multiclass'.

    Returns
    -------
    tf.keras.Model
        Compiled Keras Model instance.
    """
    layer_units = units if units is not None else [64, 32]

    # Input Layer
    inputs = tf.keras.Input(shape=input_shape, name="sensor_window_input")
    x = inputs

    # Stacked LSTM Layers
    num_lstm_layers = len(layer_units)
    for idx, u in enumerate(layer_units):
        is_last_lstm = (idx == num_lstm_layers - 1)
        return_sequences = not is_last_lstm

        lstm_layer = tf.keras.layers.LSTM(
            units=u,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f"lstm_layer_{idx + 1}",
        )

        if bidirectional:
            x = tf.keras.layers.Bidirectional(lstm_layer, name=f"bidirectional_lstm_{idx + 1}")(x)
        else:
            x = lstm_layer(x)

        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{idx + 1}")(x)

    # Dense Latent Head
    if dense_units > 0:
        x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_latent")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="dense_dropout")(x)

    # Output Layer & Compilation
    if task_type == "binary" or num_classes == 1:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="failure_probability")(x)
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ]
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="fault_distribution")(x)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy"]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"ventilator_{task_type}_lstm")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
