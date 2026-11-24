import pytest
import numpy as np
import tensorflow as tf

from src.models.factory import get_model, list_available_models, register_model
from src.models.lstm_model import build_lstm_model


def test_lstm_binary_input_output_shape():
    """Verifies that the LSTM model produces output of shape (batch, 1) in [0, 1] for binary tasks."""
    batch_size = 8
    window_size = 20
    num_features = 16

    model = build_lstm_model(
        input_shape=(window_size, num_features),
        num_classes=1,
        units=[32, 16],
        dropout_rate=0.1,
        task_type="binary",
    )

    dummy_input = np.random.randn(batch_size, window_size, num_features).astype(np.float32)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (batch_size, 1)
    assert np.all(output >= 0.0) and np.all(output <= 1.0)
    assert not np.isnan(output).any()


def test_lstm_multiclass_input_output_shape():
    """Verifies that the LSTM model produces valid softmax distributions of shape (batch, num_classes)."""
    batch_size = 8
    window_size = 20
    num_features = 16
    num_classes = 6

    model = build_lstm_model(
        input_shape=(window_size, num_features),
        num_classes=num_classes,
        units=[32, 16],
        dropout_rate=0.1,
        task_type="multiclass",
    )

    dummy_input = np.random.randn(batch_size, window_size, num_features).astype(np.float32)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (batch_size, num_classes)
    assert np.all(output >= 0.0) and np.all(output <= 1.0)
    row_sums = np.sum(output, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_lstm_forward_backward_pass():
    """Verifies that a 1-step training pass executes, computes finite loss, and updates trainable weights."""
    np.random.seed(42)
    tf.random.set_seed(42)

    window_size = 20
    num_features = 16
    batch_size = 16

    model = build_lstm_model(
        input_shape=(window_size, num_features),
        num_classes=1,
        units=[16, 8],
        task_type="binary",
    )

    X_dummy = np.random.randn(batch_size, window_size, num_features).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float32)

    # Initial weights copy
    initial_weights = [w.numpy().copy() for w in model.trainable_weights]

    # Perform a single training step
    loss_metrics = model.train_on_batch(X_dummy, y_dummy)
    assert np.all(np.isfinite(loss_metrics))

    # Verify that at least one weight tensor was updated
    updated_weights = [w.numpy() for w in model.trainable_weights]
    weights_changed = any(
        not np.allclose(init, upd, atol=1e-7)
        for init, upd in zip(initial_weights, updated_weights)
    )
    assert weights_changed, "Model weights were not updated during gradient backpropagation!"


def test_model_factory_registry():
    """Verifies model factory registry, discovery, and unsupported architecture handling."""
    available = list_available_models()
    assert "lstm" in available

    model = get_model("lstm", input_shape=(20, 16), task_type="binary")
    assert isinstance(model, tf.keras.Model)

    with pytest.raises(ValueError, match="Unknown model architecture"):
        get_model("nonexistent_arch", input_shape=(20, 16))

    # Test dynamic registration of a future architecture
    @register_model("mock_cnn")
    def mock_cnn_builder(input_shape, num_classes=1, **kwargs):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(inputs[:, -1, :])
        m = tf.keras.Model(inputs=inputs, outputs=outputs)
        m.compile(optimizer="adam", loss="binary_crossentropy")
        return m

    assert "mock_cnn" in list_available_models()
    mock_model = get_model("mock_cnn", input_shape=(20, 16))
    assert isinstance(mock_model, tf.keras.Model)
