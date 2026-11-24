"""Model factory and architecture registry for ventilator sensor telemetry models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import tensorflow as tf

# Global registry mapping architecture names to builder functions
_MODEL_REGISTRY: Dict[str, Callable[..., tf.keras.Model]] = {}


def register_model(name: str) -> Callable[[Callable[..., tf.keras.Model]], Callable[..., tf.keras.Model]]:
    """Decorator to register a new model architecture builder function.

    Parameters
    ----------
    name : str
        Canonical identifier for the model (e.g. 'lstm', 'cnn', 'gru').
    """
    def decorator(fn: Callable[..., tf.keras.Model]) -> Callable[..., tf.keras.Model]:
        _MODEL_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def list_available_models() -> List[str]:
    """Lists all registered model architecture names."""
    return sorted(list(_MODEL_REGISTRY.keys()))


def get_model(
    name: str,
    input_shape: Tuple[int, int] = (20, 16),
    num_classes: int = 1,
    **kwargs: Any,
) -> tf.keras.Model:
    """Instantiates and compiles a registered model architecture.

    Parameters
    ----------
    name : str
        Name of the architecture (e.g. 'lstm').
    input_shape : Tuple[int, int], default=(20, 16)
        Window input shape: (window_size, num_features).
    num_classes : int, default=1
        Number of output classes (1 for binary failure detection).
    **kwargs : Any
        Architecture-specific hyperparameters (units, dropout_rate, etc.).

    Returns
    -------
    tf.keras.Model
        Compiled TensorFlow/Keras model.

    Raises
    ------
    ValueError
        If the requested architecture name is not registered.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        available = list_available_models()
        raise ValueError(
            f"Unknown model architecture '{name}'. Available registered models: {available}. "
            f"You can register new architectures using @register_model('{name}')."
        )
    return _MODEL_REGISTRY[key](input_shape=input_shape, num_classes=num_classes, **kwargs)
