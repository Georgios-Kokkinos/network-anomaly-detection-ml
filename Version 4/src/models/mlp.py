"""
mlp.py

Defines the baseline MLP model for multiclass classification.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from src.config import Config


def build_mlp(input_dim: int, num_classes: int, config: Config) -> tf.keras.Model:
    """
    Build and compile a baseline MLP classifier.

    The architecture is configured via `Config.hidden_units` and `Config.dropout`.

    Parameters:
    input_dim (int): Number of input features.
    num_classes (int): Number of output classes.
    config (Config): Configuration object with model hyperparameters.

    Returns:
    tf.keras.Model: Compiled Keras model.
    """
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    # Hidden stack: Dense + BatchNorm + Dropout
    for units in config.hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout)(x)

    # Softmax output for multiclass classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile with Adam optimizer and sparse categorical loss.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
