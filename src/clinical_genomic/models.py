"""Classical and Keras models used by the training script and API."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


def build_mlp_classifier(input_dim: int, seed: int = 42):
    import tensorflow as tf
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def build_mlp_regressor(input_dim: int, seed: int = 42):
    import tensorflow as tf
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_keras_classifier(X_train, y_train, X_val, y_val, seed: int = 42):
    import tensorflow as tf
    model = build_mlp_classifier(X_train.shape[1], seed=seed)
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=40, batch_size=32, callbacks=[cb], verbose=0,
    )
    return model


def train_keras_regressor(X_train, y_train, X_val, y_val, seed: int = 42):
    import tensorflow as tf
    model = build_mlp_regressor(X_train.shape[1], seed=seed)
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=40, batch_size=32, callbacks=[cb], verbose=0,
    )
    return model
