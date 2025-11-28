# time_feature_benchmark/model/lstm_models.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Literal
import numpy as np
import pandas as pd

from model.models import Model, ModelSpec


# ----- LSTM spec -----

@dataclass(frozen=True)
class LSTMModelSpec(ModelSpec):
    """
    Spec to construct & train an LSTM model (TensorFlow/Keras under the hood).
    Parameters mirror what LSTMModel expects.

    Construction / architecture:
      - task:       'regression' or 'classification'
      - hidden_size, num_layers, dropout
      - pad_value:  must match your LSTMTransformer's pad_value
      - loss, metrics: optional overrides (defaults depend on task)
      - lr:         learning rate

    Training:
      - batch_size, epochs, verbose
      - early_stopping_patience
      - reduce_lr_patience, reduce_lr_factor, min_lr
      - seed: optional reproducibility hook (set externally in TF if needed)
    """
    # Architecture
    task: Literal["regression", "classification"] = "regression"
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    pad_value: float = 0.0
    lr: float = 1e-3
    loss: str = "mae"
    metrics: Optional[List[str]] = "mae"

    # Training
    batch_size: int = 64
    epochs: int = 15
    verbose: int = 1
    early_stopping_patience: int = 3
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    seed: Optional[int] = None

    def to_params(self) -> Dict[str, Any]:
        """Params dict for LSTMModel constructor/train loop."""
        return asdict(self)

    def create(self) -> Model:
        # instantiate directly to avoid circular imports
        return LSTMModel(params=self.to_params())


class LSTMModel(Model):
    """
    Keras LSTM wrapper assuming inputs are padded to a fixed T with a known pad_value.
    - Works for regression (default). Set params["task"]="classification" for binary classification.
    - Expects X to be (N, T, F) float arrays (or convertible).
    - Masking is enabled with params["pad_value"] (default 0.0).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model_ = None
        self.history_ = None

    # ---------- utils ----------
    def _to_array(self, X):
        # Accept Prepared-like objects
        if hasattr(X, "X"):
            X = X.X
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"LSTMModel expects X with shape (N,T,F); got {X.shape}")
        return X.astype(np.float32, copy=False)

    def _to_target(self, y):
        # Accept Prepared-like objects
        if hasattr(y, "y"):
            y = y.y
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
        y = np.asarray(y).reshape(-1)
        return y.astype(np.float32, copy=False)

    def _build_model(self, input_shape):
        try:
            import tensorflow as tf
        except Exception as e:
            raise RuntimeError("TensorFlow is required to use LSTMModel.") from e

        tf.config.optimizer.set_jit(False)  # disable XLA JIT globally
        
        task = self.params.get("task", "regression")
        hidden_size = int(self.params.get("hidden_size", 64))
        num_layers = int(self.params.get("num_layers", 1))
        dropout = float(self.params.get("dropout", 0.0))
        lr = float(self.params.get("lr", self.params.get("learning_rate", 1e-3)))
        pad_value = float(self.params.get("pad_value", 0.0))
        loss = self.params.get("loss", "mae" if task == "regression" else "binary_crossentropy")
        metrics = self.params.get("metrics", ["mae"] if task == "regression" else ["accuracy"])
        if isinstance(metrics, str):  # accept "mae", "mse", etc.
            metrics = [metrics]

        inputs = tf.keras.Input(shape=input_shape)  # (T, F)
        x = tf.keras.layers.Masking(mask_value=pad_value)(inputs)

        # LSTM stack
        for _ in range(max(0, num_layers - 1)):
            x = tf.keras.layers.LSTM(hidden_size, return_sequences=True, dropout=dropout)(x)
        x = tf.keras.layers.LSTM(hidden_size, return_sequences=False, dropout=dropout)(x)

        if task == "classification":
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = tf.keras.layers.Dense(1, activation=None)(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
        return model

    # ---------- Model interface ----------
    def train(self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> None:
        import tensorflow as tf  # local import to avoid hard dependency elsewhere
        X_tr = self._to_array(X_train)
        X_va = self._to_array(X_val)
        y_tr = self._to_target(y_train)
        y_va = self._to_target(y_val)

        if self.model_ is None:
            # input_shape = (T, F)
            self.model_ = self._build_model(input_shape=X_tr.shape[1:])

        batch_size = int(self.params.get("batch_size", 64))
        epochs = int(self.params.get("epochs", 15))
        verbose = int(self.params.get("verbose", 1))
        patience = int(self.params.get("early_stopping_patience", 3))
        rlrop_patience = int(self.params.get("reduce_lr_patience", 2))
        rlrop_factor = float(self.params.get("reduce_lr_factor", 0.5))
        min_lr = float(self.params.get("min_lr", 1e-6))

        callbacks = []
        if patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ))
        if rlrop_patience > 0:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=rlrop_patience, factor=rlrop_factor, min_lr=min_lr
            ))

        self.history_ = self.model_.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        X_arr = self._to_array(X)
        preds = self.model_.predict(X_arr, verbose=0).reshape(-1)
        return pd.Series(preds)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        if self.model_ is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        X_te = self._to_array(X_test)
        y_te = self._to_target(y_test)

        results = self.model_.evaluate(X_te, y_te, verbose=0)
        names = self.model_.metrics_names  # e.g., ['loss', 'mae']
        if not isinstance(results, (list, tuple)):
            results = [results]
        return {name: float(val) for name, val in zip(names, results)}
