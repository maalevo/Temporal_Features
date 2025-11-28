# time_feature_benchmark/model/xgb_models.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Literal, List
import numpy as np
import pandas as pd

from .models import Model, ModelSpec


# ---------------- XGBoost spec ----------------

@dataclass(frozen=True)
class XGBModelSpec(ModelSpec):
    """
    Spec for XGBoost-style models.
    `variant` selects the concrete class here:
      - 'plain'             -> XGBModel
      - 'no_bucket'         -> NoBucketXGBModel
      - 'clustering_bucket' -> ClusteringBucketXGBModel
      - 'prefixlen_bucket'  -> PrefixLenBucketXGBModel

    Core hyperparameters mirror XGBRegressor/XGBClassifier:
      - objective: e.g., 'reg:squarederror', 'binary:logistic'
      - n_estimators, learning_rate, max_depth, subsample, colsample_bytree
      - reg_alpha, reg_lambda, tree_method, random_state, verbosity
      - early_stopping_rounds is used in .train() via eval_set
    """
    variant: Literal["plain", "no_bucket", "clustering_bucket", "prefixlen_bucket"] = "plain"

    # Core XGB params
    objective: str = "reg:squarederror"
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    tree_method: Optional[str] = None   # e.g., 'hist', 'gpu_hist'
    random_state: Optional[int] = None
    verbosity: int = 1

    # Training conveniences
    early_stopping_rounds: Optional[int] = None
    eval_metric: Optional[str] = None  # defaulted based on objective if None

    def to_params(self) -> Dict[str, Any]:
        return asdict(self)

    def create(self) -> Model:
        params = self.to_params().copy()
        variant = params.pop("variant", "plain")
        # leave early_stopping_rounds/eval_metric to the train() call
        # (keep them in params so the instance can access them)

        if variant == "plain":
            return XGBModel(params=params)
        if variant == "no_bucket":
            return NoBucketXGBModel(params=params)
        if variant == "clustering_bucket":
            return ClusteringBucketXGBModel(params=params)
        if variant == "prefixlen_bucket":
            return PrefixLenBucketXGBModel(params=params)
        raise ValueError(f"Unknown XGB variant '{variant}'")


# ---------------- Concrete models ----------------

class XGBModel(Model):
    """
    XGBoost wrapper for regression/binary classification.
    - Accepts 2D features (N, F). If a Prepared-like object is passed, uses .X / .y.
    - Chooses regressor/classifier from 'objective'.
    - Supports early stopping using the provided validation set.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.model_ = None  # xgboost.XGBRegressor or XGBClassifier

    # ----- utils -----
    def _is_classification(self) -> bool:
        obj = str(self.params.get("objective", "")).lower()
        return ("logistic" in obj) or ("binary:" in obj) or ("multi" in obj)

    def _to_array(self, X):
        if hasattr(X, "X"):
            X = X.X
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"XGBModel expects 2D array (N,F); got shape {X.shape}")
        return X

    def _to_target(self, y):
        if hasattr(y, "y"):
            y = y.y
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
        y = np.asarray(y).reshape(-1)
        return y

    def _build_model(self):
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("xgboost is required to use XGBModel.") from e

        p = self.params.copy()
        objective = p.pop("objective", "reg:squarederror")
        tree_method = p.pop("tree_method", None)
        if tree_method is not None:
            p["tree_method"] = tree_method
        verbosity = p.pop("verbosity", 1)

        clf_mode = self._is_classification()
        if clf_mode:
            # For binary classification; for multi-class, extend as needed
            model = xgb.XGBClassifier(objective=objective, verbosity=verbosity, **p)
        else:
            model = xgb.XGBRegressor(objective=objective, verbosity=verbosity, **p)
        return model

    # ----- Model interface -----
    def train(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
              y_train: pd.Series, y_val: pd.Series) -> None:
        X_tr = self._to_array(X_train)
        X_va = self._to_array(X_val)
        y_tr = self._to_target(y_train)
        y_va = self._to_target(y_val)

        if self.model_ is None:
            self.model_ = self._build_model()

        # fit settings
        es_rounds = self.params.get("early_stopping_rounds", None)
        eval_metric = self.params.get("eval_metric", None)
        if eval_metric is None:
            eval_metric = "logloss" if self._is_classification() else "rmse"

        self.model_.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=eval_metric,
            verbose=False,
            early_stopping_rounds=es_rounds,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        X_arr = self._to_array(X)

        if self._is_classification():
            # return probability of positive class when available
            if hasattr(self.model_, "predict_proba"):
                preds = self.model_.predict_proba(X_arr)[:, 1]
            else:
                preds = self.model_.predict(X_arr)
        else:
            preds = self.model_.predict(X_arr)
        return pd.Series(np.asarray(preds).reshape(-1))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        X_te = self._to_array(X_test)
        y_te = self._to_target(y_test)
        y_pred = self.predict(X_te).to_numpy()

        clf = self._is_classification()
        metrics: Dict[str, float] = {}

        try:
            from sklearn import metrics as skm
        except Exception:
            # fallback: simple metrics without sklearn
            if clf:
                # threshold at 0.5
                pred_lbl = (y_pred >= 0.5).astype(int)
                acc = float((pred_lbl == y_te.astype(int)).mean())
                metrics["accuracy"] = acc
            else:
                rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
                mae = float(np.mean(np.abs(y_pred - y_te)))
                metrics.update({"rmse": rmse, "mae": mae})
            return metrics

        if clf:
            # Binary classification
            pred_lbl = (y_pred >= 0.5).astype(int)
            metrics["accuracy"] = float(skm.accuracy_score(y_te, pred_lbl))
            # logloss requires probabilities
            try:
                metrics["logloss"] = float(skm.log_loss(y_te, y_pred, labels=[0, 1]))
            except Exception:
                pass
            # AUC where possible
            try:
                metrics["auc"] = float(skm.roc_auc_score(y_te, y_pred))
            except Exception:
                pass
        else:
            metrics["rmse"] = float(np.sqrt(skm.mean_squared_error(y_te, y_pred)))
            metrics["mae"] = float(skm.mean_absolute_error(y_te, y_pred))

        return metrics


class NoBucketXGBModel(XGBModel):
    """Identical to XGBModel; bucketing handled upstream (formatter/pipeline)."""
    pass


class ClusteringBucketXGBModel(XGBModel):
    """Identical to XGBModel; train per cluster upstream if desired."""
    pass


class PrefixLenBucketXGBModel(XGBModel):
    """Identical to XGBModel; train per prefix-length bucket upstream."""
    pass
