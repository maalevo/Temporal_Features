# model/transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class Transformed:
    X: Any                    # np.ndarray or framework tensor
    y: np.ndarray
    feature_names: Optional[List[str]] = None
    groups: Optional[np.ndarray] = None      # for bucketing
    seq_lengths: Optional[np.ndarray] = None # for LSTM
    meta: Optional[Dict[str, Any]] = None


class Transformer(ABC):
    """Fit on TRAIN only; use transform for VAL/TEST; returns Prepared."""
    def fit(self, df: pd.DataFrame, y: pd.Series, **ctx) -> None: ...
    @abstractmethod
    def transform(self, df: pd.DataFrame, y: Optional[pd.Series]=None, **ctx) -> Transformed: ...
    def fit_transform(self, df: pd.DataFrame, y: pd.Series, **ctx) -> Transformed:
        self.fit(df, y, **ctx)
        return self.transform(df, y, **ctx)


class LSTMTransformer(Transformer):
    """
    Builds padded sequences (N, T, F) for Keras LSTMs.

    Data assumptions:
      - Input df contains at least [case_col, time_col] plus feature columns.
      - Target y can be row-level; we reduce to the *last* value per case by time.
      - Use a Keras Masking layer with `mask_value=pad_value` to ignore padded timesteps.

    Normalization:
      - X_normalization: optional feature scaling learned on TRAIN. One of:
            None, 'zscore', '0/1', '-1/1'
      - y_normalization: optional target scaling learned on TRAIN (after last-per-case
        reduction). One of:
            None, 'zscore', '0/1', '-1/1'
    """

    def __init__(
        self,
        maxlen: Optional[int] = None,     # if None, inferred from TRAIN sequences
        pad_value: float = 0.0,
        X_normalization: Optional[str] = None,   # 'zscore' | '0/1' | '-1/1' | None
        y_normalization: Optional[str] = None,   # 'zscore' | '0/1' | '-1/1' | None
        dtype: str = "float32",
    ):
        self.maxlen = maxlen
        self.pad_value = float(pad_value)
        self.X_normalization = (X_normalization or None)
        self.y_normalization = (y_normalization or None)
        self.dtype = dtype

        # learned during fit()
        self._fitted: bool = False
        self._feat_cols: List[str] = []
        self._T: Optional[int] = None

        # X stats
        self._x_mode: Optional[str] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._x_min: Optional[np.ndarray] = None
        self._x_max: Optional[np.ndarray] = None

        # y stats
        self._y_mode: Optional[str] = None
        self._y_mean: Optional[float] = None
        self._y_std: Optional[float] = None
        self._y_min: Optional[float] = None
        self._y_max: Optional[float] = None

    # -----------------------------
    # Transformer interface
    # -----------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series, *, case_col: str, time_col: str) -> None:
        # validate modes
        def _norm_ok(mode: Optional[str]) -> bool:
            return (mode is None) or (str(mode).lower() in {"zscore", "0/1", "-1/1"})
        if not _norm_ok(self.X_normalization):
            raise ValueError("X_normalization must be one of: None, 'zscore', '0/1', '-1/1'")
        if not _norm_ok(self.y_normalization):
            raise ValueError("y_normalization must be one of: None, 'zscore', '0/1', '-1/1'")

        # feature columns = everything except case/time
        self._feat_cols = [c for c in df.columns if c not in {case_col, time_col}]
        if not self._feat_cols:
            raise ValueError("No feature columns found. Provide [case_col, time_col] + feature columns.")

        # ensure datetime for stable sorting
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        if df[time_col].isna().any():
            raise ValueError(f"{time_col} contains non-coercible timestamps.")

        # sequence lengths on TRAIN
        lengths = (
            df.sort_values([case_col, time_col])
              .groupby(case_col, sort=False).size().to_numpy()
        )
        inferred_T = int(lengths.max()) if lengths.size else 0
        self._T = int(self.maxlen) if self.maxlen is not None else inferred_T

        # -------- X normalization stats (per feature) on TRAIN --------
        self._x_mode = None if self.X_normalization is None else str(self.X_normalization).lower()
        feat_mat = df[self._feat_cols].to_numpy(dtype=float)

        if self._x_mode == "zscore":
            mean = feat_mat.mean(axis=0)
            std = feat_mat.std(axis=0, ddof=0)
            std[std == 0.0] = 1.0
            self._x_mean, self._x_std = mean, std
        elif self._x_mode in {"0/1", "-1/1"}:
            xmin = feat_mat.min(axis=0)
            xmax = feat_mat.max(axis=0)
            self._x_min, self._x_max = xmin, xmax
        else:
            # no normalization
            self._x_mean = self._x_std = self._x_min = self._x_max = None

        # -------- y normalization stats (learned on last value per case) --------
        self._y_mode = None if self.y_normalization is None else str(self.y_normalization).lower()
        if self._y_mode is not None:
            if y is None:
                raise ValueError("y must be provided to fit() when y_normalization is enabled.")
            y_tmp = pd.DataFrame({
                case_col: df[case_col].values,
                time_col: df[time_col].values,
                "_y": y.values,
            })
            y_last = (
                y_tmp.sort_values([case_col, time_col])
                     .groupby(case_col, sort=False)["_y"].last()
                     .astype(float)
                     .to_numpy()
            )
            if self._y_mode == "zscore":
                m = float(np.mean(y_last)) if y_last.size else 0.0
                s = float(np.std(y_last, ddof=0)) if y_last.size else 1.0
                if s == 0.0:
                    s = 1.0
                self._y_mean, self._y_std = m, s
            elif self._y_mode in {"0/1", "-1/1"}:
                ymin = float(np.min(y_last)) if y_last.size else 0.0
                ymax = float(np.max(y_last)) if y_last.size else 1.0
                self._y_min, self._y_max = ymin, ymax

        self._fitted = True

    def transform(
        self,
        df: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        case_col: str,
        time_col: str
    ) -> Transformed:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        # basic checks
        for c in (case_col, time_col):
            if c not in df.columns:
                raise KeyError(f"Missing required column '{c}' in DataFrame.")

        # ensure datetime for sorting
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        if df[time_col].isna().any():
            raise ValueError(f"{time_col} contains non-coercible timestamps.")

        # align feature set to training order; unseen features -> 0.0
        work = df.copy()
        for c in self._feat_cols:
            if c not in work.columns:
                work[c] = 0.0

        # build feature matrix
        feat_mat = work[self._feat_cols].to_numpy(dtype=float)

        # apply X normalization learned in fit()
        if self._x_mode == "zscore" and self._x_mean is not None and self._x_std is not None:
            feat_mat = (feat_mat - self._x_mean) / self._x_std
        elif self._x_mode in {"0/1", "-1/1"} and self._x_min is not None and self._x_max is not None:
            denom = (self._x_max - self._x_min)
            denom[denom == 0.0] = 1.0  # avoid division by zero per feature
            x01 = (feat_mat - self._x_min) / denom
            feat_mat = x01 if self._x_mode == "0/1" else (x01 * 2.0 - 1.0)

        # rebuild with just case/time/normalized features for grouping
        work = pd.concat(
            [work[[case_col, time_col]].reset_index(drop=True),
             pd.DataFrame(feat_mat, columns=self._feat_cols)],
            axis=1,
        )

        # cases → sequences
        g = work.sort_values([case_col, time_col]).groupby(case_col, sort=False)
        case_order = list(g.groups.keys())
        cases = [g.get_group(cid)[self._feat_cols].to_numpy(dtype=self.dtype) for cid in case_order]

        lengths = np.array([len(c) for c in cases], dtype=np.int32)
        N = len(cases)
        T = int(self._T or 0)
        F = len(self._feat_cols)

        X = np.full((N, T, F), self.pad_value, dtype=self.dtype)
        for i, c in enumerate(cases):
            t = min(len(c), T)
            if t > 0:
                X[i, :t, :] = c[:t]

        # targets: reduce to last per case; apply y normalization if configured
        if y is not None:
            y_tmp = pd.DataFrame({
                case_col: df[case_col].values,
                time_col: df[time_col].values,
                "_y": y.values,
            })
            y_last = (
                y_tmp.sort_values([case_col, time_col])
                     .groupby(case_col, sort=False)["_y"].last()
            )
            y_vec = y_last.reindex(case_order).to_numpy().astype(self.dtype)

            if self._y_mode == "zscore":
                m = self._y_mean if self._y_mean is not None else 0.0
                s = self._y_std if self._y_std is not None else 1.0
                if s == 0.0:
                    s = 1.0
                y_vec = (y_vec - m) / s
            elif self._y_mode in {"0/1", "-1/1"}:
                ymin = self._y_min if self._y_min is not None else float(np.min(y_vec)) if y_vec.size else 0.0
                ymax = self._y_max if self._y_max is not None else float(np.max(y_vec)) if y_vec.size else 1.0
                denom = ymax - ymin
                if denom == 0.0:
                    y_vec = np.zeros_like(y_vec, dtype=self.dtype)
                else:
                    y01 = (y_vec - ymin) / denom
                    y_vec = y01 if self._y_mode == "0/1" else (y01 * 2.0 - 1.0)
        else:
            y_vec = np.zeros((N,), dtype=self.dtype)

        return Transformed(
            X=X,
            y=y_vec,
            feature_names=self._feat_cols.copy(),
            groups=None,
            seq_lengths=lengths,
            meta={
                "case_order": case_order,
                "pad_value": self.pad_value,
                "X_normalization": self._x_mode,
                "X_stats": {
                    "mean": None if self._x_mean is None else self._x_mean.copy(),
                    "std":  None if self._x_std  is None else self._x_std.copy(),
                    "min":  None if self._x_min  is None else self._x_min.copy(),
                    "max":  None if self._x_max  is None else self._x_max.copy(),
                },
                "y_normalization": self._y_mode,
                "y_stats": {
                    "mean": self._y_mean,
                    "std":  self._y_std,
                    "min":  self._y_min,
                    "max":  self._y_max,
                },
            },
        )

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        case_col: str,
        time_col: str
    ) -> Transformed:
        super().fit_transform  # keeps linting happy; we override to preserve type hints
        self.fit(df, y, case_col=case_col, time_col=time_col)
        return self.transform(df, y, case_col=case_col, time_col=time_col)

    # -------- Optional helper: invert y normalization on predictions --------
    def is_y_normalized(self) -> bool:
        """Return True if y was normalized (i.e., a mode was selected and fit() ran)."""
        return self._y_mode is not None

    def make_y_denormalizer(self):
        """
        Return a pure function f(y_norm) -> y_original using TRAIN stats captured in fit().
        Helpful to save in Prepared.meta and call later on model predictions.
        """
        mode = self._y_mode
        m, s = self._y_mean, self._y_std
        ymin, ymax = self._y_min, self._y_max

        def denorm(y):
            arr = np.asarray(y, dtype=float)
            if mode is None:
                return arr
            if mode == "zscore":
                mu = 0.0 if m is None else m
                sd = 1.0 if s is None or s == 0.0 else s
                return arr * sd + mu
            if mode == "0/1":
                lo = 0.0 if ymin is None else ymin
                hi = 1.0 if ymax is None else ymax
                return arr * (hi - lo) + lo
            if mode == "-1/1":
                lo = 0.0 if ymin is None else ymin
                hi = 1.0 if ymax is None else ymax
                return ((arr + 1.0) / 2.0) * (hi - lo) + lo
            return arr

        return denorm

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        y = np.asarray(y_norm, dtype=float)
        mode = self._y_mode
        if mode is None:
            return y  # nothing to undo

        if mode == "zscore":
            if self._y_mean is None or self._y_std is None:
                raise RuntimeError("y denorm failed: zscore stats not set. Did you call fit() with y_normalization?")
            return y * float(self._y_std) + float(self._y_mean)

        if mode == "0/1":
            if self._y_min is None or self._y_max is None:
                raise RuntimeError("y denorm failed: min/max not set. Did you call fit() with y_normalization?")
            return y * (float(self._y_max) - float(self._y_min)) + float(self._y_min)

        if mode == "-1/1":
            if self._y_min is None or self._y_max is None:
                raise RuntimeError("y denorm failed: min/max not set. Did you call fit() with y_normalization?")
            return ((y + 1.0) / 2.0) * (float(self._y_max) - float(self._y_min)) + float(self._y_min)

        return y


