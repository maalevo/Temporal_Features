# time_feature_benchmark/feature/encodings.py

from __future__ import annotations

from typing import Optional, Dict, Any, Set, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


# --------------------------------------------------------
# ------------ Abstract Base Class for Encoding ----------
# --------------------------------------------------------
class Encoding(ABC):
    """
    Base class for column encoders working on a single pandas Series.
    Contract:
      - validate(col: Series, params) -> None  (raise on problems)
      - fit(col: Series, params) -> None       (store any state)
      - encode(col: Optional[Series]) -> pd.DataFrame
          If `col` is provided, encode it using the fitted state without refitting.
          If omitted, encode the fitted `self.raw_col` (legacy behavior).
      - fit_encode(col: Series, params) -> pd.DataFrame  (convenience)
    """

    column_name: Optional[str] = None
    params: Dict[str, Any]
    raw_col: Optional[pd.Series]
    encoded_df: Optional[pd.DataFrame]

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        col: Optional[pd.Series] = None,
        column_name: Optional[str] = None,
    ):
        self.params = dict(params or {})
        self.raw_col = col.copy() if col is not None else None
        self.column_name = column_name
        self.encoded_df = None
        self._is_fitted: bool = False  # set True in fit()

    # ---------- Required interface ----------
    @abstractmethod
    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abstractmethod
    def encode(self, col: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Encode a Series using the fitted state.
        If `col` is None, encode the Series used at fit time (`self.raw_col`).
        Subclasses should NOT refit here; only transform/encode using stored state.
        """
        ...

    # ---------- Helper / common logic ----------
    def _ensure_series(self, col: pd.Series) -> None:
        if not isinstance(col, pd.Series):
            raise TypeError(f"Expected a pandas Series, got {type(col).__name__}")

    def _base_name(self, col_override: Optional[pd.Series] = None) -> str:
        """
        Pick a stable base column name for output.
        Priority:
          1) explicit self.column_name, if set
          2) the provided `col_override`'s name, if any
          3) the fitted self.raw_col's name, if any
          4) "value"
        """
        if self.column_name:
            return self.column_name
        if col_override is not None and col_override.name is not None:
            return str(col_override.name)
        if self.raw_col is not None and self.raw_col.name is not None:
            return str(self.raw_col.name)
        return "value"

    def fit_encode(self, col: pd.Series, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Uniform fit+encode across all encoders. Do NOT refit inside encode();
        it uses self.raw_col/self.params set in fit().
        """
        if col is None:
            raise ValueError("`col` must be provided for fit_encode().")

        self.raw_col = col.copy()
        if params is not None:
            self.params.update(params)

        # Validate & fit, then encode (no extra params to encode)
        self.validate(self.raw_col, self.params)
        self.fit(self.raw_col, self.params)
        return self.encode()

    # ---------- Fitted-state inspection ----------
    def is_fitted(self) -> bool:
        """Return True if fit() has been called at least once on this encoder."""
        return bool(self._is_fitted)


# ---------------------------------------------------
# ------------ Concrete Encoding Classes ------------
# ---------------------------------------------------

# ------------- Textual Encoding -------------
class TextualEncoding(Encoding):
    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)

        if col.isnull().to_numpy().any():
            raise ValueError("Input Series contains null values.")

        # No params allowed
        if params and len(params) > 0:
            raise ValueError(f"TextualEncoding does not accept parameters, got {params}.")

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for TextualEncoding.fit().")
        self.validate(col, params)

        self.raw_col = col.copy()
        self.params = dict(params or {})
        self._is_fitted = True

    def encode(self, col: Optional[pd.Series] = None) -> pd.DataFrame:
        series = col if col is not None else self.raw_col
        if series is None:
            raise ValueError("TextualEncoding.encode() requires a Series (argument or fitted `self.raw_col`).")

        base = self._base_name(series)
        df = series.astype(str).to_frame(name=base)

        self.encoded_df = df
        return self.encoded_df


# ------------- Numerical Encoding -------------
class NumericalEncoding(Encoding):
    """
    Numerical encoding:
      - Validates numeric Series with no nulls.
      - Returns the numeric column (optionally renamed).
      - No parameters are accepted.
    """

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)

        if col.isnull().to_numpy().any():
            raise ValueError("Input Series contains null values.")

        # Numeric check
        try:
            pd.to_numeric(col, errors="raise")
        except Exception as e:
            raise TypeError("Input Series must contain numeric values only.") from e

        # No params accepted
        if params and len(params) > 0:
            raise ValueError(f"NumericalEncoding does not accept parameters, got {params}.")

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for NumericalEncoding.fit().")
        self.validate(col, params)

        self.raw_col = col.copy()
        self.params = dict(params or {})
        self._is_fitted = True

    def encode(self, col: Optional[pd.Series] = None) -> pd.DataFrame:
        series = col if col is not None else self.raw_col
        if series is None:
            raise ValueError("NumericalEncoding.encode() requires a Series (argument or fitted `self.raw_col`).")

        s = pd.to_numeric(series, errors="raise")
        base_name = self._base_name(series)
        df = s.to_frame(name=f"{base_name}::numeric")
        self.encoded_df = df
        return self.encoded_df


# ------------- One-hot Encoding -------------
class OneHotEncoding(Encoding):
    """
    Params:
      - cutoff: float  (fraction in [0,1], or percentage > 1)
        Categories with frequency strictly less than cutoff are grouped into "Others".

    Behavior:
      - Deterministic column order stored at fit time (lexicographic; Others last).
      - Stable feature space across datasets via `transform()`.
      - If a cutoff is provided, an "Others" column is always present (all zeros
        when no rare categories were seen during fit).
    """

    OTHERS = "__OTHER__"  # non-colliding token

    _cutoff: Optional[float] = None
    _rare_set: Optional[Set[str]] = None
    _fitted_cols_: Optional[List[str]] = None
    _kept_no_others_: Optional[Set[str]] = None

    def _normalized_cutoff(self, params: Optional[Dict[str, Any]]) -> Optional[float]:
        if not params or params.get("cutoff") is None:
            return None
        cutoff = float(params["cutoff"])
        if cutoff > 1.0:
            cutoff /= 100.0
        return cutoff

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)

        if col.isnull().to_numpy().any():
            raise ValueError("Input Series contains null values.")

        allowed = {"cutoff"}
        if params is None:
            return
        invalid = set(params.keys()) - allowed
        if invalid:
            raise ValueError(f"Invalid parameter(s): {sorted(invalid)}. Allowed: {sorted(allowed)}")

        cutoff_raw = params.get("cutoff", None)
        if cutoff_raw is None:
            return

        # Reject bools explicitly (since bool is a subclass of int)
        if isinstance(cutoff_raw, bool) or not isinstance(cutoff_raw, (int, float)):
            raise ValueError('"cutoff" must be a number (fraction or percentage).')

        cutoff = self._normalized_cutoff(params)

        # Reject NaN and out-of-range
        if cutoff is None or pd.isna(cutoff) or cutoff < 0.0 or cutoff > 1.0:
            raise ValueError('"cutoff" must normalize to a value in [0, 1].')

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for OneHotEncoding.fit().")
        self.validate(col, params)

        self.raw_col = col.copy()
        self.params = dict(params or {})

        self._cutoff = self._normalized_cutoff(self.params)
        s = self.raw_col.astype(str)
        n = len(s)

        if n == 0:
            # Empty fit: no categories, but if cutoff specified keep Others for stability
            self._rare_set = set()
            base = self._base_name()
            kept: List[str] = []
            if self._cutoff is not None:
                kept.append(self.OTHERS)
            self._kept_no_others_ = set()
            self._fitted_cols_ = [f"{base}::onehot::{c}" for c in kept]
            self.encoded_df = pd.DataFrame(columns=self._fitted_cols_, dtype=int)
            self._is_fitted = True
            return

        counts = s.value_counts(dropna=False)
        if self._cutoff is None:
            self._rare_set = set()
        else:
            threshold = self._cutoff * n
            self._rare_set = set(counts[counts < threshold].index.astype(str))

        # Kept frequent categories (unprefixed), sorted lexicographically
        categories = set(counts.index.astype(str))
        kept_no_others = sorted(categories - self._rare_set)
        self._kept_no_others_ = set(kept_no_others)

        # Build fitted column list (Others last if cutoff provided)
        kept = list(kept_no_others)
        if self._cutoff is not None:
            kept.append(self.OTHERS)

        base = self._base_name()
        self._fitted_cols_ = [f"{base}::onehot::{c}" for c in kept]

        # Produce encoded_df for the fitted column as reference
        self.encoded_df = self.encode()
        self._is_fitted = True

    def transform(self, col: pd.Series, handle_unknown: str = "other") -> pd.DataFrame:
        """
        Transform a new Series using the fitted categories.
        handle_unknown:
          - "other": map unseen to Others (requires cutoff / Others column).
          - "ignore": unseen become all-zero rows after reindexing.
          - "error": raise if unseen categories are present.
        """
        self._ensure_series(col)
        if col.isnull().to_numpy().any():
            raise ValueError("Input Series contains null values.")
        if self._fitted_cols_ is None or self._kept_no_others_ is None:
            raise ValueError("Call fit() before transform().")

        s = col.astype(str)
        kept_no_others = self._kept_no_others_
        has_others = (self._cutoff is not None)

        # Unknown handling
        unknown = set(s.unique()) - kept_no_others
        if handle_unknown == "error" and unknown:
            raise ValueError(f"Unseen categories present: {sorted(unknown)}")
        elif handle_unknown == "other" and has_others:
            s = s.where(s.isin(kept_no_others), other=self.OTHERS)
        elif handle_unknown == "ignore":
            # Leave unknowns as NaN so get_dummies ignores them → all-zero rows after reindex
            s = s.where(s.isin(kept_no_others), other=np.nan)
        else:
            # If user asked for "other" but we don't have Others, fall back to "ignore"
            if handle_unknown == "other" and not has_others:
                s = s.where(s.isin(kept_no_others), other=np.nan)

        dummies = pd.get_dummies(s, dtype=int)

        # Prefix and reindex to fitted columns for stability
        base = self._base_name(col)
        dummies.columns = [f"{base}::onehot::{c}" for c in dummies.columns]
        dummies = dummies.reindex(columns=self._fitted_cols_, fill_value=0)

        return dummies

    def encode(self, col: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Encode using fitted categories.
        - If `col` is provided: encode it using fitted state (no refit).
        - If omitted: encode the fitted `self.raw_col` (train-time realization).
        """
        series = col if col is not None else self.raw_col
        if series is None:
            raise ValueError("OneHotEncoding.encode() requires a Series (argument or fitted `self.raw_col`).")
        if self._fitted_cols_ is None or self._kept_no_others_ is None:
            raise ValueError("Call fit() before encode().")

        df = self.transform(series, handle_unknown="other")
        self.encoded_df = df
        return df

    # Optional: light-weight accessors
    @property
    def categories_(self) -> List[str]:
        """Unprefixed kept categories (excluding Others)."""
        return sorted(self._kept_no_others_ or [])

    @property
    def fitted_columns_(self) -> List[str]:
        """Fully-prefixed fitted column names, in deterministic order."""
        return list(self._fitted_cols_ or [])


# ------------- Sinusoidal Encoding -------------
class SinCosEncoding(Encoding):
    """
    Sinusoidal encoding for cyclical features.

    Params:
      - period: positive number (required)

    Output:
      - Two columns: "<base>::sin::<period>", "<base>::cos::<period>"
      - No additional normalization is applied; range is native [-1, 1] (with minor
        numeric clipping to that interval).
    """

    _period: Optional[float] = None

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)

        if col.isnull().to_numpy().any():
            raise ValueError("Input Series contains null values.")

        try:
            pd.to_numeric(col, errors="raise")
        except Exception as e:
            raise TypeError("Input Series must contain numeric values only.") from e

        allowed = {"period"}
        if params is None or "period" not in params:
            raise ValueError('Parameter "period" is required for SinCosEncoding.')
        invalid = set(params.keys()) - allowed
        if invalid:
            raise ValueError(f"Invalid parameter(s): {sorted(invalid)}. Allowed: {sorted(allowed)}")

        period = params.get("period")
        if not isinstance(period, (int, float)):
            raise TypeError('"period" must be a number.')
        if period <= 0:
            raise ValueError('"period" must be a positive number.')

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for SinCosEncoding.fit().")
        self.validate(col, params)

        self.raw_col = col.copy()
        self.params = dict(params or {})
        self._period = float(self.params["period"])
        self._is_fitted = True

    def encode(self, col: Optional[pd.Series] = None) -> pd.DataFrame:
        series = col if col is not None else self.raw_col
        if series is None:
            raise ValueError("SinCosEncoding.encode() requires a Series (argument or fitted `self.raw_col`).")
        if self._period is None:
            raise ValueError("Period not set. Did you call fit()?")

        base_name = self._base_name(series)
        s = pd.to_numeric(series, errors="raise")

        # Sinusoidal transform (native range [-1, 1])
        sin_vals = pd.Series(np.sin(2 * np.pi * s / self._period), index=s.index)
        cos_vals = pd.Series(np.cos(2 * np.pi * s / self._period), index=s.index)

        # Clip numerical noise to [-1, 1]
        sin_vals = sin_vals.clip(-1.0, 1.0)
        cos_vals = cos_vals.clip(-1.0, 1.0)

        # Append the period to the column names
        if float(self._period).is_integer():
            period_tag = str(int(self._period))
        else:
            period_tag = f"{self._period:g}"  # compact formatting

        df = pd.DataFrame(
            {
                f"{base_name}::sincos::sin::{period_tag}": sin_vals,
                f"{base_name}::sincos::cos::{period_tag}": cos_vals,
            },
            index=series.index,
        )

        self.encoded_df = df
        return self.encoded_df
