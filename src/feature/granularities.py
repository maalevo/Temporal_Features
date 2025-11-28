# time_feature_benchmark/feature/granularities.py
from __future__ import annotations

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# --------------------------------------------------------
# -------------------- Base Granularity ------------------
# --------------------------------------------------------
class Granularity(ABC):
    """
    Base class for time granularities.
    API (mirrors your encoders): validate -> fit -> transform, and fit_transform convenience.
    Input: pd.Series of timestamps (datetime-like) OR numeric seconds.
    Output: pd.Series of epoch seconds snapped to the granularity start.
    """
    params: Dict[str, Any]
    raw: Optional[pd.Series]
    transformed_: Optional[pd.Series]

    def __init__(self, params: Optional[Dict[str, Any]] = None, col: Optional[pd.Series] = None):
        self.params = dict(params or {})
        self.raw = col.copy() if col is not None else None
        self.transformed_ = None

    # -------- Required interface --------
    @abstractmethod
    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None: ...
    @abstractmethod
    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None: ...
    @abstractmethod
    def transform(self) -> pd.Series: ...

    # -------- Common helpers --------
    def _ensure_series(self, col: pd.Series) -> None:
        if not isinstance(col, pd.Series):
            raise TypeError(f"Expected a pandas Series, got {type(col).__name__}")
        if col.isnull().to_numpy().any():
            raise ValueError("Input Series contains nulls.")

    def _to_epoch_seconds(self, col: pd.Series) -> pd.Series:
        """
        Normalize input to float seconds since Unix epoch (UTC).
        - If numeric → treated as seconds already.
        - Else → parsed as datetimes (UTC), then converted to seconds.
        """
        if pd.api.types.is_numeric_dtype(col):
            return pd.to_numeric(col, errors="raise").astype(float)

        ts = pd.to_datetime(col, utc=True, errors="coerce")
        if ts.isna().any():
            raise TypeError("Series contains values that cannot be parsed as timestamps.")
        # Seconds since epoch (handles tz-aware cleanly)
        return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype(float)
    
    def _out_name(self, granularity: str) -> str:
        base = self.raw.name if (self.raw is not None and self.raw.name is not None) else "time"
        return f"{base}::{granularity}"

    def fit_transform(self, col: pd.Series, params: Optional[Dict[str, Any]] = None) -> pd.Series:
        if col is None:
            raise ValueError("`col` must be provided for fit_transform().")
        self.raw = col.copy()
        if params is not None:
            self.params.update(params)
        self.validate(self.raw, self.params)
        self.fit(self.raw, self.params)
        return self.transform()


# --------------------------------------------------------
# ---------------- Second Granularity --------------------
# --------------------------------------------------------
class SecondGranularity(Granularity):
    """Snap to start-of-second; returns integer epoch seconds."""

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)
        # no params for now
        if params and len(params) > 0:
            raise ValueError(f"SecondGranularity takes no parameters, got {params}.")

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for SecondGranularity.fit().")
        # Check parseability
        _ = self._to_epoch_seconds(col)  # raises on error
        self.raw = col.copy()
        self.params = dict(params or {})

    def transform(self) -> pd.Series:
        if self.raw is None:
            raise ValueError("SecondGranularity.transform() requires a fitted Series.")
        s = self._to_epoch_seconds(self.raw)
        out = np.floor(s).astype("int64")
        return pd.Series(out, index=s.index, name=self._out_name("second"))

# --------------------------------------------------------
# ---------------- Minute Granularity --------------------
# --------------------------------------------------------
class MinuteGranularity(Granularity):
    """Snap to start-of-minute; returns integer epoch seconds."""

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)
        if params and len(params) > 0:
            raise ValueError(f"MinuteGranularity takes no parameters, got {params}.")
        # ensure parseable
        _ = self._to_epoch_seconds(col)

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for MinuteGranularity.fit().")
        self.validate(col, params)
        self.raw = col.copy()
        self.params = dict(params or {})

    def transform(self) -> pd.Series:
        if self.raw is None:
            raise ValueError("MinuteGranularity.transform() requires a fitted Series.")
        s = self._to_epoch_seconds(self.raw)
        out = (np.floor(s / 60.0) * 60.0).astype("int64")
        return pd.Series(out, index=s.index, name=self._out_name("minute"))

# --------------------------------------------------------
# ----------------- Hour Granularity ---------------------
# --------------------------------------------------------
class HourGranularity(Granularity):
    """Snap to start-of-hour; returns integer epoch seconds."""

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)
        if params and len(params) > 0:
            raise ValueError(f"HourGranularity takes no parameters, got {params}.")

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for HourGranularity.fit().")
        _ = self._to_epoch_seconds(col)
        self.raw = col.copy()
        self.params = dict(params or {})

    def transform(self) -> pd.Series:
        if self.raw is None:
            raise ValueError("HourGranularity.transform() requires a fitted Series.")
        s = self._to_epoch_seconds(self.raw)
        out = (np.floor(s / 3600.0) * 3600.0).astype("int64")
        return pd.Series(out, index=s.index, name=self._out_name("hour"))

# --------------------------------------------------------
# ----------------- Day Granularity ----------------------
# --------------------------------------------------------
class DayGranularity(Granularity):
    """Snap to start-of-day (00:00:00 UTC); returns integer epoch seconds."""

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)
        # no params for this granularity
        if params and len(params) > 0:
            raise ValueError(f"DayGranularity takes no parameters, got {params}.")

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for DayGranularity.fit().")
        # ensure input is parseable to epoch seconds
        _ = pd.to_numeric(col, errors="coerce") if pd.api.types.is_numeric_dtype(col) else pd.to_datetime(col, utc=True, errors="coerce")
        if getattr(_, "isna", lambda: False)().any():
            raise TypeError("Series contains values that cannot be parsed as timestamps or numeric seconds.")
        self.raw = col.copy()
        self.params = dict(params or {})

    def transform(self) -> pd.Series:
        if self.raw is None:
            raise ValueError("DayGranularity.transform() requires a fitted Series.")
        ts = pd.to_datetime(self.raw, utc=True, errors="coerce") if not pd.api.types.is_numeric_dtype(self.raw) \
            else pd.to_datetime(self.raw, unit="s", utc=True, errors="coerce")
        if ts.isna().any():
            raise TypeError("Series contains values that cannot be parsed as timestamps.")
        secs = (ts.dt.floor("D") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
        return secs.rename(self._out_name("day"))


# --------------------------------------------------------
# ----------------- Week Granularity ---------------------
# --------------------------------------------------------
class WeekGranularity(Granularity):
    """
    Snap to start-of-week; returns integer epoch seconds.
    Params:
      - week_start: one of {"MON","TUE","WED","THU","FRI","SAT","SUN"} (default "MON")
        Uses pandas' weekly frequency anchors ("W-<DAY>").
    """
    _week_start: str = "MON"

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)
        allowed = {"week_start"}
        if params:
            invalid = set(params.keys()) - allowed
            if invalid:
                raise ValueError(f"Invalid parameter(s): {sorted(invalid)}. Allowed: {sorted(allowed)}")
            if "week_start" in params:
                wk = str(params["week_start"]).upper()
                if wk not in {"MON","TUE","WED","THU","FRI","SAT","SUN"}:
                    raise ValueError('week_start must be one of {"MON","TUE","WED","THU","FRI","SAT","SUN"}.')
        # Ensure parseability
        _ = self._to_epoch_seconds(col)

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for WeekGranularity.fit().")
        self.raw = col.copy()
        self.params = dict(params or {})
        self._week_start = str(self.params.get("week_start", "MON")).upper()

    def transform(self) -> pd.Series:
        if self.raw is None:
            raise ValueError("WeekGranularity.transform() requires a fitted Series.")

        # Convert input to UTC timestamps
        if pd.api.types.is_numeric_dtype(self.raw):
            ts = pd.to_datetime(self.raw, unit="s", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(self.raw, utc=True, errors="coerce")
        if ts.isna().any():
            raise TypeError("Series contains values that cannot be parsed as timestamps.")

        # Map week start string to weekday index (MON=0, ..., SUN=6)
        WEEKIDX = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
        start_idx = WEEKIDX[self._week_start]

        # Days to subtract to reach the week's anchor day for each timestamp
        delta_days = (ts.dt.weekday - start_idx) % 7

        # Start-of-week at 00:00:00 UTC
        week_start = (ts - pd.to_timedelta(delta_days, unit="D")).dt.floor("D")

        # Return epoch seconds, named "<col.name>::week"
        secs = (week_start - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
        return secs.rename(self._out_name("week"))

# --------------------------------------------------------
# ----------------- Month Granularity --------------------
# --------------------------------------------------------
class MonthGranularity(Granularity):
    """Snap to start-of-month (00:00:00 UTC on the 1st); returns integer epoch seconds."""

    def validate(self, col: pd.Series, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_series(col)
        # no params for this granularity
        if params and len(params) > 0:
            raise ValueError(f"MonthGranularity takes no parameters, got {params}.")
        # ensure parseable
        _ = self._to_epoch_seconds(col)

    def fit(self, col: Optional[pd.Series], params: Optional[Dict[str, Any]] = None) -> None:
        if col is None:
            raise ValueError("`col` must be provided for MonthGranularity.fit().")
        self.validate(col, params)
        self.raw = col.copy()
        self.params = dict(params or {})

    def transform(self) -> pd.Series:
        if self.raw is None:
            raise ValueError("MonthGranularity.transform() requires a fitted Series.")

        # Convert to UTC timestamps (handles numeric seconds or datetime-like inputs)
        ts = (
            pd.to_datetime(self.raw, unit="s", utc=True, errors="coerce")
            if pd.api.types.is_numeric_dtype(self.raw)
            else pd.to_datetime(self.raw, utc=True, errors="coerce")
        )
        if ts.isna().any():
            raise TypeError("Series contains values that cannot be parsed as timestamps.")

        # Start of the month at 00:00:00 UTC
        day_start = ts.dt.floor("D")
        month_start = day_start - pd.to_timedelta(ts.dt.day - 1, unit="D")

        secs = (month_start - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
        return secs.rename(self._out_name("month"))
