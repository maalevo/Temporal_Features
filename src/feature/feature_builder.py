# feature_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from .feature_registry import FeatureRegistry

_SECONDS_PER_DAY  = 86_400
_SECONDS_PER_WEEK = 7 * _SECONDS_PER_DAY
_SECONDS_PER_31D  = 31 * _SECONDS_PER_DAY


# ---------------------- Unified spec (registry-key based) ----------------------

@dataclass(frozen=True)
class FeatureBuildSpec:
    feature_key: str
    source_col_name: str
    encoding_key: str
    encoding_params: Optional[Dict[str, Any]] = None
    target_col_name: Optional[str] = None
    granularity_key: Optional[str] = None
    granularity_params: Optional[Dict[str, Any]] = None
    case_source: Optional[str] = None


# ---------------------- Helpers ----------------------

def _to_utc_ts(col: pd.Series) -> pd.Series:
    ts = (pd.to_datetime(col, utc=True, errors="coerce")
          if not pd.api.types.is_numeric_dtype(col)
          else pd.to_datetime(col, unit="s", utc=True, errors="coerce"))
    if ts.isna().any():
        raise TypeError("Series contains values that cannot be parsed as timestamps.")
    return ts


# ---------------------- Generic Feature ----------------------

class Feature:
    def __init__(self, col: pd.Series, spec: FeatureBuildSpec, registry: FeatureRegistry, encoding: Optional[Any] = None):
        if not isinstance(col, pd.Series):
            raise TypeError(f"Expected a pandas Series, got {type(col).__name__}")
        if not spec.encoding_key and encoding is None:
            raise ValueError("Either an encoding or FeatureBuildSpec.encoding_key is required.")

        self.col = col
        self.spec = spec
        self.registry = registry

        if encoding is not None:
            self.encoding = encoding
            if spec.target_col_name:
                setattr(self.encoding, "column_name", spec.target_col_name)
            if spec.encoding_params:
                setattr(self.encoding, "params", dict(spec.encoding_params))
        else:
            enc_cls = self.registry.get_encoding(spec.encoding_key)
            self.encoding = enc_cls(
                params=dict(spec.encoding_params or {}),
                col=None,
                column_name=spec.target_col_name,
            )

        self.encoded_df: Optional[pd.DataFrame] = None

    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        if callable(getattr(self.encoding, "is_fitted", None)) and self.encoding.is_fitted():
            self.encoded_df = self.encoding.encode(self.col)
        else:
            self.encoded_df = self.encoding.fit_encode(self.col, self.encoding.params)
        return self.encoded_df, self.encoding
    

class ActivityFeature(Feature):
    pass


# ---------------------- General Time Feature ----------------------

class TimeFeature(Feature):
    def __init__(self, col: pd.Series, spec: FeatureBuildSpec, registry: FeatureRegistry, encoding: Optional[Any] = None):
        if not isinstance(col, pd.Series):
            raise TypeError(f"Expected a pandas Series, got {type(col).__name__}")
        if not spec.granularity_key:
            raise ValueError("FeatureBuildSpec.granularity_key is required for TimeFeature.")

        self.col = col
        self.spec = spec
        self.registry = registry

        gran_cls = self.registry.get_granularity(spec.granularity_key)
        self.granularity = gran_cls(params=dict(spec.granularity_params or {}), col=None)

        if encoding is not None:
            self.encoding = encoding
            if spec.target_col_name:
                setattr(self.encoding, "column_name", spec.target_col_name)
            if spec.encoding_params:
                setattr(self.encoding, "params", dict(spec.encoding_params))
        else:
            enc_cls = self.registry.get_encoding(spec.encoding_key)
            self.encoding = enc_cls(
                params=dict(spec.encoding_params or {}),
                col=None,
                column_name=spec.target_col_name,
            )

        self.granulated: Optional[pd.Series] = None
        self.encoded_df: Optional[pd.DataFrame] = None

    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        self.granulated = self.granularity.fit_transform(self.col, self.granularity.params)
        if callable(getattr(self.encoding, "is_fitted", None)) and self.encoding.is_fitted():
            self.encoded_df = self.encoding.encode(self.granulated)
        else:
            self.encoded_df = self.encoding.fit_encode(self.granulated, self.encoding.params)
        return self.encoded_df, self.encoding


# ---------------------- Specific Time Features ----------------------

class UTCTimeFeature(TimeFeature):
    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        ts = _to_utc_ts(self.col)
        secs = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype(float)
        secs.name = self.spec.target_col_name or ((self.col.name or "time") + "::utc")

        granulated = self.granularity.fit_transform(secs, self.granularity.params)
        if callable(getattr(self.encoding, "is_fitted", None)) and self.encoding.is_fitted():
            encoded = self.encoding.encode(granulated)
        else:
            encoded = self.encoding.fit_encode(granulated, self.encoding.params)
        return encoded, self.encoding


class TimeInDayFeature(TimeFeature):
    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        ts = _to_utc_ts(self.col)
        seconds_in_day = (ts - ts.dt.floor("D")).dt.total_seconds().astype(float)
        seconds_in_day.name = self.spec.target_col_name or ((self.col.name or "time") + "::time_in_day")

        granulated = self.granularity.fit_transform(seconds_in_day, self.granularity.params)
        if self.encoding.is_fitted():
            encoded = self.encoding.encode(granulated)
        else:
            encoded = self.encoding.fit_encode(granulated, self.encoding.params)
        return encoded, self.encoding


class TimeInWeekFeature(TimeFeature):
    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        ts = _to_utc_ts(self.col)
        week_start = (self.granularity.params or {}).get("week_start", "MON")
        week_idx = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
        start_idx = week_idx.get(str(week_start).upper(), 0)

        delta_days = (ts.dt.weekday - start_idx) % 7
        week_anchor = (ts - pd.to_timedelta(delta_days, unit="D")).dt.floor("D")
        seconds_in_week = (ts - week_anchor).dt.total_seconds().astype(float)
        seconds_in_week.name = self.spec.target_col_name or ((self.col.name or "time") + "::time_in_week")

        granulated = self.granularity.fit_transform(seconds_in_week, self.granularity.params)
        if self.encoding.is_fitted():
            encoded = self.encoding.encode(granulated)
        else:
            encoded = self.encoding.fit_encode(granulated, self.encoding.params)
        return encoded, self.encoding


class TimeInMonthFeature(TimeFeature):
    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        ts = _to_utc_ts(self.col)
        day_start = ts.dt.floor("D")
        month_start = day_start - pd.to_timedelta(ts.dt.day - 1, unit="D")
        seconds_in_month = (ts - month_start).dt.total_seconds().astype(float)
        seconds_in_month.name = self.spec.target_col_name or ((self.col.name or "time") + "::time_in_month")

        granulated = self.granularity.fit_transform(seconds_in_month, self.granularity.params)
        if self.encoding.is_fitted():
            encoded = self.encoding.encode(granulated)
        else:
            encoded = self.encoding.fit_encode(granulated, self.encoding.params)
        return encoded, self.encoding


class TimeInYearFeature(TimeFeature):
    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        ts = _to_utc_ts(self.col)
        day_start = ts.dt.floor("D")
        year_start = day_start - pd.to_timedelta(ts.dt.dayofyear - 1, unit="D")

        seconds_in_year = (ts - year_start).dt.total_seconds().astype(float)
        seconds_in_year.name = self.spec.target_col_name or ((self.col.name or "time") + "::time_in_year")

        granulated = self.granularity.fit_transform(seconds_in_year, self.granularity.params)
        if self.encoding.is_fitted():
            encoded = self.encoding.encode(granulated)
        else:
            encoded = self.encoding.fit_encode(granulated, self.encoding.params)
        return encoded, self.encoding


class TimeSinceStartFeature(TimeFeature):
    def __init__(self, time_col: pd.Series, case_col: pd.Series, spec: FeatureBuildSpec, registry: FeatureRegistry, encoding: Optional[Any] = None):
        if not isinstance(case_col, pd.Series):
            raise TypeError(f"case_col must be a pandas Series, got {type(case_col).__name__}")
        if len(time_col) != len(case_col):
            raise ValueError(f"Length mismatch: time_col={len(time_col)} vs case_col={len(case_col)}")

        super().__init__(time_col, spec, registry, encoding=encoding)
        self.case_col = case_col if case_col.index.equals(time_col.index) else case_col.reindex(time_col.index)
        if self.case_col.isnull().any():
            raise ValueError("case_col contains nulls; need a case id for every row.")

    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        ts = _to_utc_ts(self.col)
        starts = ts.groupby(self.case_col).transform("min")
        since_seconds = (ts - starts).dt.total_seconds().astype(float)
        since_seconds.name = self.spec.target_col_name or ((self.col.name or "time") + "::time_since_case_start")

        granulated = self.granularity.fit_transform(since_seconds, self.granularity.params)
        if self.encoding.is_fitted():
            encoded = self.encoding.encode(granulated)
        else:
            encoded = self.encoding.fit_encode(granulated, self.encoding.params)
        return encoded, self.encoding


class TimeSinceLastEventFeature(TimeFeature):
    def __init__(self, time_col: pd.Series, case_col: pd.Series, spec: FeatureBuildSpec, registry: FeatureRegistry, encoding: Optional[Any] = None):
        if not isinstance(case_col, pd.Series):
            raise TypeError(f"case_col must be a pandas Series, got {type(case_col).__name__}")
        if len(time_col) != len(case_col):
            raise ValueError(f"Length mismatch: time_col={len(time_col)} vs case_col={len(case_col)}")

        super().__init__(time_col, spec, registry, encoding=encoding)
        self.case_col = case_col if case_col.index.equals(time_col.index) else case_col.reindex(time_col.index)
        if self.case_col.isnull().any():
            raise ValueError("case_col contains nulls; need a case id for every row.")

    def build_feature(self) -> Tuple[pd.DataFrame, Any]:
        ts = _to_utc_ts(self.col)

        def _prev_in_case(s: pd.Series) -> pd.Series:
            s_sorted = s.sort_values()
            prev_sorted = s_sorted.shift(1)
            return prev_sorted.reindex(s.index)

        prev_ts = ts.groupby(self.case_col, group_keys=False).apply(_prev_in_case)

        since_last_seconds = (ts - prev_ts).dt.total_seconds().fillna(0.0).astype(float)
        since_last_seconds.name = self.spec.target_col_name or ((self.col.name or "time") + "::time_since_last_event")

        granulated = self.granularity.fit_transform(since_last_seconds, self.granularity.params)
        if self.encoding.is_fitted():
            encoded = self.encoding.encode(granulated)
        else:
            encoded = self.encoding.fit_encode(granulated, self.encoding.params)
        return encoded, self.encoding
