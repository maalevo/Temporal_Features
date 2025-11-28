# time_feature_benchmark/feature/feature_registry.py
from __future__ import annotations

from typing import Dict, Type, Optional, Iterable, Tuple, Union, List, TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only for type hints; avoid importing at module import time to prevent cycles.
    from .encodings import Encoding
    from .granularities import Granularity
    from .feature_builder import Feature, TimeFeature

# ---- Type aliases -----------------------------------------------------------
NameOrEncoding = Union[str, Type["Encoding"]]
NameOrGranularity = Union[str, Type["Granularity"]]
NameOrFeature = Union[str, Type["Feature"]]

Triplet = Tuple[str, str, str]  # (feature_key, encoding_key, granularity_key)

# FeatureSpec support is optional. If you have dedicated preset classes/types,
# you can import and use them. Otherwise this remains as a generic "Any".
FeatureSpecLike = Any
FeatureSpecPair = Tuple[str, str]  # (feature_key, featurespec_key)


class FeatureRegistry:
    """
    Plugin registry for:
      - Encodings     (key -> Encoding subclass)
      - Granularities (key -> Granularity subclass)
      - Features      (key -> Feature/TimeFeature subclass)
      - FeatureSpec presets (key -> FeatureSpec/TimeFeatureSpec instance) [optional]

    It can also enforce *constraints*:
      - Triplets: allowed (feature, encoding, granularity) combinations
      - Feature ↔ FeatureSpec: allowed curated presets per feature
    """

    def __init__(self):
        # Classes
        self._encodings: Dict[str, Type["Encoding"]] = {}
        self._granularities: Dict[str, Type["Granularity"]] = {}
        self._features: Dict[str, Type["Feature"]] = {}

        # Reverse lookup class -> key
        self._encoding_name_by_cls: Dict[Type["Encoding"], str] = {}
        self._granularity_name_by_cls: Dict[Type["Granularity"], str] = {}
        self._feature_name_by_cls: Dict[Type["Feature"], str] = {}

        # Constraints
        self._allowed_triplets: set[Triplet] = set()

        # FeatureSpec presets (name -> spec instance) [optional]
        self._feature_specs: Dict[str, FeatureSpecLike] = {}
        # Allowed feature↔featurespec pairs
        self._allowed_feature_x_spec: set[FeatureSpecPair] = set()

    # -------------------------------------------------------------------------
    # Encodings
    # -------------------------------------------------------------------------
    def register_encoding(self, name: str, cls: Type["Encoding"], *, overwrite: bool = False) -> None:
        if not overwrite and name in self._encodings:
            raise ValueError(f"Encoding '{name}' already registered.")
        self._encodings[name] = cls
        self._encoding_name_by_cls[cls] = name

    def get_encoding(self, name: str) -> Type["Encoding"]:
        try:
            return self._encodings[name]
        except KeyError as e:
            raise KeyError(f"Unknown encoding '{name}'. Registered: {sorted(self._encodings)}") from e

    def encoding(self, name: str):
        """Decorator: @registry.encoding('name')"""
        def deco(cls: Type["Encoding"]):
            self.register_encoding(name, cls, overwrite=True)
            return cls
        return deco

    def list_encodings(self) -> Dict[str, Type["Encoding"]]:
        return dict(self._encodings)

    # -------------------------------------------------------------------------
    # Granularities
    # -------------------------------------------------------------------------
    def register_granularity(self, name: str, cls: Type["Granularity"], *, overwrite: bool = False) -> None:
        if not overwrite and name in self._granularities:
            raise ValueError(f"Granularity '{name}' already registered.")
        self._granularities[name] = cls
        self._granularity_name_by_cls[cls] = name

    def get_granularity(self, name: str) -> Type["Granularity"]:
        try:
            return self._granularities[name]
        except KeyError as e:
            raise KeyError(f"Unknown granularity '{name}'. Registered: {sorted(self._granularities)}") from e

    def granularity(self, name: str):
        """Decorator: @registry.granularity('name')"""
        def deco(cls: Type["Granularity"]):
            self.register_granularity(name, cls, overwrite=True)
            return cls
        return deco

    def list_granularities(self) -> Dict[str, Type["Granularity"]]:
        return dict(self._granularities)

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------
    def register_feature(self, name: str, cls: Type["Feature"], *, overwrite: bool = False) -> None:
        if not overwrite and name in self._features:
            raise ValueError(f"Feature '{name}' already registered.")
        self._features[name] = cls
        self._feature_name_by_cls[cls] = name

    def get_feature(self, name: str) -> Type["Feature"]:
        try:
            return self._features[name]
        except KeyError as e:
            raise KeyError(f"Unknown feature '{name}'. Registered: {sorted(self._features)}") from e

    def feature(self, name: str):
        """Decorator: @registry.feature('name')"""
        def deco(cls: Type["Feature"]):
            self.register_feature(name, cls, overwrite=True)
            return cls
        return deco

    def list_features(self) -> Dict[str, Type["Feature"]]:
        return dict(self._features)

    # -------------------------------------------------------------------------
    # FeatureSpec presets (optional)
    # -------------------------------------------------------------------------
    def register_feature_spec(self, name: str, spec: FeatureSpecLike, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._feature_specs:
            raise ValueError(f"FeatureSpec preset '{name}' already registered.")
        self._feature_specs[name] = spec

    def get_feature_spec(self, name: str) -> FeatureSpecLike:
        try:
            return self._feature_specs[name]
        except KeyError as e:
            raise KeyError(f"Unknown FeatureSpec preset '{name}'. Registered: {sorted(self._feature_specs)}") from e

    def list_feature_specs(self) -> Dict[str, FeatureSpecLike]:
        return dict(self._feature_specs)

    # -------------------------------------------------------------------------
    # Triplet constraints (feature, encoding, granularity)
    # -------------------------------------------------------------------------
    def allow_triplet(self, feature: NameOrFeature, encoder: NameOrEncoding, granularity: NameOrGranularity) -> None:
        f = self._normalize_feature_key(feature)
        e = self._normalize_encoding_key(encoder)
        g = self._normalize_granularity_key(granularity)
        self._allowed_triplets.add((f, e, g))

    def set_allowed_triplets(self, triplets: Iterable[Tuple[NameOrFeature, NameOrEncoding, NameOrGranularity]]) -> None:
        self._allowed_triplets.clear()
        for f, e, g in triplets:
            self.allow_triplet(f, e, g)

    def get_allowed_triplets(self) -> List[Triplet]:
        return sorted(self._allowed_triplets)

    def get_allowed_for_feature(self, feature: NameOrFeature) -> List[Triplet]:
        f = self._normalize_feature_key(feature)
        return sorted(t for t in self._allowed_triplets if t[0] == f)

    def validate_triplet(
        self,
        feature: NameOrFeature,
        encoder: NameOrEncoding,
        granularity: NameOrGranularity,
        *,
        raise_on_error: bool = True,
    ) -> bool:
        f = self._normalize_feature_key(feature)
        e = self._normalize_encoding_key(encoder)
        g = self._normalize_granularity_key(granularity)
        key = (f, e, g)
        if key in self._allowed_triplets:
            return True
        if raise_on_error:
            allowed = self.get_allowed_for_feature(f)
            suggestions = ", ".join([f"({af},{ae},{ag})" for af, ae, ag in allowed]) or "[]"
            raise ValueError(
                f"Combination not allowed: feature='{f}', encoder='{e}', granularity='{g}'. "
                f"Allowed for '{f}': {suggestions}"
            )
        return False

    # -------------------------------------------------------------------------
    # Feature ↔ FeatureSpec constraints (optional)
    # -------------------------------------------------------------------------
    def allow_feature_spec(self, feature: NameOrFeature, spec_key: str) -> None:
        f = self._normalize_feature_key(feature)
        if spec_key not in self._feature_specs:
            raise KeyError(
                f"Unknown FeatureSpec preset '{spec_key}'. "
                f"Registered: {sorted(self._feature_specs)}"
            )
        self._allowed_feature_x_spec.add((f, spec_key))

    def set_allowed_feature_specs(self, pairs: Iterable[Tuple[NameOrFeature, str]]) -> None:
        self._allowed_feature_x_spec.clear()
        for f, s in pairs:
            self.allow_feature_spec(f, s)

    def get_allowed_feature_specs(self) -> List[FeatureSpecPair]:
        return sorted(self._allowed_feature_x_spec)

    def get_specs_for_feature(self, feature: NameOrFeature) -> List[FeatureSpecPair]:
        f = self._normalize_feature_key(feature)
        return sorted(p for p in self._allowed_feature_x_spec if p[0] == f)

    def validate_feature_spec(
        self,
        feature: NameOrFeature,
        spec_key: str,
        *,
        raise_on_error: bool = True,
    ) -> bool:
        f = self._normalize_feature_key(feature)
        if spec_key not in self._feature_specs:
            raise KeyError(
                f"Unknown FeatureSpec preset '{spec_key}'. "
                f"Registered: {sorted(self._feature_specs)}"
            )
        key = (f, spec_key)
        if key in self._allowed_feature_x_spec:
            return True
        if raise_on_error:
            allowed = self.get_specs_for_feature(f)
            suggestions = ", ".join([f"({af},{as_})" for af, as_ in allowed]) or "[]"
            raise ValueError(
                f"Combination not allowed: feature='{f}', featurespec='{spec_key}'. "
                f"Allowed for '{f}': {suggestions}"
            )
        return False

    # -------------------------------------------------------------------------
    # Helpers (name/class normalization)
    # -------------------------------------------------------------------------
    def _normalize_feature_key(self, obj: NameOrFeature) -> str:
        if isinstance(obj, str):
            self.get_feature(obj)  # validate existence
            return obj
        try:
            return self._feature_name_by_cls[obj]  # type: ignore[index]
        except KeyError as e:
            raise KeyError(f"Feature class {obj} is not registered; register it before using constraints.") from e

    def _normalize_encoding_key(self, obj: NameOrEncoding) -> str:
        if isinstance(obj, str):
            self.get_encoding(obj)  # validate existence
            return obj
        try:
            return self._encoding_name_by_cls[obj]  # type: ignore[index]
        except KeyError as e:
            raise KeyError(f"Encoding class {obj} is not registered; register it before using constraints.") from e

    def _normalize_granularity_key(self, obj: NameOrGranularity) -> str:
        if isinstance(obj, str):
            self.get_granularity(obj)  # validate existence
            return obj
            
        try:
            return self._granularity_name_by_cls[obj]  # type: ignore[index]
        except KeyError as e:
            raise KeyError(f"Granularity class {obj} is not registered; register it before using constraints.") from e


# -----------------------------------------------------------------------------


def default_feature_registry() -> FeatureRegistry:
    """
    Create a registry and preload built-ins + allowed triplets.
    Designed to work with the provided feature_builder.py (FeatureBuildSpec-based).

    This function fails loudly if the core modules are missing (so the user
    doesn't end up with an *empty* registry), but it keeps FeatureSpec presets
    optional and will skip them if the preset classes are not present.
    """
    reg = FeatureRegistry()

    # --- Encodings -----------------------------------------------------------
    # Expect these in time_feature_benchmark/feature/encodings.py
    try:
        from .encodings import TextualEncoding, NumericalEncoding, OneHotEncoding, SinCosEncoding
    except Exception as e:
        raise ImportError(
            "Failed to import encodings. Ensure feature/encodings.py defines "
            "TextualEncoding, NumericalEncoding, OneHotEncoding, SinCosEncoding."
        ) from e

    reg.register_encoding("text", TextualEncoding)
    reg.register_encoding("numeric", NumericalEncoding)
    reg.register_encoding("onehot", OneHotEncoding)
    reg.register_encoding("sincos", SinCosEncoding)

    # --- Granularities -------------------------------------------------------
    # Expect these in time_feature_benchmark/feature/granularities.py
    try:
        from .granularities import (
            SecondGranularity,
            MinuteGranularity,
            HourGranularity,
            DayGranularity,
            WeekGranularity,
            MonthGranularity
        )
    except Exception as e:
        raise ImportError(
            "Failed to import granularities. Ensure feature/granularities.py defines "
            "SecondGranularity, MinuteGranularity, HourGranularity, DayGranularity, WeekGranularity, MonthGranularity."
        ) from e

    reg.register_granularity("second", SecondGranularity)
    reg.register_granularity("minute", MinuteGranularity)
    reg.register_granularity("hour", HourGranularity)
    reg.register_granularity("day", DayGranularity)
    reg.register_granularity("week", WeekGranularity)
    reg.register_granularity("month", MonthGranularity)

    # --- Features ------------------------------------------------------------
    # Import *inside* the function to avoid circular imports at module import time.
    try:
        from .feature_builder import (
            ActivityFeature,
            TimeFeature,
            UTCTimeFeature,
            TimeInDayFeature,
            TimeInWeekFeature,
            TimeInMonthFeature,
            TimeInYearFeature,
            TimeSinceStartFeature,
            TimeSinceLastEventFeature
        )
    except Exception as e:
        raise ImportError(
            "Failed to import features from feature_builder.py. Make sure the file you shared "
            "is located at feature/feature_builder.py and is importable."
        ) from e

    reg.register_feature("activity", ActivityFeature)  
    reg.register_feature("time", TimeFeature)  # abstract router; useful for constraints & presets
    reg.register_feature("utc", UTCTimeFeature)
    reg.register_feature("time_in_day", TimeInDayFeature)
    reg.register_feature("time_in_week", TimeInWeekFeature)
    reg.register_feature("time_in_month", TimeInMonthFeature)
    reg.register_feature("time_in_year", TimeInYearFeature)
    reg.register_feature("time_since_start", TimeSinceStartFeature)
    reg.register_feature("time_since_last_event", TimeSinceLastEventFeature)


    # --- Allowed triplets ----------------------------------------------------
    # utc
    for g in ("second", "minute", "hour", "day", "week", "month"):
        reg.allow_triplet("utc", "numeric", g)

    # time_in_day
    for g in ("second", "minute", "hour"):
        reg.allow_triplet("time_in_day", "numeric", g)
        reg.allow_triplet("time_in_day", "sincos", g)
    reg.allow_triplet("time_in_day", "onehot", "hour")

    # time_in_week
    for g in ("second", "minute", "hour", "day"):
        reg.allow_triplet("time_in_week", "numeric", g)
        reg.allow_triplet("time_in_week", "sincos", g)
    reg.allow_triplet("time_in_week", "onehot", "day")

    # time_in_month
    for g in ("second", "minute", "hour", "day", "week"):
        reg.allow_triplet("time_in_month", "numeric", g)
        reg.allow_triplet("time_in_month", "sincos", g)
    for g in ("day", "week"):
        reg.allow_triplet("time_in_month", "onehot", g)
    
    # time_in_year
    for g in ("second", "minute", "hour", "day", "week", "month"):
        reg.allow_triplet("time_in_year", "numeric", g)
        reg.allow_triplet("time_in_year", "sincos", g)
    for g in ("week", "month"):
        reg.allow_triplet("time_in_year", "onehot", g)

    # time_since_start
    for g in ("second", "minute", "hour", "day", "week", "month"):
        reg.allow_triplet("time_since_start", "numeric", g)
    
    # time_since_last_event
    for g in ("second", "minute", "hour", "day", "week", "month"):
        reg.allow_triplet("time_since_last_event", "numeric", g)

    return reg
