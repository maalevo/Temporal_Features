# dataframe_builder.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any, Iterable, Union
import pandas as pd
import copy
import itertools

from .feature_registry import FeatureRegistry as Registry
from .feature_builder import (
    Feature,
    TimeFeature,
    TimeSinceStartFeature,
    TimeSinceLastEventFeature,
    FeatureBuildSpec,
)


class FeatureDataFrameBuilder:
    """
    Build a single dataset by horizontally concatenating outputs of features
    defined via FeatureBuildSpec items.

    NEW:
      - `build_dataset` returns (df, specs, enc_pairs)
        where enc_pairs is List[(FeatureBuildSpec, Encoding)].
      - `encodings` argument can be either a dict or a list of pairs.
    """
    def __init__(self, registry: Registry, *, validate_triplets: bool = True, cache: Optional[dict] = None, memory: Optional[object] = None):
        self.registry = registry
        self.validate_triplets = validate_triplets
        # Optional joblib.Memory cache for per-feature builds
        self._memory = None
        if memory is not None:
            self._memory = memory
        else:
            if cache and cache.get("enabled", False):
                try:
                    from joblib import Memory
                except Exception:
                    Memory = None
                if Memory is None:
                    raise RuntimeError("joblib.Memory backend requested but joblib is not installed.")
                cache_dir = cache.get("dir", "artifacts/cache_feature_joblib")
                self._memory = Memory(cache_dir, verbose=0)

    # ---------------- helpers ----------------
    @staticmethod
    def _lookup_encoding(
        enc_pairs: Optional[List[Tuple[FeatureBuildSpec, Any]]],
        spec: FeatureBuildSpec,
    ) -> Optional[Any]:
        """
        Retrieve an Encoding for 'spec' from a list of (FeatureBuildSpec, Encoding) pairs.
        Compares by equality, not identity.
        """
        if not enc_pairs:
            return None

        for k, v in enc_pairs:
            try:
                if k == spec:
                    return v
            except Exception:
                continue
        return None

    def _build_one(
        self,
        df: pd.DataFrame,
        spec: FeatureBuildSpec,
        enc_override: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, List[FeatureBuildSpec], Any]:
        """Build a single feature and return (df, specs, encoder_used)."""

        # 1) input column
        if spec.source_col_name not in df.columns:
            raise KeyError(f"Source column '{spec.source_col_name}' not found.")
        col = df[spec.source_col_name]

        # 2) resolve feature class
        feature_cls = self.registry.get_feature(spec.feature_key)

        # 3) construct feature object
        if issubclass(feature_cls, TimeFeature):
            if self.validate_triplets:
                if not spec.granularity_key:
                    raise ValueError(f"Time feature '{spec.feature_key}' requires 'granularity_key'.")
                self.registry.validate_triplet(spec.feature_key, spec.encoding_key, spec.granularity_key)

            if feature_cls in (TimeSinceStartFeature, TimeSinceLastEventFeature):
                if not spec.case_source:
                    raise ValueError(f"{feature_cls.__name__} requires 'case_source'.")
                if spec.case_source not in df.columns:
                    raise KeyError(f"case_source '{spec.case_source}' not found.")
                feat = feature_cls(col, df[spec.case_source], spec, self.registry, encoding=enc_override)
            else:
                feat = feature_cls(col, spec, self.registry, encoding=enc_override)
        else:
            feat = feature_cls(col, spec, self.registry, encoding=enc_override)

        out_df, used_enc = feat.build_feature()
        if not isinstance(out_df, pd.DataFrame):
            raise TypeError(
                f"Feature '{spec.feature_key}' did not return a DataFrame (got {type(out_df).__name__})."
            )

        # duplicate spec for each column
        specs: List[FeatureBuildSpec] = [copy.deepcopy(spec) for _ in range(len(out_df.columns))]

        return out_df, specs, used_enc

    # ---------------- main ----------------
    def build_dataset(
        self,
        df: pd.DataFrame,
        specs: List[FeatureBuildSpec],
        encodings: Optional[Union[Dict[FeatureBuildSpec, Any], List[Tuple[FeatureBuildSpec, Any]]]] = None,
    ) -> Tuple[pd.DataFrame, List[FeatureBuildSpec], List[Tuple[FeatureBuildSpec, Any]]]:
        """
        Build a dataset with multiple features.

        Returns:
          result_df        : pd.DataFrame with engineered features
          result_specs     : List[FeatureBuildSpec], 1 per output column
          result_encodings : List[(FeatureBuildSpec, Encoding)], 1 per input spec
        """
        built_parts: List[pd.DataFrame] = []
        spec_lists: List[List[FeatureBuildSpec]] = []
        result_encodings: List[Tuple[FeatureBuildSpec, Any]] = []

        for s in specs:
            enc_override = self._lookup_encoding(encodings, s)
            feat_df, spec_list_for_feat, used_enc = self._build_one(df, s, enc_override=enc_override)
            built_parts.append(feat_df)
            spec_lists.append(spec_list_for_feat)
            result_encodings.append((s, used_enc))

        # concat dataframes
        if built_parts:
            result_df = pd.concat(built_parts, axis=1)
        else:
            result_df = pd.DataFrame(index=df.index)

        result_specs: List[FeatureBuildSpec] = list(itertools.chain.from_iterable(spec_lists))

        if len(result_specs) != result_df.shape[1]:
            raise ValueError(
                f"Mismatched build outputs: got {result_df.shape[1]} columns but {len(result_specs)} specs."
            )

        if result_df.columns.duplicated().any():
            dups = result_df.columns[result_df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate output columns in dataset: {dups}")

        return result_df, result_specs, result_encodings
