from __future__ import annotations

"""
A composable, cache-friendly rewrite of the Experiment module.

Key ideas
---------
- Each pipeline step (load, split, prefix, target, feature, transform) is a Stage with a
  clear input/output dataclass.
- Caching is handled by small wrappers. You can cache any *composition* of stages without
  duplicating logic (e.g., cache 1–3 or 1–5) while still reusing the same stage code.
- Invalidation is safer via light fingerprints and explicit version strings; we also
  continue to support joblib.Memory as the cache backend to minimize dependency churn.
- The public Experiment API and behavior remain the same.

External dependencies (as before):
- load.loaders: LoadSpec, load_events
- split.splitters: TemporalSplitSpec, temporal_split_by_case_end, KFoldSplitSpec, kfold_split
- prefix.prefixer: PrefixSpec, build_prefix_log
- feature.feature_registry: default_feature_registry, FeatureRegistry
- feature.dataframe_builder: FeatureDataFrameBuilder
- feature.feature_builder: FeatureBuildSpec
- model.transformer: LSTMTransformer
- model.lstm_models: LSTMModelSpec
- model.xgb_models: XGBModelSpec
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# --- package imports (align with your tree) ---
from load.loaders import LoadSpec, load_events
from split.splitters import (
    TemporalSplitSpec, temporal_split_by_case_end,
    KFoldSplitSpec, kfold_split,
)
from prefix.prefixer import PrefixSpec, build_prefix_log
from feature.feature_registry import default_feature_registry, FeatureRegistry
from feature.dataframe_builder import FeatureDataFrameBuilder
from feature.feature_builder import FeatureBuildSpec

# Optional joblib Memory for caching
try:
    from joblib import Memory
except Exception:  # pragma: no cover (joblib is optional)
    Memory = None


# ============================
# Utilities
# ============================

def _json_fingerprint(obj: Any) -> str:
    """Stable-ish fingerprint for config-like objects.

    Avoids serializing full DataFrames; this is used only on configs/metadata.
    """
    def _default(o):
        # Basic safe fallback for non-serializables
        if isinstance(o, Path):
            return str(o)
        try:
            return vars(o)
        except Exception:
            return str(o)

    s = json.dumps(obj, default=_default, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _dataset_fingerprint(path: Optional[Union[str, Path]]) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(path)
        st = p.stat()
        return f"{p.resolve()}::{int(st.st_mtime)}::{st.st_size}"
    except Exception:
        return str(path)


# ============================
# Stage I/O payloads
# ============================

@dataclass(frozen=True)
class LoadInput:
    dataset_cfg: Dict[str, Any]

@dataclass
class LoadOutput:
    log: pd.DataFrame
    case_col: str
    act_col: str
    ts_col: str
    dataset_fp: Optional[str]


@dataclass(frozen=True)
class SplitInput:
    log: pd.DataFrame
    case_col: str
    ts_col: str
    split_cfg: Dict[str, Any]

@dataclass
class SplitOutput:
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]


@dataclass(frozen=True)
class PrefixInput:
    log: pd.DataFrame
    case_col: str
    act_col: str
    ts_col: str
    split: SplitOutput
    prefix_cfg: Dict[str, Any]

@dataclass
class PrefixOutput:
    pref_tr: pd.DataFrame
    pref_va: pd.DataFrame
    pref_te: pd.DataFrame


@dataclass(frozen=True)
class TargetInput:
    prefix: PrefixOutput
    dataset_cfg: Dict[str, Any]

@dataclass
class TargetOutput:
    y_tr: pd.Series
    y_va: pd.Series
    y_te: pd.Series
    target_key: str


@dataclass(frozen=True)
class FeatureInput:
    prefix: PrefixOutput
    features_cfg: List[Dict[str, Any]]
    feature_cache_cfg: Dict[str, Any]
    feature_memory: Optional[Any]
    feature_registry: FeatureRegistry
    validate_triplets: bool

@dataclass
class FeatureOutput:
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    X_te: pd.DataFrame
    feature_specs: List[FeatureBuildSpec]


@dataclass(frozen=True)
class TransformInput:
    case_col: str
    ts_col: str
    prefix: PrefixOutput
    features: FeatureOutput
    targets: TargetOutput
    transformer_cfg: Dict[str, Any]

@dataclass
class TransformOutput:
    tr: Any
    va: Any
    te: Any
    inverse_y: Callable[[np.ndarray], np.ndarray]
    feature_names: List[str]


# ============================
# Stage implementations (pure logic)
# ============================

class LoadStage:
    VERSION = "v1"

    def run(self, inp: LoadInput) -> LoadOutput:
        ds = inp.dataset_cfg
        case_col = ds.get("case_id_col", "case:concept:name")
        act_col = ds.get("activity_col", "concept:name")
        ts_col = ds.get("timestamp_col", "time:timestamp")

        spec = LoadSpec(
            path=ds["path"],
            case_id_col=case_col,
            activity_col=act_col,
            timestamp_col=ts_col,
            rename=ds.get("rename"),
        )
        log = load_events(spec)
        ds_fp = _dataset_fingerprint(ds.get("path"))
        return LoadOutput(log=log, case_col=case_col, act_col=act_col, ts_col=ts_col, dataset_fp=ds_fp)


class SplitStage:
    VERSION = "v1"

    def run(self, inp: SplitInput) -> SplitOutput:
        split_cfg = inp.split_cfg or {}
        split_type = (split_cfg.get("type") or "temporal").lower()
        params = split_cfg.get("params", {})

        if split_type == "temporal":
            sp = TemporalSplitSpec(
                test_ratio=float(params.get("test_ratio", 0.15)),
                val_ratio=float(params.get("val_ratio", 0.15)),
                shuffle_within_same_timestamps=bool(params.get("shuffle_within_same_timestamps", False)),
                seed=int(params.get("seed", 42)),
                case_id_col=inp.case_col,
                timestamp_col=inp.ts_col,
            )
            res = temporal_split_by_case_end(inp.log, sp)
            return SplitOutput(res.train_cases, res.val_cases, res.test_cases)

        if split_type == "kfold":
            sp = KFoldSplitSpec(
                k_splits=int(params.get("k_splits", 5)),
                shuffle=bool(params.get("shuffle", True)),
                seed=int(params.get("seed", 42)),
                case_id_col=inp.case_col,
            )
            folds = kfold_split(inp.log, sp).folds
            tr, va = folds[0]
            return SplitOutput(tr, va, va)

        raise ValueError(f"Unknown split.type '{split_type}'")


class PrefixStage:
    VERSION = "v1"

    def run(self, inp: PrefixInput) -> PrefixOutput:
        case_col, act_col, ts_col = inp.case_col, inp.act_col, inp.ts_col

        # slice logs
        log_tr = inp.log[inp.log[case_col].isin(inp.split.train_cases)].reset_index(drop=True)
        log_va = inp.log[inp.log[case_col].isin(inp.split.val_cases)].reset_index(drop=True)
        log_te = inp.log[inp.log[case_col].isin(inp.split.test_cases)].reset_index(drop=True)

        px = inp.prefix_cfg or {}
        spec = PrefixSpec(
            min_prefix_len=int(px.get("min_prefix_len", 1)),
            max_prefix_len=px.get("max_prefix_len"),
            case_id_col=case_col,
            activity_col=act_col,
            time_col=ts_col,
        )

        pref_tr = build_prefix_log(log_tr, spec)
        pref_va = build_prefix_log(log_va, spec)
        pref_te = build_prefix_log(log_te, spec)
        return PrefixOutput(pref_tr=pref_tr, pref_va=pref_va, pref_te=pref_te)


class TargetStage:
    VERSION = "v1"

    def run(self, inp: TargetInput) -> TargetOutput:
        target_key = (inp.dataset_cfg.get("target") or "remaining_time").lower()
        px = inp.prefix
        if target_key == "remaining_time":
            y_tr = px.pref_tr.remaining_time.astype(float)
            y_va = px.pref_va.remaining_time.astype(float)
            y_te = px.pref_te.remaining_time.astype(float)
        elif target_key == "time_until_next_event":
            y_tr = px.pref_tr.time_until_next_event.astype(float)
            y_va = px.pref_va.time_until_next_event.astype(float)
            y_te = px.pref_te.time_until_next_event.astype(float)
        else:
            raise ValueError("target must be 'remaining_time' or 'time_until_next_event'")
        return TargetOutput(y_tr=y_tr, y_va=y_va, y_te=y_te, target_key=target_key)


class FeatureStage:
    VERSION = "v1"

    def run(self, inp: FeatureInput) -> FeatureOutput:
        builder = FeatureDataFrameBuilder(
            registry=inp.feature_registry,
            validate_triplets=inp.validate_triplets,
            cache=inp.feature_cache_cfg or {},
            memory=inp.feature_memory,
        )
        build_specs = [
            FeatureBuildSpec(
                feature_key=f["feature_key"],
                source_col_name=f["source_col_name"],
                encoding_key=f["encoding_key"],
                encoding_params=f.get("encoding_params"),
                target_col_name=f.get("target_col_name"),
                granularity_key=f.get("granularity_key"),
                granularity_params=f.get("granularity_params"),
                case_source=f.get("case_source"),
            )
            for f in (inp.features_cfg or [])
        ]

        X_tr, tr_specs, enc = builder.build_dataset(inp.prefix.pref_tr.prefix_log, build_specs)
        X_va, va_specs, _   = builder.build_dataset(inp.prefix.pref_va.prefix_log, build_specs, encodings=enc)
        X_te, te_specs, _   = builder.build_dataset(inp.prefix.pref_te.prefix_log, build_specs, encodings=enc)

        if list(X_tr.columns) != list(X_va.columns) or list(X_tr.columns) != list(X_te.columns):
            raise ValueError("Engineered feature columns differ across splits (train/val/test).")
        if len(tr_specs) != X_tr.shape[1] or len(va_specs) != X_va.shape[1] or len(te_specs) != X_te.shape[1]:
            raise ValueError("Mismatch between number of engineered columns and feature specs in one of the splits.")

        return FeatureOutput(X_tr=X_tr, X_va=X_va, X_te=X_te, feature_specs=te_specs)


class TransformStage:
    VERSION = "v1"

    def run(self, inp: TransformInput) -> TransformOutput:
        key = (inp.transformer_cfg.get("key") or "lstm").lower()
        params = inp.transformer_cfg.get("params", {})

        if key != "lstm":
            raise ValueError(f"Unknown transformer.key '{key}'")

        # Build sequence DataFrames: [case, time] + features
        df_tr = pd.concat([inp.prefix.pref_tr.prefix_log[[inp.case_col, inp.ts_col]], inp.features.X_tr], axis=1)
        df_va = pd.concat([inp.prefix.pref_va.prefix_log[[inp.case_col, inp.ts_col]], inp.features.X_va], axis=1)
        df_te = pd.concat([inp.prefix.pref_te.prefix_log[[inp.case_col, inp.ts_col]], inp.features.X_te], axis=1)

        # Seed determinism at transformer-level (best-effort)
        try:
            import random, numpy as _np
            os.environ["PYTHONHASHSEED"] = str(params.get("seed", 42))
            random.seed(params.get("seed", 42))
            _np.random.seed(params.get("seed", 42))
            try:
                import tensorflow as tf
                tf.keras.utils.set_random_seed(params.get("seed", 42))
                try:
                    tf.config.experimental.enable_op_determinism()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        from model.transformer import LSTMTransformer
        lstm_trf = LSTMTransformer(**params)
        prep_tr = lstm_trf.fit_transform(df_tr, inp.targets.y_tr, case_col=inp.case_col, time_col=inp.ts_col)
        prep_va = lstm_trf.transform(df_va, inp.targets.y_va, case_col=inp.case_col, time_col=inp.ts_col)
        prep_te = lstm_trf.transform(df_te, inp.targets.y_te, case_col=inp.case_col, time_col=inp.ts_col)

        # Expose feature names if available
        feature_names = list(getattr(prep_te, "feature_names", inp.features.X_te.columns))
        inverse_y = getattr(lstm_trf, "inverse_transform_y", lambda x: x)
        return TransformOutput(tr=prep_tr, va=prep_va, te=prep_te, inverse_y=inverse_y, feature_names=feature_names)


# ============================
# Composition & caching
# ============================

class Pipeline:
    def __init__(self, stages: List[Any]):
        self.stages = stages

    def run(self, x):
        for s in self.stages:
            x = s.run(x)
        return x


class CacheWrapper:
    """Wrapper around joblib.Memory (or noop) for composable caching.

    We keep joblib optional; if not available or disabled, it's a pass-through.
    """
    def __init__(self, memory: Optional[Any]):
        self.memory = memory

    def cache(self, fn: Callable) -> Callable:
        if self.memory is None:
            return fn
        return self.memory.cache(fn)


# --- Cached composite helpers (keep args primitive for stable caching keys) ---

def _load_split_prefix_impl(dataset_cfg: Dict[str, Any], split_cfg: Dict[str, Any], prefix_cfg: Dict[str, Any], *, feature_registry: FeatureRegistry) -> Dict[str, Any]:
    """Pure helper for caching stages 1–3 with primitive args.
    Returns a dict payload that mirrors the stage outputs.
    """
    # Load
    load_out = LoadStage().run(LoadInput(dataset_cfg=dataset_cfg))
    # Split
    split_out = SplitStage().run(SplitInput(log=load_out.log, case_col=load_out.case_col, ts_col=load_out.ts_col, split_cfg=split_cfg))
    # Prefix
    prefix_out = PrefixStage().run(PrefixInput(log=load_out.log, case_col=load_out.case_col, act_col=load_out.act_col, ts_col=load_out.ts_col, split=split_out, prefix_cfg=prefix_cfg))
    return dict(
        load_out=load_out,
        split_out=split_out,
        prefix_out=prefix_out,
    )


def _load_split_prefix_feat_trf_impl(
    *,
    dataset_cfg: Dict[str, Any],
    split_cfg: Dict[str, Any],
    prefix_cfg: Dict[str, Any],
    features_cfg: List[Dict[str, Any]],
    transformer_cfg: Dict[str, Any],
    feature_registry: FeatureRegistry,
    validate_feature_triplets: bool,
    feature_cache_cfg: Dict[str, Any],
    feature_memory: Optional[Any],
) -> Dict[str, Any]:
    """Pure helper for caching stages 1–5 with primitive args."""
    # 1–3
    prim = _load_split_prefix_impl(dataset_cfg, split_cfg, prefix_cfg, feature_registry=feature_registry)

    # 4: features
    feat_out = FeatureStage().run(FeatureInput(
        prefix=prim["prefix_out"],
        features_cfg=features_cfg,
        feature_cache_cfg=feature_cache_cfg,
        feature_memory=feature_memory,
        feature_registry=feature_registry,
        validate_triplets=validate_feature_triplets,
    ))

    # 3b: targets (needs dataset_cfg)
    tgt_out = TargetStage().run(TargetInput(prefix=prim["prefix_out"], dataset_cfg=dataset_cfg))

    # 5: transform
    trf_out = TransformStage().run(TransformInput(
        case_col=prim["load_out"].case_col,
        ts_col=prim["load_out"].ts_col,
        prefix=prim["prefix_out"],
        features=feat_out,
        targets=tgt_out,
        transformer_cfg=transformer_cfg,
    ))

    return dict(**prim, feat_out=feat_out, tgt_out=tgt_out, trf_out=trf_out)


# ============================
# Results container (public API)
# ============================

@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    # raw inputs
    log: pd.DataFrame
    # split IDs
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]
    # prefixing
    prefixes_train: pd.DataFrame
    prefixes_val: pd.DataFrame
    prefixes_test: pd.DataFrame
    # engineered features (event-level)
    X_tr_engineered: pd.DataFrame
    X_va_engineered: pd.DataFrame
    X_te_engineered: pd.DataFrame
    # targets (event-level)
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    # transformed payloads for model input
    X_tr_transformed: Any
    X_va_transformed: Any
    X_te_transformed: Any
    # per-feature specs aligned with feature_names / engineered columns
    feature_build_specs: List[FeatureBuildSpec]
    # fitted model + metrics (original scale)
    model: Optional[Any] = None
    metrics: Dict[str, float] = None
    # optional artifact path where results were saved
    artifact_path: Optional[str] = None
    # runtime in seconds for the whole experiment run()
    duration_sec: Optional[float] = None


# ============================
# Experiment Orchestrator
# ============================

class Experiment:
    """
    Orchestrates: Load -> Split -> Prefix -> Feature Engineering -> Transform -> Train/Eval.

    Config supports three cache knobs (same keys as before):
      load_split_prefix_cache: { enabled: bool, backend: "joblib", dir: "..." }
      feature_cache:           { enabled: bool, backend: "joblib", dir: "..." }
      load_split_prefix_feat_trf_cache: { enabled: bool, backend: "joblib", dir: "..." }

    To save results, add to your config (optional):
      output:
        path: "artifacts/exp_results.json"   # default: artifacts/experiment_<timestamp>.json
        format: "json"                        # "json" or "yaml"
    """

    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        verbose: int = 1,
        logger: Optional[Callable[[str], None]] = None,
        *,
        feature_registry: Optional[FeatureRegistry] = None,
        validate_feature_triplets: bool = True,
    ):
        self.config = self._load_config(config)
        self.feature_registry = feature_registry or default_feature_registry()
        self.validate_feature_triplets = validate_feature_triplets
        self.verbose = int(verbose)
        self._logger = logger

        # Per-feature cache
        feature_cache_cfg = self.config.get("feature_cache", {}) or {}
        self._feature_memory = None
        if feature_cache_cfg.get("enabled", False) and (feature_cache_cfg.get("backend") or "joblib").lower() == "joblib":
            if Memory is None:
                raise RuntimeError("joblib is not installed.")
            cache_dir = feature_cache_cfg.get("dir") or os.path.join(tempfile.gettempdir(), "fc")
            self._feature_memory = Memory(cache_dir, verbose=0)

        # Cached composites
        lsp_cfg = self.config.get("load_split_prefix_cache", {}) or {}
        self._lsp_memory = None
        if lsp_cfg.get("enabled", False) and (lsp_cfg.get("backend") or "joblib").lower() == "joblib":
            if Memory is None:
                raise RuntimeError("joblib.Memory backend requested but joblib is not installed.")
            cache_dir = lsp_cfg.get("dir") or os.path.join(tempfile.gettempdir(), "lspc")
            self._lsp_memory = Memory(cache_dir, verbose=0)

        lspft_cfg = self.config.get("load_split_prefix_feat_trf_cache", {}) or {}
        self._lspft_memory = None
        if lspft_cfg.get("enabled", False) and (lspft_cfg.get("backend") or "joblib").lower() == "joblib":
            if Memory is None:
                raise RuntimeError("joblib.Memory backend requested but joblib is not installed.")
            cache_dir = lspft_cfg.get("dir") or os.path.join(tempfile.gettempdir(), "lspftc")
            self._lspft_memory = Memory(cache_dir, verbose=0)

    # --------------- public API ---------------
    def run(self) -> ExperimentResult:
        cfg = self.config
        t0 = time.perf_counter()
        seed = int(cfg.get("seed", 42))
        self._set_global_seed(seed)

        def tick(msg: str) -> float:
            self._log(msg)
            return time.perf_counter()

        def dt(s: float, e: float) -> str:
            return f"{(e - s):.2f}s"

        # Prefer 1–5 cached composite if configured
        if self._lspft_memory is not None:
            self._log("(1–5) Using composite cache…")
            cached_fn = CacheWrapper(self._lspft_memory).cache(_load_split_prefix_feat_trf_impl)
            composite = cached_fn(
                dataset_cfg=cfg["dataset"],
                split_cfg=cfg.get("split", {}),
                prefix_cfg=cfg.get("prefix", {}),
                features_cfg=cfg.get("features", []) or [],
                transformer_cfg=cfg.get("transformer", {}) or {"key": "lstm", "params": {}},
                feature_registry=self.feature_registry,
                validate_feature_triplets=self.validate_feature_triplets,
                feature_cache_cfg=cfg.get("feature_cache", {}) or {},
                feature_memory=self._feature_memory,
            )
            # unpack everything we need
            load_out: LoadOutput = composite["load_out"]
            split_out: SplitOutput = composite["split_out"]
            prefix_out: PrefixOutput = composite["prefix_out"]
            feat_out: FeatureOutput = composite["feat_out"]
            tgt_out: TargetOutput = composite["tgt_out"]
            trf_out: TransformOutput = composite["trf_out"]
        else:
            # Either cache 1–3 or do them raw, then uncached 4–5
            self._log("(1–3) Loading, splitting, prefixing…")
            if self._lsp_memory is not None:
                self._log("   → using 1–3 cache")
                lsp_cached = CacheWrapper(self._lsp_memory).cache(_load_split_prefix_impl)
                lsp = lsp_cached(cfg["dataset"], cfg.get("split", {}), cfg.get("prefix", {}), feature_registry=self.feature_registry)
            else:
                lsp = _load_split_prefix_impl(cfg["dataset"], cfg.get("split", {}), cfg.get("prefix", {}), feature_registry=self.feature_registry)

            load_out: LoadOutput = lsp["load_out"]
            split_out: SplitOutput = lsp["split_out"]
            prefix_out: PrefixOutput = lsp["prefix_out"]

            self._log("(4) Building feature matrices…")
            feat_out = FeatureStage().run(FeatureInput(
                prefix=prefix_out,
                features_cfg=cfg.get("features", []) or [],
                feature_cache_cfg=cfg.get("feature_cache", {}) or {},
                feature_memory=self._feature_memory,
                feature_registry=self.feature_registry,
                validate_triplets=self.validate_feature_triplets,
            ))

            self._log("(5) Transforming data for the model…")
            tgt_out = TargetStage().run(TargetInput(prefix=prefix_out, dataset_cfg=cfg.get("dataset", {})))
            trf_out = TransformStage().run(TransformInput(
                case_col=load_out.case_col,
                ts_col=load_out.ts_col,
                prefix=prefix_out,
                features=feat_out,
                targets=tgt_out,
                transformer_cfg=cfg.get("transformer", {}) or {"key": "lstm", "params": {}},
            ))

        # Model train/eval
        self._log("(6) Creating model and training…")
        model_type = (cfg.get("model", {}).get("type") or "lstm").lower()
        model_params = cfg.get("model", {}).get("params", {})
        model = self._create_model(model_type, model_params)
        self._log(f"   → model='{model_type}' with params keys={list(model_params.keys())}")
        model.train(trf_out.tr.X, trf_out.va.X, trf_out.tr.y, trf_out.va.y)

        self._log("(7) Evaluating on test (original scale)…")
        y_pred_norm = model.predict(trf_out.te.X).to_numpy().reshape(-1)
        y_true = trf_out.inverse_y(trf_out.te.y)
        y_pred = trf_out.inverse_y(y_pred_norm)
        metrics = self._compute_metrics(y_true, y_pred)

        # Save results
        artifact_path = self._maybe_save_results(cfg, metrics)

        duration = time.perf_counter() - t0
        self._log(f"✅ Done in {duration:.2f}s")

        # Feature names: prefer prepared object names
        feature_names = list(getattr(trf_out.te, "feature_names", feat_out.X_te.columns))
        # Build ExperimentResult
        return ExperimentResult(
            config=cfg,
            log=load_out.log,
            train_cases=split_out.train_cases, val_cases=split_out.val_cases, test_cases=split_out.test_cases,
            prefixes_train=prefix_out.pref_tr.prefix_log,
            prefixes_val=prefix_out.pref_va.prefix_log,
            prefixes_test=prefix_out.pref_te.prefix_log,
            X_tr_engineered=feat_out.X_tr, X_va_engineered=feat_out.X_va, X_te_engineered=feat_out.X_te,
            y_train=tgt_out.y_tr, y_val=tgt_out.y_va, y_test=tgt_out.y_te,
            X_tr_transformed=trf_out.tr, X_va_transformed=trf_out.va, X_te_transformed=trf_out.te,
            feature_build_specs=feat_out.feature_specs,
            model=model,
            metrics=metrics,
            artifact_path=str(artifact_path) if artifact_path else None,
            duration_sec=duration,
        )

    # --------------- helpers ---------------
    @staticmethod
    def _load_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yml", ".yaml"):
            try:
                import yaml  # PyYAML
            except Exception as e:
                raise RuntimeError("YAML config provided but PyYAML is not installed.") from e
            return yaml.safe_load(text) or {}
        if suffix == ".json":
            return json.loads(text)
        # Try YAML fallback
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except Exception:
            # Try JSON fallback
            return json.loads(text)

    def _set_global_seed(self, seed: int):
        import random
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(seed)
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        if self.verbose:
            formatted = f"[Experiment]\t    {msg}"
            if self._logger:
                self._logger(formatted)
            else:
                print(formatted)

    def _create_model(self, model_type: str, model_params: Dict[str, Any]):
        mt = model_type.lower()
        if mt == "lstm":
            from model.lstm_models import LSTMModelSpec
            spec = LSTMModelSpec(**model_params)
            return spec.create()
        if mt == "xgb":
            from model.xgb_models import XGBModelSpec
            spec = XGBModelSpec(**model_params)
            return spec.create()
        raise ValueError(f"Unknown model.type '{model_type}'")

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        # drop NaNs
        m = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[m]
        y_pred = y_pred[m]

        if y_true.size == 0:
            return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan")}

        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        eps = 1e-8
        nz = np.abs(y_true) > eps
        mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0) if nz.any() else float("nan")

        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}

    def _maybe_save_results(self, cfg: Dict[str, Any], metrics: Dict[str, float]) -> Optional[Path]:
        out_cfg = cfg.get("output", {}) or {}
        save_path = out_cfg.get("path")
        save_fmt = (out_cfg.get("format") or "json").lower()

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not save_path:
            Path("artifacts").mkdir(parents=True, exist_ok=True)
            save_path = f"artifacts/experiment_{ts}.json"

        path = Path(str(save_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp_utc": ts, "config": cfg, "metrics": metrics}

        if save_fmt == "yaml" or path.suffix.lower() in (".yml", ".yaml"):
            try:
                import yaml
                path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
                self._log(f"(8) Saved results (YAML) → {path}")
                return path
            except Exception:
                pass  # fallback to JSON

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"(8) Saved results (JSON) → {path}")
        return path
