from __future__ import annotations
"""Permutation-Importance RFE

This module implements greedy Recursive Feature Elimination (RFE) driven by
permutation importance for sequence models that consume tensors of shape
``(N, T, F)``. At each iteration, it:

1. Trains once on the current feature set (base + remaining elimination features).
2. Chooses an evaluation split (test/val/train) and computes a baseline metric
   on the *original* target scale.
3. For each *feature group* (i.e., the set of engineered columns produced by the
   same `FeatureBuildSpec`), permutes that group **together** within equal-length
   sequences and re-scores **without retraining**.
4. Computes permutation importance per group and removes the group with the
   smallest importance.
5. Repeats until `min_features` remain.

Public API:
- ``RFEIterationResult``: record for each elimination step
- ``RFEPermutationBatcher``: main driver class

Notes
-----
- Grouping by feature spec is critical: engineered columns from the same feature
  must move together during permutation to avoid leakages across time/columns.
- Permutations are constrained within equal-length sequences to respect padding.
- Passes through optional RFE-level cache config blocks (``feature_cache`` and
  ``load_split_prefix_cache``) to each sub-experiment so Experiment can leverage
  caching consistently across iterations.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple
import json

import sys
import argparse

import numpy as np

import time

from .experiment import Experiment, ExperimentResult
from feature.feature_registry import FeatureRegistry


# =============================================================================
# Metric helpers
# =============================================================================

def _metric_direction_is_higher_better(metric: str) -> bool:
    """Return True if larger values of the metric are better.

    This helps convert between "loss-like" metrics (MAE/MSE/etc.) and
    "score-like" metrics (accuracy/AUC/etc.) when computing permutation
    importance deltas.
    """
    return metric.lower() in {"accuracy", "auc", "f1", "precision", "recall"}


def _compute_metrics_original_scale(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics on the target's original scale.

    NaNs are ignored pairwise. Returns a dict with keys: ``mae``, ``mse``,
    ``rmse``, ``mape``.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.size == 0:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    eps = 1e-8
    nz = np.abs(y_true) > eps
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0) if nz.any() else float("nan")

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


# =============================================================================
# Padding / lengths helpers
# =============================================================================

def _infer_lengths_from_padding(X: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Infer per-sequence lengths from padded 3D tensor ``X`` of shape ``(N, T, F)``.

    A timestep is considered padded if **all** features at that step equal
    ``pad_value``. Length = number of **unpadded** steps.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X (N,T,F); got {X.shape}")

    N, T, F = X.shape
    is_pad = np.all(np.isclose(X, pad_value), axis=2)  # (N, T)
    # argmax on the reversed axis finds the first pad from the end; T - idx gives length
    lengths = T - np.argmax(is_pad[:, ::-1], axis=1)
    return lengths


# =============================================================================
# Feature spec grouping helpers
# =============================================================================

def _spec_group_key_from_spec(spec: Any) -> str:
    """Stable JSON key for grouping columns by ``FeatureBuildSpec``.

    Includes all relevant fields so only truly identical specs group together.
    """
    payload = {
        "feature_key": getattr(spec, "feature_key", None),
        "source_col_name": getattr(spec, "source_col_name", None),
        "encoding_key": getattr(spec, "encoding_key", None),
        "encoding_params": getattr(spec, "encoding_params", None),
        "target_col_name": getattr(spec, "target_col_name", None),
        "granularity_key": getattr(spec, "granularity_key", None),
        "granularity_params": getattr(spec, "granularity_params", None),
        "case_source": getattr(spec, "case_source", None),
    }
    return json.dumps(payload, sort_keys=True)


def _spec_group_key_from_cfg(d: Dict[str, Any]) -> str:
    """Same key builder as ``_spec_group_key_from_spec`` but from a config dict.

    Missing fields default to ``None`` to match keys derived from
    ``ExperimentResult`` specs.
    """
    payload = {
        "feature_key": d.get("feature_key"),
        "source_col_name": d.get("source_col_name"),
        "encoding_key": d.get("encoding_key"),
        "encoding_params": d.get("encoding_params"),
        "target_col_name": d.get("target_col_name"),
        "granularity_key": d.get("granularity_key"),
        "granularity_params": d.get("granularity_params"),
        "case_source": d.get("case_source"),
    }
    return json.dumps(payload, sort_keys=True)


def _feat_id(f: Dict[str, Any]) -> str:
    """Human-friendly identifier for a feature config dict."""
    return "-".join([
        str(f.get("feature_key", "")),
        str(f.get("granularity_key", "")),
        str(f.get("encoding_key", "")),
    ])


# =============================================================================
# Target (y) de-normalization
# =============================================================================

def _build_y_denorm_from_meta(prep: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Create a denormalizer for y from a prepared payload's meta.

    Mirrors the behavior expected by the LSTM/Transformer training pipeline.
    Supported modes: ``None``, ``"zscore"``, ``"0/1"``, ``"-1/1"``.
    """
    meta = getattr(prep, "meta", {}) or {}
    mode = (meta.get("y_normalization") or None)
    stats = meta.get("y_stats", {}) or {}

    m, s = stats.get("mean"), stats.get("std")
    ymin, ymax = stats.get("min"), stats.get("max")

    def denorm(y: np.ndarray) -> np.ndarray:
        arr = np.asarray(y, dtype=float)
        if mode is None:
            return arr
        if mode == "zscore":
            mu = 0.0 if m is None else float(m)
            sd = 1.0 if s is None or s == 0 else float(s)
            return arr * sd + mu
        if mode == "0/1":
            lo = 0.0 if ymin is None else float(ymin)
            hi = 1.0 if ymax is None else float(ymax)
            return arr * (hi - lo) + lo
        if mode == "-1/1":
            lo = 0.0 if ymin is None else float(ymin)
            hi = 1.0 if ymax is None else float(ymax)
            return ((arr + 1.0) / 2.0) * (hi - lo) + lo
        return arr

    return denorm


# =============================================================================
# Permutation helper
# =============================================================================

def _permute_feature_group_by_seq_len(
    X: np.ndarray,
    feature_indices: Sequence[int],
    lengths: Optional[np.ndarray] = None,
    pad_value: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Permute a *group* of feature columns together, respecting sequence lengths.

    For each **unique** sequence length ``L`` and for each timestep ``t < L``,
    apply the **same row permutation** to all columns in the group. Padded steps
    (``t >= L``) are left untouched.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X (N,T,F); got {X.shape}")

    N, T, F = X.shape

    idxs = np.asarray(list(feature_indices))
    if idxs.ndim != 1 or (idxs < 0).any() or (idxs >= F).any():
        raise ValueError(f"feature_indices out of bounds for F={F}")

    if lengths is None:
        lengths = _infer_lengths_from_padding(X, pad_value=pad_value)
    lengths = np.asarray(lengths)
    if lengths.shape != (N,):
        raise ValueError("lengths must have shape (N,)")

    Xp = X.copy()
    rng = rng or np.random.default_rng()

    # Iterate over sequences grouped by the same (integer) length
    for L in np.unique(lengths.astype(int)):
        if L <= 0:
            continue
        row_idx = np.flatnonzero(lengths == L)
        if row_idx.size <= 1:
            continue

        p = rng.permutation(row_idx.size)
        src_rows = row_idx[p]
        block = Xp[np.ix_(src_rows, np.arange(L), idxs)].copy()  # avoid aliasing
        Xp[np.ix_(row_idx, np.arange(L), idxs)] = block

    return Xp


def _mask_feature_group(
    X: np.ndarray,
    feature_indices: Sequence[int],
    lengths: Optional[np.ndarray] = None,
    pad_value: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Mask a *group* of feature columns, respecting sequence lengths.

    Default: set each selected feature column to its mean computed over all
    non-padded timesteps across the batch. If the selected columns appear to be
    a one-hot encoding (per timestep across the group), set them all to zero instead.

    Padding (t >= length[n]) is left untouched.
    """
    # Basic checks
    if X.ndim != 3:
        raise ValueError(f"Expected X (N,T,F); got {X.shape}")
    N, T, F = X.shape

    idxs = np.asarray(list(feature_indices))
    if idxs.ndim != 1 or (idxs < 0).any() or (idxs >= F).any():
        raise ValueError(f"feature_indices out of bounds for F={F}")
    if idxs.size == 0:
        return X.copy()

    # Infer lengths if needed
    if lengths is None:
        lengths = _infer_lengths_from_padding(X, pad_value=pad_value)
    lengths = np.asarray(lengths)
    if lengths.shape != (N,):
        raise ValueError("lengths must have shape (N,)")

    Xp = X.copy()

    # Mask for valid (non-padded) positions: shape (N, T)
    valid_mask = (np.arange(T)[None, :] < lengths[:, None])

    # Gather values for detection & statistics (only valid positions)
    # Shape: (#valid, G) where G = len(idxs)
    group_vals_2d = X[:, :, idxs][valid_mask]
    if group_vals_2d.size == 0:
        # Nothing to modify if there are no valid timesteps
        return Xp

    # --- Detect one-hot encoding across the group ---
    # Allow tiny numerical noise around 0/1.
    atol = 1e-6
    is_binary = np.all(np.isclose(group_vals_2d, 0.0, atol=atol) | np.isclose(group_vals_2d, 1.0, atol=atol))
    row_sums = group_vals_2d.sum(axis=1)
    # Accept sums of 0 or 1 to allow "all-zero" states (e.g., unknown) alongside true one-hots.
    sums_are_0_or_1 = np.all(np.isclose(row_sums, 0.0, atol=atol) | np.isclose(row_sums, 1.0, atol=atol))
    looks_one_hot = is_binary and sums_are_0_or_1

    if looks_one_hot:
        # Set all selected columns to zero for valid positions
        for f in idxs:
            # Use simple slicing so boolean assignment writes into Xp
            arr = Xp[:, :, f]
            arr[valid_mask] = 0.0
    else:
        # Compute per-feature means over valid positions (ignoring padding)
        means = np.empty(idxs.size, dtype=np.float64)
        for j, f in enumerate(idxs):
            vals = X[:, :, f][valid_mask]
            means[j] = float(vals.mean()) if vals.size else 0.0

        # Assign means back into valid positions
        for j, f in enumerate(idxs):
            arr = Xp[:, :, f]
            arr[valid_mask] = means[j]

    return Xp

# =============================================================================
# Results dataclass
# =============================================================================

@dataclass
class RFEIterationResult:
    """Record of a single permutation-importance elimination step."""

    timestamp_utc: str
    eliminated_features: List[Dict[str, Any]]
    base_features: List[Dict[str, Any]]
    remaining_elimination_features: List[Dict[str, Any]]
    n_features: int
    metric_key: str
    baseline_metric_value: float
    elimination_metric_value: float
    elimination_importance: float
    artifact_path: Optional[str]


# =============================================================================
# Main driver
# =============================================================================

class RFEPermutationBatcher:
    """Greedy RFE driven by permutation importance.

    Parameters
    ----------
    config
        Experiment config (dict or path to file). Supports top-level
        ``{"experiments": [ ... ]}`` by using the first experiment.
    verbose
        Verbosity level (``0`` = silent).
    logger
        Optional callable used for logging messages (defaults to ``print`` when
        ``verbose`` > 0).
    feature_registry
        Optional ``FeatureRegistry`` used by ``Experiment``.
    """

    def __init__(
        self,
        config: Union[str, Path, Dict[str, Any]],
        verbose: int = 1,
        logger: Optional[Callable[[str], None]] = None,
        *,
        feature_registry: Optional[FeatureRegistry] = None,
    ) -> None:
        self.verbose = int(verbose)
        self._logger = logger
        self.feature_registry = feature_registry

        self._cfg_raw = Experiment._load_config(config)
        self._base_cfg = json.loads(json.dumps(self._cfg_raw))  # deep copy

        rfe_cfg = self._base_cfg.get("rfe", {}) or {}
        self.min_features = int(rfe_cfg.get("min_features", 1))
        self.metric_key = (rfe_cfg.get("metric", "mae") or "mae").lower()
        self.output_dir = rfe_cfg.get("output_dir", "artifacts/recursive_feature_elimination")
        # Optional cache config blocks to be injected into each sub-experiment
        self.feature_cache_cfg = json.loads(json.dumps(rfe_cfg.get("feature_cache", {}) or {}))
        self.load_split_prefix_cache_cfg = json.loads(json.dumps(rfe_cfg.get("load_split_prefix_cache", {}) or {}))

        pi_cfg = rfe_cfg.get("permutation", {}) or {}
        self.pad_value = float(pi_cfg.get("pad_value", 0.0))
        self.eval_split = str(pi_cfg.get("eval_split", "test")).lower()  # "test" | "val" | "train"
        # Seed precedence: permutation.random_state (legacy) > rfe.seed > 13
        perm_seed = pi_cfg.get("random_state")
        self.permutation_seed = int(perm_seed) if perm_seed is not None else int(rfe_cfg.get("seed", 13))

        # Support configs wrapped under top-level `experiments:` (use first)
        if isinstance(self._cfg_raw, dict) and "experiments" in self._cfg_raw:
            exps = self._cfg_raw.get("experiments") or []
            if isinstance(exps, list) and len(exps) > 0:
                self._base_cfg = json.loads(json.dumps(exps[0]))

        self.base_features = list(self._base_cfg.get("base_features", []) or [])
        self.initial_elimination_features = list(self._base_cfg.get("elimination_features", []) or [])
        if not self.initial_elimination_features:
            raise ValueError("Config must contain a non-empty 'elimination_features' list to eliminate from.")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if not self.verbose:
            return
        if self._logger:
            self._logger(f"[RFE]\t{msg}")
        else:
            print(msg)

    def _make_cfg(self, features_subset: List[Dict[str, Any]], save_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Build a sub-experiment config with a subset of features."""
        cfg = json.loads(json.dumps(self._base_cfg))
        cfg["features"] = json.loads(json.dumps(features_subset))
        cfg_name = cfg.get("name") or "pi_rfe_subexp"
        cfg["name"] = f"{cfg_name}_nfeat{len(features_subset)}"
        # Inject cache blocks so Experiment can reuse cached artifacts if configured
        if self.load_split_prefix_cache_cfg:
            cfg["load_split_prefix_cache"] = json.loads(json.dumps(self.load_split_prefix_cache_cfg))
        if self.feature_cache_cfg:
            cfg["feature_cache"] = json.loads(json.dumps(self.feature_cache_cfg))
        # Optional: forward an explicit artifact path to the Experiment
        if save_path is not None:
            out = cfg.get("output") or {}
            out["path"] = str(save_path)
            cfg["output"] = out
        return cfg

    def _get_eval_payload(self, res: ExperimentResult):
        """Select tensors and helpers for the desired evaluation split.

        Returns
        -------
        X_eval : np.ndarray
            Evaluation features, expected shape (N, T, F).
        y_eval : np.ndarray
            Ground-truth target values on original scale.
        y_denorm : Callable[[np.ndarray], np.ndarray]
            Function to map model predictions from normalized to original scale.
        lengths : np.ndarray
            Sequence lengths for each sample (shape: (N,)).
        """
        split = self.eval_split
        if split == "test":
            prep, y_series = res.X_te_transformed, res.y_test
        elif split == "val":
            prep, y_series = res.X_va_transformed, res.y_val
        elif split == "train":
            prep, y_series = res.X_tr_transformed, res.y_train
        else:
            raise KeyError("rfe.permutation.eval_split must be one of {'test','val','train'}")

        X_eval = prep.X  # expected (N, T, F)
        y_eval_norm = getattr(prep, "y", None)

        # Denormalizer from meta (mirrors LSTM/Transformer)
        y_denorm = _build_y_denorm_from_meta(prep)
        y_eval = y_denorm(y_eval_norm) if y_eval_norm is not None else y_series.to_numpy()

        # Sequence lengths
        lengths = getattr(prep, "seq_lengths", None)
        if lengths is None:
            lengths = _infer_lengths_from_padding(X_eval, pad_value=self.pad_value)
        return X_eval, y_eval, y_denorm, lengths

    # ---------------------------------------------------------------------
    # Main routine
    # ---------------------------------------------------------------------
    def run(self) -> List[RFEIterationResult]:
        """Run permutation-importance RFE and return per-iteration records."""
        records: List[RFEIterationResult] = []
        current_elim = json.loads(json.dumps(self.initial_elimination_features))

        self._log(f"Dataset: {self._base_cfg.get('dataset', {}).get('path', 'N/A')}")
        total_iters = max(len(current_elim) - max(self.min_features, 0), 0)
        self._log(f"Total iterations: {total_iters}")

        iter_idx = 0
        higher_better = _metric_direction_is_higher_better(self.metric_key)

        while True:
            n_cur = len(current_elim)
            if iter_idx > 0 and n_cur <= self.min_features:
                self._log("Reached min_features; stopping.")
                break

            self._log("========================================")
            self._log(
                f"PI-RFE ITERATION ({iter_idx + 1}/{total_iters}) — pool size: {n_cur}"
            )
            self._log("========================================")
            iter_start = time.perf_counter()

            # Train baseline on current feature set
            full_features = json.loads(json.dumps(self.base_features)) + json.loads(json.dumps(current_elim))
            train_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            cand_path = Path(self.output_dir) / f"pi_perm_iter{iter_idx}_{train_ts}.json"
            cfg = self._make_cfg(full_features, save_path=cand_path)
            exp = Experiment(cfg, verbose=max(0, self.verbose - 1), feature_registry=self.feature_registry)
            res: ExperimentResult = exp.run()

            # Build groups of feature indices by their FeatureBuildSpec
            specs = getattr(res, "feature_build_specs", None)
            if specs is None:
                raise RuntimeError(
                    "ExperimentResult is missing 'feature_build_specs'. "
                    "Ensure Experiment returns (df, specs) from the builder and passes them through."
                )

            group_to_indices: Dict[str, List[int]] = {}
            for i, sp in enumerate(specs):
                gk = _spec_group_key_from_spec(sp)
                group_to_indices.setdefault(gk, []).append(i)

            # Select eval payload and baseline metric (original scale)
            X_eval, y_eval, y_denorm, lengths = self._get_eval_payload(res)
            if not isinstance(X_eval, np.ndarray):
                X_eval = np.asarray(X_eval)

            # Baseline metric: if we're on test, reuse experiment metrics; else recompute
            if self.eval_split == "test":
                base_val = float(res.metrics.get(self.metric_key, np.nan))
            else:
                y_pred_norm = res.model.predict(X_eval).to_numpy().reshape(-1)
                y_pred = y_denorm(y_pred_norm)
                base_metrics_tmp = _compute_metrics_original_scale(y_eval, y_pred)
                base_val = float(base_metrics_tmp.get(self.metric_key, np.nan))

            self._log(f"• Baseline {self.metric_key} = {base_val:.6f}")

            # Permutation loop (by spec groups)
            rng = np.random.default_rng(self.permutation_seed + iter_idx)
            feature_scores: Dict[str, float] = {}
            feature_metrics: Dict[str, Dict[str, float]] = {}

            def importance_from(permuted_val: float) -> float:
                # Loss-like: permuted - baseline; Score-like: baseline - permuted
                return (permuted_val - base_val) if not higher_better else (base_val - permuted_val)

            for f in current_elim:
                # Map elimination feature config to its spec-group key, then to column indices
                gk = _spec_group_key_from_cfg(f)
                idxs = group_to_indices.get(gk)
                if not idxs:
                    preview = list(group_to_indices.keys())[:3]
                    raise KeyError(
                        "Could not find engineered columns for elimination feature. "
                        "Config does not match any FeatureBuildSpec from ExperimentResult.\n"
                        f"Missing group key={gk}\nAvailable (first 3): {preview}"
                    )

                fid = _feat_id(f)  # friendly identifier for logs/artifacts
                
                #Xp = _mask_feature_group(
                Xp = _permute_feature_group_by_seq_len(
                    X_eval, idxs, lengths=lengths, pad_value=self.pad_value, rng=rng
                )
                y_pred_norm = res.model.predict(Xp).to_numpy().reshape(-1)
                y_pred = y_denorm(y_pred_norm)
                m = _compute_metrics_original_scale(y_eval, y_pred)
                mv = float(m.get(self.metric_key, np.nan))
                feature_metrics[fid] = m
                feature_scores[fid] = importance_from(mv)
                self._log(f"  • Permute group {fid} (cols={idxs}): metric={mv:.1f} importance={feature_scores[fid]:+.1f}")

            if not feature_scores:
                self._log("No permutation scores computed; stopping.")
                break

            # Choose the least important feature (smallest importance)
            remove_fid = sorted(feature_scores.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
            best_removed = next(f for f in current_elim if _feat_id(f) == remove_fid)
            best_remaining = [f for f in current_elim if _feat_id(f) != remove_fid]

            # Eliminated feature (attach per-permutation metrics)
            eliminated_list: List[Dict[str, Any]] = []
            elim = json.loads(json.dumps(best_removed))
            elim_m = json.loads(json.dumps(feature_metrics.get(remove_fid, {})))
            elim["metrics"] = elim_m
            elim["permutation_importance"] = float(feature_scores[remove_fid])
            eliminated_list.append(elim)

            # Remaining features: add metrics too
            remaining_with_metrics: List[Dict[str, Any]] = []
            for f in best_remaining:
                fid = _feat_id(f)
                out = json.loads(json.dumps(f))
                out["metrics"] = json.loads(json.dumps(feature_metrics.get(fid, {})))
                out["permutation_importance"] = float(feature_scores.get(fid, np.nan))
                remaining_with_metrics.append(out)

            # Record (no overall `metrics`)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            elim_metric_val = float(elim_m.get(self.metric_key, float("nan")))
            rec = RFEIterationResult(
                timestamp_utc=ts,
                eliminated_features=eliminated_list,
                base_features=json.loads(json.dumps(self.base_features)),
                remaining_elimination_features=remaining_with_metrics,
                n_features=len(self.base_features) + len(best_remaining),
                metric_key=self.metric_key,
                baseline_metric_value=float(base_val),
                elimination_metric_value=float(elim_metric_val),
                elimination_importance=float(feature_scores[remove_fid]),
                artifact_path=res.artifact_path,
            )
            records.append(rec)

            # Persist running log of results so far
            out_path = Path(self.output_dir) / f"pi_rfe_iter_{iter_idx}_{ts}.json"
            out_path.write_text(json.dumps([r.__dict__ for r in records], indent=2), encoding="utf-8")
            
            iter_end = time.perf_counter()
            self._log("--------------")
            self._log(f"→ Iteration duration: {(iter_end - iter_start):.2f}s")
            self._log(
                f"→ Selected removal: {remove_fid} ; baseline={base_val:.1f} ; eliminated={elim_metric_val:.1f} ; importance={feature_scores[remove_fid]:.1f} ; saved={out_path.name}"
            )

            current_elim = best_remaining
            iter_idx += 1
            if len(current_elim) <= max(self.min_features, 0):
                break

        self._log("========================================")
        self._log("Permutation-Importance RFE finished")
        self._log("========================================")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        summary_path = Path(self.output_dir) / f"pi_rfe_results_{ts}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps([r.__dict__ for r in records], indent=2), encoding="utf-8")
        self._log(f"Summary artifact: {summary_path}")

        return records


# --- CLI entry point ---
# add below other helpers, above __all__
def _apply_overrides_from_args(b: "RFEPermutationBatcher", args: argparse.Namespace) -> None:
    if args.min_features is not None:
        b.min_features = int(args.min_features)
    if args.metric is not None:
        b.metric_key = args.metric.lower()
    if args.output_dir is not None:
        b.output_dir = args.output_dir
        Path(b.output_dir).mkdir(parents=True, exist_ok=True)
    if args.eval_split is not None:
        b.eval_split = args.eval_split.lower()
    if args.pad_value is not None:
        b.pad_value = float(args.pad_value)
    if args.seed is not None:
        b.permutation_seed = int(args.seed)
    if args.verbose is not None:
        b.verbose = int(args.verbose)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pi-rfe",
        description="Greedy RFE via permutation importance for (N, T, F) sequence models.",
    )
    p.add_argument("config", help="Path to experiment config (JSON/YAML) or a JSON string.")

    # Common overrides
    p.add_argument("-o", "--output-dir", help="Artifacts directory (overrides rfe.output_dir).")
    p.add_argument("--min-features", type=int, help="Stop when this many features remain.")
    p.add_argument("--metric", help="Metric key to optimize (mae, mse, rmse, mape, accuracy, auc, f1, ...).")
    p.add_argument("--eval-split", choices=["train", "val", "test"], help="Split to score on.")
    p.add_argument("--pad-value", type=float, help="Padding value for inferring lengths.")
    p.add_argument("--seed", type=int, help="Random seed for permutations.")
    p.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2], help="Verbosity (0=silent, 1=default, 2=chatty).")

    # Output controls
    outgrp = p.add_mutually_exclusive_group()
    outgrp.add_argument("--summary", dest="summary", action="store_true",
                        help="Print human-readable per-step summary (default).")
    outgrp.add_argument("--no-summary", dest="summary", action="store_false",
                        help="Do not print the human-readable summary.")
    p.set_defaults(summary=True)

    p.add_argument("--print-json", action="store_true",
                   help="Also print final results as JSON to stdout.")

    return p


def _print_console_summary(records: Sequence["RFEIterationResult"]) -> None:
    for i, r in enumerate(records, 1):
        print(f"\n=== RFE Step {i} ===")
        elim_keys = (
            [f.get("feature_key") for f in (r.eliminated_features or [])]
            if getattr(r, "eliminated_features", None)
            else None
        )
        print(f"Eliminated: {elim_keys}")
        print(f"Remaining elimination features: {[f.get('feature_key') for f in (r.remaining_elimination_features or [])]}")
        print(f"Total features: {r.n_features}")
        print(f"Metric {r.metric_key}: {r.baseline_metric_value}")
        print(f"Artifact: {r.artifact_path}")


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        # Use logger=print when verbosity > 0 to mirror your snippet
        use_logger = print if (args.verbose is None or args.verbose > 0) else None
        batcher = RFEPermutationBatcher(
            config=args.config,
            verbose=args.verbose if args.verbose is not None else 1,
            logger=use_logger,
        )
        _apply_overrides_from_args(batcher, args)

        records = batcher.run()

        if args.summary:
            _print_console_summary(records)

        if args.print_json:
            print(json.dumps([r.__dict__ for r in records], indent=2))

        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


__all__ = ["RFEIterationResult", "RFEPermutationBatcher", "cli"]

if __name__ == "__main__":
    sys.exit(cli())
    