# hpo_k_slices_experiment.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import itertools
import copy
import math
import time
import tempfile
import json

import numpy as np
import pandas as pd

# --- your package imports ---
from experiment.experiment import Experiment, _load_split_prefix_impl

from feature.feature_registry import default_feature_registry, FeatureRegistry
from feature.dataframe_builder import FeatureDataFrameBuilder
from feature.feature_builder import FeatureBuildSpec

# Optional joblib for caching + parallelism
try:  # pragma: no cover
    from joblib import Memory, Parallel, delayed
except Exception:  # pragma: no cover
    Memory = None
    Parallel = None
    delayed = None


# ---------------- config I/O ----------------

def _load_config(cfg):
    if isinstance(cfg, dict):
        return cfg
    return Experiment._load_config(cfg)



# ---------------- helpers ----------------

def _to_feature_build_spec(d: Dict[str, Any]) -> FeatureBuildSpec:
    return FeatureBuildSpec(
        feature_key=d["feature_key"],
        source_col_name=d.get("source_col_name"),
        encoding_key=d["encoding_key"],
        encoding_params=d.get("encoding_params"),
        target_col_name=d.get("target_col_name"),
        granularity_key=d.get("granularity_key"),
        granularity_params=d.get("granularity_params"),
        case_source=d.get("case_source"),
    )

def _spec_signature(d: Dict[str, Any]) -> Tuple:
    """Unique ID for a feature group (so we can keep/drop groups as units)."""
    return (
        d.get("feature_key"),
        d.get("source_col_name"),
        d.get("encoding_key"),
        _freeze(d.get("encoding_params")),
        d.get("granularity_key"),
        _freeze(d.get("granularity_params")),
        d.get("target_col_name"),
        d.get("case_source"),
    )

def _freeze(x: Any):
    if x is None:
        return None
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(v) for v in x)
    return x

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    eps = 1e-8
    nz = np.abs(y_true) > eps
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0) if nz.any() else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


# -------------- result container --------------

@dataclass
class KSlicesResult:
    best_params: Dict[str, Any]               # candidate-only params
    leaderboard: pd.DataFrame                 # per-candidate aggregated scores
    slice_feature_sets: List[Dict[str, Any]]  # kept feature groups per slice
    updated_experiment_config: Dict[str, Any] # same as input but with model.params filled


# -------------- main HPO runner --------------

class KSlicesHPO:
    """
    K-slices HPO that evaluates candidates by constructing a concrete Experiment config
    and calling Experiment.run(), while leveraging caches configured in the experiment cfg.

    Your config can include:
      - output_dir: where the leaderboard JSON will be written
      - load_split_prefix_cache / feature_cache / load_split_prefix_feat_trf_cache sections
    """

    def __init__(
        self,
        hpo_config: Union[str, Path, Dict[str, Any]],
        *,
        feature_registry: Optional[FeatureRegistry] = None,
        use_caches: bool = True,
        # Fallback dirs (used only if cfg doesn’t specify and joblib is available)
        cache_dir_lsp: Optional[str] = ".cache_lsp",
        cache_dir_feat: Optional[str] = ".cache_feat",
        cache_dir_lspft: Optional[str] = ".cache_lspft",
        verbose: int = 1,
        logger: Optional[Any] = None,
        n_jobs: int = 1,  # <—— NEW: parallelize across candidates when >1
    ):
        self.top_cfg = _load_config(hpo_config)
        self.verbose = int(verbose)
        self._logger = logger
        self.feature_registry = feature_registry or default_feature_registry()
        self.use_caches = use_caches
        self.n_jobs = int(n_jobs)

        # resolve the single experiment block
        if "experiments" in self.top_cfg:
            if not self.top_cfg["experiments"]:
                raise ValueError("Config has empty 'experiments' list.")
            self.cfg = self.top_cfg["experiments"][0]
        else:
            self.cfg = self.top_cfg

        # output dir from config, may be None
        self.output_dir_cfg = self.cfg.get("output_dir")

        # cache dirs: prefer those in cfg; otherwise fall back to args
        self.cache_dir_lsp = self._resolve_cache_dir(self.cfg.get("load_split_prefix_cache"), cache_dir_lsp, "lspc")
        self.cache_dir_feat = self._resolve_cache_dir(self.cfg.get("feature_cache"), cache_dir_feat, "fc")
        self.cache_dir_lspft = self._resolve_cache_dir(self.cfg.get("load_split_prefix_feat_trf_cache"), cache_dir_lspft, "lspftc")

        # Memory handles for ranking step (feature build + 1–3 composite)
        self._mem_lsp = None
        self._mem_feat = None
        if self.use_caches and Memory is not None:
            self._mem_lsp = Memory(str(self.cache_dir_lsp), verbose=0) if self.cache_dir_lsp else None
            self._mem_feat = Memory(str(self.cache_dir_feat), verbose=0) if self.cache_dir_feat else None

    # ---------- public API ----------

    def run(self) -> KSlicesResult:
        import random

        t0 = time.perf_counter()
        self._log("K-slices HPO (candidate-parallel) starting…")

        # --- 0) Read config sections we need ---
        base_feats_cfg = self.cfg.get("base_features", []) or []
        elim_feats_cfg = self.cfg.get("elimination_features", []) or []
        if not elim_feats_cfg:
            self._log("⚠️ No elimination_features provided; using only base_features in all slices.")
        elim_sigs = [_spec_signature(d) for d in elim_feats_cfg]
        n_groups = len(elim_sigs)

        hpo = self.cfg.get("hpo_params", {}) or {}
        ratios: List[float] = hpo.get("slices", [1.0, 0.6, 0.3])
        metric_key = (hpo.get("metric") or "mae").lower()
        aggregate = (hpo.get("aggregate") or "mean").lower()

        # Random slice generator (replicas per non-1.0 ratio)
        REPLICAS = 1  # tweak here if you want more/less replicas
        rng = random.Random(int(self.cfg.get("seed", 42)))

        # --- 1) Build random slices (always keep base_features) ---
        slice_feature_sets: List[Dict[str, Any]] = []
        for r in ratios:
            if r >= 0.999 or n_groups == 0:
                kept = elim_sigs
                kept_set = set(kept)
                kept_feat_defs = copy.deepcopy(base_feats_cfg) + [
                    d for d in elim_feats_cfg if _spec_signature(d) in kept_set
                ]
                slice_feature_sets.append({
                    "ratio": r,
                    "replica": 0,
                    "kept_group_count": len(kept),
                    "features_def": kept_feat_defs,
                })
                self._log(
                    f"   [slices] ratio={r:.2f}, rep=0 → kept all {len(kept)} elim groups; "
                    f"total feature defs={len(kept_feat_defs)}"
                )
            else:
                k = max(1, int(math.ceil(r * n_groups)))
                for rep in range(REPLICAS):
                    kept = rng.sample(elim_sigs, k)
                    kept_set = set(kept)
                    kept_feat_defs = copy.deepcopy(base_feats_cfg) + [
                        d for d in elim_feats_cfg if _spec_signature(d) in kept_set
                    ]
                    fp = ", ".join(map(str, kept[:3]))  # short fingerprint
                    self._log(
                        f"   [slices] ratio={r:.2f}, rep={rep} → sampled {k}/{n_groups} elim groups "
                        f"(first 3: {fp}); total feature defs={len(kept_feat_defs)}"
                    )
                    slice_feature_sets.append({
                        "ratio": r,
                        "replica": rep,
                        "kept_group_count": len(kept),
                        "features_def": kept_feat_defs,
                    })

        self._log("   Slices built: " + ", ".join(
            f"{sf['ratio']} (groups={sf['kept_group_count']}, rep={sf.get('replica',0)})"
            for sf in slice_feature_sets
        ))

        # --- 2) Build candidate grid (model.candidate_params ⨉ base_params) ---
        model_cfg = self.cfg.get("model", {}) or {}
        base_params = copy.deepcopy(model_cfg.get("base_params", {}) or {})
        cand_grid = copy.deepcopy(model_cfg.get("candidate_params", {}) or {})
        grid_keys = list(cand_grid.keys())
        grid_vals = [cand_grid[k] for k in grid_keys]
        candidates = [dict(zip(grid_keys, vals)) for vals in itertools.product(*grid_vals)]
        self._log(f"   Candidates: {len(candidates)} (keys: {grid_keys})")

        # --- 3) Evaluate candidates (parallel across candidates if n_jobs>1 & joblib available) ---
        def _serial_rows():
            rows_local = []
            for ci, cand in enumerate(candidates):
                rows_local.append(
                    self._eval_candidate(
                        ci=ci,
                        cand=cand,
                        total_candidates=len(candidates),
                        slice_feature_sets=slice_feature_sets,
                        base_params=base_params,
                        metric_key=metric_key,
                        aggregate=aggregate,
                    )
                )
            return rows_local

        if self.n_jobs != 1 and Parallel is not None and delayed is not None:
            self._log(f"   Parallelizing across candidates with n_jobs={self.n_jobs} (backend=loky).")
            rows = Parallel(n_jobs=self.n_jobs, backend="loky", prefer="processes")(
                delayed(self._eval_candidate)(
                    ci=ci,
                    cand=cand,
                    total_candidates=len(candidates),
                    slice_feature_sets=slice_feature_sets,
                    base_params=base_params,
                    metric_key=metric_key,
                    aggregate=aggregate,
                )
                for ci, cand in enumerate(candidates)
            )
        else:
            if self.n_jobs != 1 and Parallel is None:
                self._log("   joblib not available; falling back to serial execution.")
            rows = _serial_rows()

        # --- 4) Leaderboard & best params ---
        leaderboard = pd.DataFrame(rows).sort_values("aggregate", ascending=True).reset_index(drop=True)
        best_row = leaderboard.iloc[0].to_dict()
        best_params = {k: best_row[k] for k in grid_keys}
        self._log(f"Best params: {best_params} (aggregate {metric_key}={best_row['aggregate']:.4f})")

        # Write leaderboard to output dir if specified
        output_dir = self.output_dir_cfg
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            lb_path = output_path / "k_slices_hpo_leaderboard.json"
            leaderboard.to_json(lb_path, orient="records", indent=2)
            self._log(f"Leaderboard written to: {str(lb_path)}")

        # --- 5) Build final full experiment config with best params (optional, useful downstream) ---
        updated_exp_cfg = self._make_experiment_config_for_slice(
            features_def=base_feats_cfg + elim_feats_cfg,  # full feature set
            model_params=self._merge_params(base_params, best_params),
        )

        self._log(f"K-slices HPO finished in {time.perf_counter() - t0:.2f}s")

        return KSlicesResult(
            best_params=best_params,
            leaderboard=leaderboard,
            slice_feature_sets=slice_feature_sets,
            updated_experiment_config=updated_exp_cfg,
        )

    # ---------- helpers used by parallel path ----------

    def _eval_single_slice(
        self,
        sf: Dict[str, Any],
        model_params: Dict[str, Any],
        metric_key: str,
        si: Optional[int] = None,
        total_slices: Optional[int] = None,
    ) -> float:
        """Evaluate a single slice and return the requested metric."""
        slice_start = time.perf_counter()
        exp_cfg = self._make_experiment_config_for_slice(
            features_def=sf["features_def"],
            model_params=model_params,
        )
        single_cfg = exp_cfg["experiments"][0] if "experiments" in exp_cfg else exp_cfg
        exp_verbose = 1 if self.verbose == 2 else 0
        exp = Experiment(single_cfg, verbose=exp_verbose, feature_registry=self.feature_registry)
        res = exp.run()
        score = float(res.metrics.get(metric_key, math.inf))
        slice_dur = time.perf_counter() - slice_start

        # Logging (will interleave if running in parallel)
        idx = (si + 1) if si is not None else "?"
        tot = total_slices if total_slices is not None else "?"
        self._log(
            f"   Slice {idx}/{tot} (ratio={sf.get('ratio')}, rep={sf.get('replica',0)}) "
            f"→ {metric_key}={score:.4f} (took {slice_dur:.2f}s)"
        )
        return score

    def _eval_candidate(
        self,
        *,
        ci: int,
        cand: Dict[str, Any],
        total_candidates: int,
        slice_feature_sets: List[Dict[str, Any]],
        base_params: Dict[str, Any],
        metric_key: str,
        aggregate: str,
    ) -> Dict[str, Any]:
        """Evaluate all slices for a single candidate and return the leaderboard row."""
        cand_start = time.perf_counter()
        self._log("========================================")
        self._log(f"▶ Starting candidate {ci+1}/{total_candidates} with params: {cand}")
        self._log("========================================")

        cand_params = self._merge_params(base_params, cand)
        per_slice_scores: List[float] = []
        for si, sf in enumerate(slice_feature_sets):
            per_slice_scores.append(
                self._eval_single_slice(
                    sf, cand_params, metric_key, si=si, total_slices=len(slice_feature_sets)
                )
            )

        # Aggregate candidate score
        if aggregate == "avg_rank":
            agg = float(pd.Series(per_slice_scores).rank(method="average").mean())
        elif aggregate == "minimax":
            agg = float(max(per_slice_scores))
        else:
            agg = float(np.mean(per_slice_scores))

        cand_dur = time.perf_counter() - cand_start
        
        self._log("--------------")
        self._log(
            f"✔ Finished candidate {ci+1}/{total_candidates} "
            f"(aggregate {metric_key}={agg:.4f}, duration {cand_dur:.2f}s)"
        )
        
        row = {"candidate_idx": ci, **cand, "aggregate": agg}
        for i, s in enumerate(per_slice_scores):
            row[f"slice_{i}_{metric_key}"] = s
        return row

    # ---------- internals ----------

    def _load_split_prefix_once(self) -> Dict[str, Any]:
        """
        Run stages 1–3 (Load→Split→Prefix) once, reusing the same cache directory
        that Experiment will use if configured.
        """
        ds = self.cfg["dataset"]
        split = self.cfg.get("split", {}) or {}
        prefix = self.cfg.get("prefix", {}) or {}

        # If caches enabled and joblib available, wrap the composite with Memory
        if self.use_caches and Memory is not None and self.cache_dir_lsp:
            exp_verbose = 1 if self.verbose == 2 else 0
            mem = self._mem_lsp or Memory(str(self.cache_dir_lsp), verbose=exp_verbose)
            cached_lsp = mem.cache(_load_split_prefix_impl)
            return cached_lsp(ds, split, prefix, feature_registry=self.feature_registry)
        # Fallback: direct call (uncached)
        return _load_split_prefix_impl(ds, split, prefix, feature_registry=self.feature_registry)

    def _targets_from_prefix(self, pref_tr_out) -> pd.Series:
        """
        Return the training targets from the prefix stage output, robust to different
        prefix implementations:
        - Preferred: attributes on the prefix object (remaining_time / time_until_next_event)
        - Fallbacks: look inside `prefix_log` DataFrame (both attribute and dict access)
        - Final fallback: look directly on `pref_tr_out` if it is already a DataFrame
        """
        target_key = (self.cfg.get("target") or "remaining_time").lower()

        # helper to try multiple access paths
        def _get_series(obj, name: str) -> Optional[pd.Series]:
            # 1) object attribute (preferred, matches Experiment.TargetStage)
            if hasattr(obj, name):
                s = getattr(obj, name)
                if isinstance(s, (pd.Series, np.ndarray, list)):
                    return pd.Series(s)
            # 2) inside prefix_log as attribute
            if hasattr(obj, "prefix_log"):
                pl = getattr(obj, "prefix_log")
                if isinstance(pl, pd.DataFrame):
                    if hasattr(pl, name):
                        return getattr(pl, name)
                    if name in pl.columns:
                        return pl[name]
            # 3) if the object itself is a DataFrame
            if isinstance(obj, pd.DataFrame):
                if hasattr(obj, name):
                    return getattr(obj, name)
                if name in obj.columns:
                    return obj[name]
            return None

        if target_key == "remaining_time":
            s = _get_series(pref_tr_out, "remaining_time")
            if s is not None:
                return s.astype(float)
            raise AttributeError("Could not find 'remaining_time' on prefix output (tr).")

        if target_key == "time_until_next_event":
            s = _get_series(pref_tr_out, "time_until_next_event")
            if s is not None:
                return s.astype(float)
            raise AttributeError("Could not find 'time_until_next_event' on prefix output (tr).")

        raise ValueError("target must be 'remaining_time' or 'time_until_next_event'")

    def _merge_params(self, base_params: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(base_params)
        out.update(cand)
        return out

    def _make_experiment_config_for_slice(
        self, *,
        features_def: List[Dict[str, Any]],
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds a single-experiment config dict that Experiment can consume directly.
        Respects cache sections already present in self.cfg. If missing and
        caches are enabled, fills sensible joblib defaults.
        """
        # ensure a single 'experiments' list wrapping this item
        base_exp = copy.deepcopy(self.cfg)
        out = {"experiments": [base_exp]}
        exp = out["experiments"][0]

        # plug features + model params
        exp["features"] = features_def
        exp.setdefault("model", {})
        exp["model"]["type"] = exp["model"].get("type", "lstm")
        exp["model"]["params"] = model_params

        # preserve existing cache sections; fill if absent
        if self.use_caches:
            if not exp.get("load_split_prefix_cache"):
                exp["load_split_prefix_cache"] = {
                    "enabled": True,
                    "backend": "joblib",
                    "dir": str(self.cache_dir_lsp),
                }
            if not exp.get("feature_cache"):
                exp["feature_cache"] = {
                    "enabled": True,
                    "backend": "joblib",
                    "dir": str(self.cache_dir_feat),
                }
            if not exp.get("load_split_prefix_feat_trf_cache"):
                exp["load_split_prefix_feat_trf_cache"] = {
                    "enabled": True,
                    "backend": "joblib",
                    "dir": str(self.cache_dir_lspft),
                }

        # avoid artifact spam during HPO
        exp["output"] = {"path": None}

        # keep the seed consistent
        exp["seed"] = int(self.cfg.get("seed", 42))

        # optional name
        exp["name"] = "hpo_eval_slice"

        return out

    def _resolve_cache_dir(self, section: Optional[Dict[str, Any]], fallback: Optional[str], tmp_suffix: str) -> Optional[Path]:
        """
        Decide which cache directory to use:
          - If section exists and enabled with backend 'joblib', use its 'dir' if provided,
            otherwise create a temp-based default so all runs share it.
          - If section missing, use fallback path string.
          - If joblib not available or caching disabled, return None.
        """
        if not self.use_caches or Memory is None:
            return None
        if section and (section.get("enabled") and (section.get("backend", "joblib").lower() == "joblib")):
            d = section.get("dir")
            if d:
                return Path(d)
            # default temp dir if none provided
            return Path(tempfile.gettempdir()) / tmp_suffix
        # fallback
        return Path(fallback) if fallback else None

    def _log(self, msg: str) -> None:
        if self.verbose:
            s = f"[HPO]\t{msg}"
            if getattr(self, "_logger", None):
                try:
                    self._logger(s)
                except Exception:
                    print(s)
            else:
                print(s)

# -------------- convenience function --------------

def run_k_slices_hpo(
    hpo_config: Union[str, Path, Dict[str, Any]],
    verbose: int = 1,
    logger=None,
    n_jobs: int = 1,  # <—— NEW
) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        "best_params": {...},
        "leaderboard": <pd.DataFrame>,
        "slice_feature_sets": [...],
        "updated_experiment_config": <dict>,
      }
    """
    runner = KSlicesHPO(hpo_config, verbose=verbose, logger=logger, n_jobs=n_jobs)
    res = runner.run()
    return dict(
        best_params=res.best_params,
        leaderboard=res.leaderboard,
        slice_feature_sets=res.slice_feature_sets,
        updated_experiment_config=res.updated_experiment_config,
    )

# -------------- command-line interface --------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run K-slices HPO")
    parser.add_argument("config", type=str, help="Path to HPO config JSON/YAML file")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs across candidates")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    results = run_k_slices_hpo(args.config, verbose=args.verbose, n_jobs=args.n_jobs)

    

    # Pretty print results
    print("Best params:", results["best_params"])
    print("\nLeaderboard:")
    print(results["leaderboard"])