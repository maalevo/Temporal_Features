# time_feature_benchmark/load/loaders.py

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import os
import pandas as pd
import pm4py
import contextlib

@dataclass(frozen=True)
class LoadSpec:
    path: str
    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    timestamp_col: str = "time:timestamp"
    rename: Optional[Dict[str, str]] = None  # optional file-specific renames on load


def _pick_existing(col_candidates, columns) -> Optional[str]:
    for c in col_candidates:
        if c in columns:
            return c
    return None


def load_log_xes(spec: LoadSpec) -> pd.DataFrame:
    """
    Load a XES (.xes or .xes.gz) event log via pm4py.read_xes and return a DataFrame
    with canonical columns:
      - 'case:concept:name'
      - 'concept:name'
      - 'time:timestamp'
    Timestamps are returned as ISO8601 strings; handle UTC conversion downstream.
    """
    # suppress noisy output from pm4py.read_xes
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            raw = pm4py.read_xes(spec.path)

    # Ensure we have a DataFrame (newer pm4py returns DF by default; older might not)
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        # Best-effort fallback for older versions
        df = pm4py.convert_to_dataframe(raw)

    # Apply user-provided renames first (if any)
    if spec.rename:
        df = df.rename(columns=spec.rename)

    cols = set(df.columns)

    # Resolve column names (allow common fallbacks)
    case_col = _pick_existing(
        [spec.case_id_col, "case:concept:name", "concept:name"], cols
    )
    act_col = _pick_existing(
        [spec.activity_col, "concept:name", "activity", "Activity"], cols
    )
    ts_col = _pick_existing(
        [spec.timestamp_col, "time:timestamp", "timestamp", "@timestamp", "time"], cols
    )

    if not case_col or not act_col or not ts_col:
        missing = [n for n, v in {
            "case_id": case_col, "activity": act_col, "timestamp": ts_col
        }.items() if v is None]
        raise KeyError(f"Missing required columns in XES: {', '.join(missing)}")

    # Keep only the needed columns
    df = df[[case_col, act_col, ts_col]].rename(columns={
        case_col: "case:concept:name",
        act_col: "concept:name",
        ts_col: "time:timestamp",
    })

    # Hygiene
    df = df.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"]).reset_index(drop=True)
    df["case:concept:name"] = df["case:concept:name"].astype(str)

    # Normalize timestamps to ISO8601 strings (preserve tz if present)
    # (Do not force UTC here; do it downstream as your docstring notes.)
    ts = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=False)
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()]
    # If tz-aware -> keep offset; if naive -> no offset in string
    df.loc[:, "time:timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    # If tz-naive produced empty offset, .strftime ends without 'Z/offset'. That’s fine.

    return df[["case:concept:name", "concept:name", "time:timestamp"]]


# ---------- Dispatcher ----------
def load_events(spec: LoadSpec) -> pd.DataFrame:
    """
    Convenience loader: dispatch based on file extension.
    Supports: .xes and .xes.gz
    """
    path_lower = spec.path.lower()
    ext = os.path.splitext(path_lower)[1]
    if ext == ".xes" or path_lower.endswith(".xes.gz"):
        return load_log_xes(spec)
    raise ValueError(f"Unsupported file type for {spec.path}. Use .xes or .xes.gz")
