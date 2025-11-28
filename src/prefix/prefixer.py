# time_feature_benchmark/prefix/prefixer.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass(frozen=True)
class PrefixSpec:
    min_prefix_len: int = 1
    max_prefix_len: Optional[int] = None  # if None, use full case length
    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    time_col: str = "time:timestamp"

@dataclass(frozen=True)
class PrefixResult:
    prefix_log: pd.DataFrame
    # Row-aligned targets (length == len(prefix_log))
    remaining_time: pd.Series               # seconds, name="y_remaining_seconds"
    time_until_next_event: pd.Series        # seconds, name="y_time_until_next_seconds"


def build_prefix_log(log: pd.DataFrame, prefix_spec: PrefixSpec) -> PrefixResult:
    """
    Expand an event log into all prefixes per case based on the config column names.

    Returns:
      - prefix_log: DataFrame containing all prefixes (case id = <orig>::prefix_<k>).
      - remaining_time: row-aligned Series (same index/length as prefix_log) with seconds
                        from last prefix event to end of original case.
      - time_until_next_event: row-aligned Series with seconds from last prefix event to
                               the next event in the original case (NaN for full prefixes).
    """
    min_prefix_len = prefix_spec.min_prefix_len
    max_prefix_len = prefix_spec.max_prefix_len
    case_id_col = prefix_spec.case_id_col
    activity_col = prefix_spec.activity_col
    time_col = prefix_spec.time_col

    # Validate columns
    missing = {activity_col, case_id_col, time_col} - set(log.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    # Normalize timestamps (keep tz if present; UTC conversion is downstream)
    log = log.copy()
    log[time_col] = pd.to_datetime(log[time_col], utc=False, errors="coerce")
    if log[time_col].isna().any():
        bad = log[log[time_col].isna()]
        raise ValueError(
            f"{time_col} contains non-coercible timestamps (NaT). "
            f"Example bad rows:\n{bad.head()}"
        )

    # Sort deterministically by case then time (and activity for tie-break)
    log = (
        log.sort_values([case_id_col, time_col, activity_col])
           .reset_index(drop=True)
    )

    prefix_parts = []
    rem_time_map = {}   # per-prefix (case_id) -> seconds
    next_time_map = {}  # per-prefix (case_id) -> seconds

    for orig_case, g in log.groupby(case_id_col, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue

        ts = g[time_col].to_numpy()
        case_end = ts[-1]

        min_k = max(1, int(min_prefix_len))
        max_k = n if max_prefix_len is None else max(1, min(int(max_prefix_len), n))

        for k in range(min_k, max_k + 1):
            pref_case_id = f"{orig_case}::prefix_{k}"
            pref_rows = g.iloc[:k].copy()
            pref_rows[case_id_col] = pref_case_id
            pref_rows["prefix_length"] = k
            prefix_parts.append(pref_rows)

            # Targets for this prefix (per-prefix values)
            last_ts = ts[k - 1]
            rem_time_map[pref_case_id] = (case_end - last_ts).total_seconds()
            if k < n:
                next_time_map[pref_case_id] = (ts[k] - last_ts).total_seconds()
            else:
                next_time_map[pref_case_id] = float("nan")

    if not prefix_parts:
        empty_df = log.iloc[0:0].assign(prefix_length=pd.Series(dtype="int64"))
        return PrefixResult(
            prefix_log=empty_df,
            remaining_time=pd.Series([], dtype="float64", name="y_remaining_seconds"),
            time_until_next_event=pd.Series([], dtype="float64", name="y_time_until_next_seconds"),
        )

    prefix_log = pd.concat(prefix_parts, ignore_index=True)

    # Keep original columns + prefix_length
    original_cols = list(log.columns)
    if "prefix_length" not in prefix_log.columns:
        prefix_log["prefix_length"] = pd.Series(dtype="int64")
    prefix_log = prefix_log[original_cols + ["prefix_length"]]

    # Convert per-prefix maps to Series, then broadcast to ALL ROWS via case_id
    rem_per_prefix = pd.Series(rem_time_map, dtype="float64", name="y_remaining_seconds")
    next_per_prefix = pd.Series(next_time_map, dtype="float64", name="y_time_until_next_seconds")

    remaining_time_row = prefix_log[case_id_col].map(rem_per_prefix)
    time_until_next_row = prefix_log[case_id_col].map(next_per_prefix)

    # Ensure same index/length as prefix_log
    remaining_time_row.index = prefix_log.index
    time_until_next_row.index = prefix_log.index

    return PrefixResult(
        prefix_log=prefix_log,
        remaining_time=remaining_time_row,
        time_until_next_event=time_until_next_row,
    )
