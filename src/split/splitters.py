# time_feature_benchmark/split/splitters.py

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


# ------------ Temporal Splitter Function ------------
@dataclass(frozen=True)
class TemporalSplitSpec:
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    shuffle_within_same_timestamps: bool = False
    seed: int = 42
    case_id_col: str = "case:concept:name"
    timestamp_col: str = "time:timestamp"

@dataclass(frozen=True)
class TemporalSplitResult:
    train_cases: List[str]
    val_cases: List[str]
    test_cases: List[str]


def temporal_split_by_case_end(
        log: pd.DataFrame,
        split_spec: TemporalSplitSpec) -> TemporalSplitResult:
    """Chronological split by each case's **last event time**.
    Ensures val/test are strictly after train in time.

    Returns lists of case_ids for train/val/test.
    """
    test_ratio = split_spec.test_ratio
    val_ratio = split_spec.val_ratio
    shuffle_within_same_timestamps = split_spec.shuffle_within_same_timestamps
    seed = split_spec.seed
    case_id_col = split_spec.case_id_col
    timestamp_col = split_spec.timestamp_col

    log = log.copy()
    log[timestamp_col] = pd.to_datetime(log[timestamp_col], utc=True, errors='coerce')

    case_end = log.groupby(case_id_col)[timestamp_col].max().rename('case_end')
    order = case_end.sort_values(kind='mergesort')  # stable sort

    n = len(order)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(0, n - n_val - n_test)

    if shuffle_within_same_timestamps:
        rng = np.random.default_rng(seed)
        # Shuffle only within groups of identical timestamps
        ends = order.values.astype("datetime64[ns]")
        uniq, inv, counts = np.unique(ends, return_inverse=True, return_counts=True)
        new_index = order.index.to_numpy(copy=True)
        start = 0
        for c in counts:
            if c > 1:
                rng.shuffle(new_index[start:start+c])
            start += c
        order = pd.Series(ends, index=new_index).sort_values(kind="mergesort")
        order = case_end.loc[order.index]  # restore case_end values to new index

    train_ids = order.index[:n_train].tolist()
    val_ids   = order.index[n_train:n_train + n_val].tolist()
    test_ids  = order.index[n_train + n_val:].tolist()

    return TemporalSplitResult(train_ids, val_ids, test_ids)



# ------------ K-Fold Splitter Function ------------
@dataclass(frozen=True)
class KFoldSplitSpec:
    k_splits: int = 5
    shuffle: bool = True
    seed: int = 42
    case_id_col: str = "case:concept:name"

@dataclass(frozen=True)
class KFoldSplitResult:
    folds: List[Tuple[List[str], List[str]]]  # (train_cases, val_cases) per fold


def kfold_split(
        log: pd.DataFrame,
        split_spec: KFoldSplitSpec) -> KFoldSplitResult:
    """K-Fold cross-validation split by case groups.
    
    Returns a list of (train_cases, val_cases) tuples for each fold.
    """
    k_splits = split_spec.k_splits
    shuffle = split_spec.shuffles
    seed = split_spec.seed
    case_id_col = split_spec.case_id_col

    if shuffle:
        rng = np.random.default_rng(seed)
        log = log.sample(frac=1, random_state=rng)

    case_ids = log[case_id_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)

    fold_size = len(case_ids) // k_splits
    folds = []
    
    for i in range(k_splits):
        start = i * fold_size
        end = None if i == k_splits - 1 else (i + 1) * fold_size
        val_cases = case_ids[start:end].tolist()
        train_cases = [c for c in case_ids if c not in val_cases]
        folds.append((train_cases, val_cases))

    return KFoldSplitResult(folds)
