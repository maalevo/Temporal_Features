# time_feature_benchmark/model/models.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Literal
from abc import ABC, abstractmethod
import pandas as pd


# ---------------- Base model interface ----------------

class Model(ABC):
    """
    Abstract base class for models that can be trained and evaluated.
    NOTE: Concrete LSTM/XGB implementations live in lstm_models.py / xgb_models.py.
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
              y_train: pd.Series, y_val: pd.Series) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        ...


# ---------------- Spec layer (config -> model) ----------------

@dataclass(frozen=True)
class ModelSpec(ABC):
    """
    Base spec; concrete specs must implement create() and return an instance of Model.
    """
    @abstractmethod
    def create(self) -> Model:
        ...



