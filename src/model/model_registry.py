# time_feature_benchmark/model/model_registry.py
from __future__ import annotations

from typing import Dict, Type, Iterable, Tuple, Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Model, ModelSpec
    from .transformer import Transformer

NameOrModel = Union[str, Type["Model"]]
NameOrTransformer = Union[str, Type["Transformer"]]
NameOrSpec = Union[str, Type["ModelSpec"]]

Pair = Tuple[str, str]  # (model_key, transformer_key) or (model_key, spec_key)


class ModelRegistry:
    """
    Registry for:
      - Models        (key -> Model subclass)
      - Transformers  (key -> Transformer subclass)
      - ModelSpecs    (key -> ModelSpec dataclass)

    Constraints:
      - (model, transformer) must be explicitly allowed
      - (model, spec) must be explicitly allowed
    """

    def __init__(self):
        self._models: Dict[str, Type["Model"]] = {}
        self._transformers: Dict[str, Type["Transformer"]] = {}
        self._specs: Dict[str, Type["ModelSpec"]] = {}

        # reverse maps for class->name normalization
        self._model_name_by_cls: Dict[Type["Model"], str] = {}
        self._transformer_name_by_cls: Dict[Type["Transformer"], str] = {}
        self._spec_name_by_cls: Dict[Type["ModelSpec"], str] = {}

        # allowed combinations
        self._allowed_model_x_transformer: set[Pair] = set()
        self._allowed_model_x_spec: set[Pair] = set()

    # ---------- Models ----------
    def register_model(self, name: str, cls: Type["Model"], *, overwrite: bool = False) -> None:
        if not overwrite and name in self._models:
            raise ValueError(f"Model '{name}' already registered.")
        self._models[name] = cls
        self._model_name_by_cls[cls] = name

    def get_model(self, name: str) -> Type["Model"]:
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"Unknown model '{name}'. Registered models: {sorted(self._models)}")

    def model(self, name: str):
        def deco(cls: Type["Model"]):
            self.register_model(name, cls, overwrite=True)
            return cls
        return deco

    # ---------- Transformers ----------
    def register_transformer(self, name: str, cls: Type["Transformer"], *, overwrite: bool = False) -> None:
        if not overwrite and name in self._transformers:
            raise ValueError(f"Transformer '{name}' already registered.")
        self._transformers[name] = cls
        self._transformer_name_by_cls[cls] = name

    def get_transformer(self, name: str) -> Type["Transformer"]:
        try:
            return self._transformers[name]
        except KeyError:
            raise KeyError(f"Unknown transformer '{name}'. Registered transformers: {sorted(self._transformers)}")

    def transformer(self, name: str):
        def deco(cls: Type["Transformer"]):
            self.register_transformer(name, cls, overwrite=True)
            return cls
        return deco

    # ---------- Specs ----------
    def register_spec(self, name: str, cls: Type["ModelSpec"], *, overwrite: bool = False) -> None:
        if not overwrite and name in self._specs:
            raise ValueError(f"ModelSpec '{name}' already registered.")
        self._specs[name] = cls
        self._spec_name_by_cls[cls] = name

    def get_spec(self, name: str) -> Type["ModelSpec"]:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"Unknown ModelSpec '{name}'. Registered specs: {sorted(self._specs)}")

    def spec(self, name: str):
        def deco(cls: Type["ModelSpec"]):
            self.register_spec(name, cls, overwrite=True)
            return cls
        return deco

    # ---------- Constraints: (model, transformer) ----------
    def allow_model_transformer(self, model: NameOrModel, transformer: NameOrTransformer) -> None:
        m = self._normalize_model_key(model)
        t = self._normalize_transformer_key(transformer)
        self._allowed_model_x_transformer.add((m, t))

    def set_allowed_model_transformers(self, pairs: Iterable[Tuple[NameOrModel, NameOrTransformer]]) -> None:
        self._allowed_model_x_transformer.clear()
        for m, t in pairs:
            self.allow_model_transformer(m, t)

    def get_allowed_model_transformers(self) -> List[Pair]:
        return sorted(self._allowed_model_x_transformer)

    def get_transformers_for_model(self, model: NameOrModel) -> List[Pair]:
        m = self._normalize_model_key(model)
        return sorted(p for p in self._allowed_model_x_transformer if p[0] == m)

    def validate_model_transformer(
        self,
        model: NameOrModel,
        transformer: NameOrTransformer,
        *,
        raise_on_error: bool = True,
    ) -> bool:
        m = self._normalize_model_key(model)
        t = self._normalize_transformer_key(transformer)
        key = (m, t)
        if key in self._allowed_model_x_transformer:
            return True
        if raise_on_error:
            allowed = self.get_transformers_for_model(m)
            suggestions = ", ".join([f"({am},{at})" for am, at in allowed]) or "[]"
            raise ValueError(
                f"Combination not allowed: model='{m}', transformer='{t}'. "
                f"Allowed for '{m}': {suggestions}"
            )
        return False

    # ---------- Constraints: (model, spec) ----------
    def allow_model_spec(self, model: NameOrModel, spec: NameOrSpec) -> None:
        m = self._normalize_model_key(model)
        s = self._normalize_spec_key(spec)
        self._allowed_model_x_spec.add((m, s))

    def set_allowed_model_specs(self, pairs: Iterable[Tuple[NameOrModel, NameOrSpec]]) -> None:
        self._allowed_model_x_spec.clear()
        for m, s in pairs:
            self.allow_model_spec(m, s)

    def get_allowed_model_specs(self) -> List[Pair]:
        return sorted(self._allowed_model_x_spec)

    def get_specs_for_model(self, model: NameOrModel) -> List[Pair]:
        m = self._normalize_model_key(model)
        return sorted(p for p in self._allowed_model_x_spec if p[0] == m)

    def validate_model_spec(
        self,
        model: NameOrModel,
        spec: NameOrSpec,
        *,
        raise_on_error: bool = True,
    ) -> bool:
        m = self._normalize_model_key(model)
        s = self._normalize_spec_key(spec)
        key = (m, s)
        if key in self._allowed_model_x_spec:
            return True
        if raise_on_error:
            allowed = self.get_specs_for_model(m)
            suggestions = ", ".join([f"({am},{as_})" for am, as_ in allowed]) or "[]"
            raise ValueError(
                f"Combination not allowed: model='{m}', spec='{s}'. "
                f"Allowed for '{m}': {suggestions}"
            )
        return False

    # ---------- Introspection ----------
    def list_models(self) -> Dict[str, Type["Model"]]:
        return dict(self._models)

    def list_transformers(self) -> Dict[str, Type["Transformer"]]:
        return dict(self._transformers)

    def list_specs(self) -> Dict[str, Type["ModelSpec"]]:
        return dict(self._specs)

    # ---------- Helpers (name/class normalization) ----------
    def _normalize_model_key(self, obj: NameOrModel) -> str:
        if isinstance(obj, str):
            self.get_model(obj)
            return obj
        try:
            return self._model_name_by_cls[obj]  # type: ignore[index]
        except KeyError:
            raise KeyError(f"Model class {obj} is not registered; register it before using constraints.")

    def _normalize_transformer_key(self, obj: NameOrTransformer) -> str:
        if isinstance(obj, str):
            self.get_transformer(obj)
            return obj
        try:
            return self._transformer_name_by_cls[obj]  # type: ignore[index]
        except KeyError:
            raise KeyError(f"Transformer class {obj} is not registered; register it before using constraints.")

    def _normalize_spec_key(self, obj: NameOrSpec) -> str:
        if isinstance(obj, str):
            self.get_spec(obj)
            return obj
        try:
            return self._spec_name_by_cls[obj]  # type: ignore[index]
        except KeyError:
            raise KeyError(f"ModelSpec class {obj} is not registered; register it before using constraints.")

    # ---------- Optional: instantiate from config ----------
    def instantiate_from_config(self, cfg: dict) -> "Model":
        """
        cfg like:
          {'model': 'lstm', 'spec': 'lstm_default', 'params': {...}}

        We validate (model, spec) if both keys are provided, then create the model via the spec.
        """
        spec_key = cfg.get("spec")
        if not spec_key:
            raise ValueError("Config missing 'spec' key.")
        SpecCls = self.get_spec(spec_key)
        params = cfg.get("params") or {}
        spec = SpecCls(**params)

        model = spec.create()  # returns a Model instance

        # If a model key is provided, validate (model, spec) compatibility
        model_key = cfg.get("model")
        if model_key:
            self.validate_model_spec(model_key, spec_key)
            created_key = self._model_name_by_cls.get(type(model))
            if created_key and created_key != model_key:
                raise ValueError(
                    f"Spec '{spec_key}' created model '{created_key}', but config declared model '{model_key}'."
                )

        return model


def default_model_registry() -> ModelRegistry:
    """
    Create a model registry and preload built-ins + allowed pairs.
    """
    reg = ModelRegistry()

    # Models + Specs
    try:
        from .lstm_models import LSTMModel, LSTMModelSpec
        reg.register_model("lstm_mod", LSTMModel)
        reg.register_spec("lstm_spec", LSTMModelSpec)
        reg.allow_model_spec("lstm_mod", "lstm_spec")
    except Exception:
        pass

    try:
        from .xgb_models import (
            XGBModel,
            NoBucketXGBModel,
            ClusteringBucketXGBModel,
            PrefixLenBucketXGBModel,
            XGBModelSpec,
        )
        #reg.register_model("xgb_no_bucket", NoBucketXGBModel)
        #reg.register_model("xgb_cluster", ClusteringBucketXGBModel)
        #reg.register_model("xgb_prefixlen", PrefixLenBucketXGBModel)

        #reg.register_spec("xgb_plain", XGBModelSpec)
        #reg.allow_model_spec("xgb_no_bucket", "xgb_plain")
        #reg.allow_model_spec("xgb_cluster", "xgb_plain")
        #reg.allow_model_spec("xgb_prefixlen", "xgb_plain")
    except Exception:
        pass

    # Transformers
    try:
        from .transformer import LSTMTransformer
        reg.register_transformer("lstm_tra", LSTMTransformer)
    except Exception:
        pass
    # (register other transformers here)

    # Allowed model<->transformer pairs
    try:
        reg.allow_model_transformer("lstm_mod", "lstm_tra")
        # Example for future tabular transformer:
        # reg.register_transformer("tabular", TabularTransformer)
        # for m in ("xgb", "xgb_no_bucket", "xgb_cluster", "xgb_prefixlen"):
        #     reg.allow_model_transformer(m, "tabular")
    except Exception:
        pass

    return reg
