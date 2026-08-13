"""MLflow-backed model loading and prediction."""

from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import mlflow
import pandas as pd

from src.features.make_features import FeatureEngineer


class ModelNotLoadedError(RuntimeError):
    """Raised if inference is requested before the model has been loaded."""


class MLflowModelService:
    """Loads one registered MLflow model and uses it for thread-safe inference."""

    _LABELS: ClassVar[dict[int, str]] = {0: "Low", 1: "Medium", 2: "High"}

    def __init__(self, tracking_uri: str, model_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.model_uri = model_uri
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the configured registered-model URI exactly once."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                mlflow.set_tracking_uri(self.tracking_uri)
                self._model = mlflow.pyfunc.load_model(self.model_uri)

    def predict(self, raw_records: list[dict[str, object]]) -> list[str]:
        """Engineer raw records and return their decoded class labels."""
        if self._model is None:
            raise ModelNotLoadedError("The MLflow model has not been loaded.")

        raw_frame = pd.DataFrame(raw_records)
        # The registry pipeline starts at preprocessing; features are engineered
        # here so API clients only need to provide the stable raw-data contract.
        # ``engineer_dataframe`` is pure; paths are only used by the batch
        # transformation workflow, so placeholders keep this request path I/O-free.
        engineer = FeatureEngineer(Path("."), Path("."), Path("."))
        features = engineer.engineer_dataframe(raw_frame)
        predictions = self._model.predict(features)
        return [self._decode_label(value) for value in predictions]

    def _decode_label(self, value: Any) -> str:
        if isinstance(value, str) and value in self._LABELS.values():
            return value
        try:
            numeric_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Model returned an invalid prediction: {value!r}") from exc
        if numeric_value not in self._LABELS:
            raise ValueError(f"Model returned an unknown burnout-risk class: {value!r}")
        return self._LABELS[numeric_value]
