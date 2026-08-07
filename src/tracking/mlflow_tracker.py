"""Small MLflow adapter used by experiment and final-training stages."""

from contextlib import contextmanager
from typing import Any, Iterator

import mlflow
import mlflow.sklearn

from src.entity.configs.mlflow_cfg import MLflowConfig


class MLflowTracker:
    def __init__(self, config: MLflowConfig):
        self.config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    @contextmanager
    def run(self, run_name: str, nested: bool = False) -> Iterator[None]:
        with mlflow.start_run(run_name=run_name, nested=nested, tags=self.config.tags):
            yield

    def log_params(self, parameters: dict[str, Any]) -> None:
        mlflow.log_params({key: str(value) for key, value in parameters.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_model(self, model: Any) -> None:
        mlflow.sklearn.log_model(model, artifact_path="model")

    def log_feature_importance(self, model: Any, feature_names: list[str]) -> None:
        if not hasattr(model, "feature_importances_"):
            return
        importance = {
            name: float(value)
            for name, value in zip(feature_names, model.feature_importances_, strict=True)
        }
        mlflow.log_dict(importance, "feature_importance.json")
