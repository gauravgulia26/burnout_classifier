"""Small MLflow adapter used by experiment and final-training stages."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import ModelSignature
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

    def log_model(
        self,
        model: Any,
        signature: ModelSignature | None = None,
        input_example: Any = None,
    ) -> None:
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    def log_feature_importance(self, model: Any, feature_names: list[str]) -> None:
        if not hasattr(model, "feature_importances_"):
            return
        importance = {
            name: float(value)
            for name, value in zip(feature_names, model.feature_importances_, strict=True)
        }
        mlflow.log_dict(importance, "feature_importance.json")

    def log_dataset(self, data: Any, name: str, context: str = "evaluation") -> None:
        mlflow.log_input(mlflow.data.from_pandas(data, name=name), context=context)

    def log_artifact_file(self, local_path: str | Path, artifact_path: str) -> str:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        return self.get_artifact_uri(f"{artifact_path}/{Path(local_path).name}")

    def get_artifact_uri(self, artifact_path: str) -> str:
        return mlflow.get_artifact_uri(artifact_path)

    def get_run_id(self) -> str:
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("No active MLflow run is available.")
        return active_run.info.run_id

    def register_model(self, model_uri: str, name: str) -> ModelVersion:
        return mlflow.register_model(model_uri, name)

    def set_registered_model_alias(self, name: str, version: int, alias: str) -> None:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(name, alias, version)
