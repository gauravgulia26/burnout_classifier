"""Register the tuned best model into the MLflow Model Registry."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from mlflow.models import infer_signature
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.configs.paths import MLFLOW_ARTIFACT_PATH
from src.core.logger import get_logger
from src.entity.artifacts.model_registry_artifact import ModelRegistryArtifact
from src.entity.configs.mlflow_cfg import MLflowConfig
from src.entity.configs.model_registry_cfg import ModelRegistryConfig
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.file_utils import load_json_artifact


class ModelRegistrar:
    """Builds the preprocessing + estimator pipeline and registers it in MLflow.

    The pipeline is assembled from the preprocessor pipeline produced by the
    data-transformation component and the estimator trained by the model-tuning
    stage, so the registered model is directly usable for inference on raw
    (engineered) inputs. Held-out test metrics are computed here.
    """

    def __init__(
        self,
        model_registry_config: ModelRegistryConfig,
        tracker: MLflowTracker | None = None,
    ):
        self.config = model_registry_config
        self.tracker = tracker or MLflowTracker(
            MLflowConfig(
                tracking_uri=model_registry_config.tracking_uri,
                experiment_name=model_registry_config.experiment_name,
                tags={"pipeline_stage": "model_registry", **model_registry_config.tags},
            )
        )
        self.logger = get_logger(logger_name="ModelRegistrar")

    def __load_sample(self, processed_data_path: Path) -> pd.DataFrame:
        processed = pd.read_csv(processed_data_path)
        features = processed.drop(columns=[self.config.target_variable])
        return features.head(self.config.input_sample_size).reset_index(drop=True)

    def __evaluate_model(self, model: Any, transformation: dict) -> dict[str, float]:
        x_test = pd.read_parquet(transformation["x_test_path"])
        y_test = pd.read_parquet(transformation["y_test_path"]).squeeze("columns")
        predictions = model.predict(x_test)
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        }

    def __build_pipeline(self, model: Any, transformation: dict) -> Pipeline:
        pipeline = joblib.load(transformation["pipeline_preprocessor_path"])
        pipeline.steps.append(("classifier", model))
        return pipeline

    def run(self) -> ModelRegistryArtifact:
        tuning = load_json_artifact(self.config.tuning_artifact_path)
        feature_engineering = load_json_artifact(self.config.feature_engineer_artifact_path)
        transformation = load_json_artifact(self.config.transformation_artifact_path)
        bundle = joblib.load(self.config.model_path)

        model_name = tuning["model_name"]
        model = bundle["model"]
        test_metrics = self.__evaluate_model(model, transformation)
        pipeline = self.__build_pipeline(model, transformation)
        sample = self.__load_sample(Path(feature_engineering["processed_data_path"]))
        signature = infer_signature(sample, pipeline.predict(sample))

        tracker = self.tracker
        self.logger.info(
            f"Registering best model '{model_name}' as sklearn pipeline "
            f"{[step[0] for step in pipeline.steps]} into MLflow model registry "
            f"at {self.config.tracking_uri}"
        )

        with tracker.run("best-model-registration"):
            run_id = tracker.get_run_id()
            tracker.log_params(
                {
                    "model_name": model_name,
                    "pipeline_steps": ",".join(step[0] for step in pipeline.steps),
                    **tuning["best_parameters"],
                }
            )
            tracker.log_metrics(
                {
                    "best_tuning_score": float(tuning["best_score"]),
                    **test_metrics,
                }
            )
            tracker.log_model(pipeline, signature=signature, input_example=sample)
            tracker.log_dataset(sample, name="inference_sample", context="evaluation")

            preprocessor_path = Path(transformation["pipeline_preprocessor_path"])
            preprocessor_uri = tracker.log_artifact_file(
                preprocessor_path, artifact_path="preprocessor"
            )

            sample_path = MLFLOW_ARTIFACT_PATH / "inference_sample.parquet"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample.to_parquet(sample_path, index=False)
            sample_data_uri = tracker.log_artifact_file(
                sample_path, artifact_path="inference_sample"
            )
            model_uri = f"runs:/{run_id}/model"

        model_version = tracker.register_model(model_uri, self.config.registered_model_name)
        if self.config.alias is not None:
            tracker.set_registered_model_alias(
                self.config.registered_model_name, model_version.version, self.config.alias
            )

        self.logger.info(
            f"Registered model '{model_name}' as "
            f"models:/{self.config.registered_model_name}/{model_version.version}"
        )

        return ModelRegistryArtifact(
            model_name=model_name,
            registered_model_name=self.config.registered_model_name,
            registered_model_version=model_version.version,
            registered_model_uri=(
                f"models:/{self.config.registered_model_name}/{model_version.version}"
            ),
            run_id=run_id,
            tracking_uri=self.config.tracking_uri,
            experiment_name=self.config.experiment_name,
            model_signature=signature.to_dict(),
            best_score=float(tuning["best_score"]),
            test_metrics=test_metrics,
            registered_at=datetime.now(timezone.utc).isoformat(),
            alias=self.config.alias,
            model_uri=model_uri,
            preprocessor_uri=preprocessor_uri,
            sample_data_uri=sample_data_uri,
            local_model_path=str(self.config.model_path),
        )
