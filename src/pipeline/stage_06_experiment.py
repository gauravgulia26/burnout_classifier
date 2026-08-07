"""Run exhaustive MLflow-tracked model selection."""

from pathlib import Path

import pandas as pd

from src.configs.paths import (
    EXPERIMENT_ARTIFACT_DIR_PATH,
    TRANSFORMATION_ARTIFACT_DIR_PATH,
)
from src.entity.configs.mlflow_cfg import MLflowConfig
from src.experiment.runner import ExperimentRunner
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.file_utils import dump_model_to_json, load_json_artifact, load_yaml
import dagshub

dagshub.init(repo_owner="grvgulia007", repo_name="burnout_classifier", mlflow=True)


def main() -> dict:
    config_path = Path(__file__).parents[2] / "configs" / "experiment_params.yaml"
    config = load_yaml(config_path)
    tracking = config.experiment.tracking
    tracker = MLflowTracker(
        MLflowConfig(
            tracking_uri=tracking.tracking_uri,
            experiment_name=tracking.experiment_name,
            tags={"pipeline_stage": "model_selection"},
        )
    )
    transformation = load_json_artifact(TRANSFORMATION_ARTIFACT_DIR_PATH)
    x_train = pd.read_parquet(transformation["x_train_path"])
    y_train = pd.read_parquet(transformation["y_train_path"]).squeeze("columns")

    with tracker.run("model-selection"):
        artifact = ExperimentRunner(config, tracker).run(x_train, y_train)
        tracker.log_params(
            {"best_model_name": artifact.best_model_name, **artifact.best_parameters}
        )

    dump_model_to_json(artifact, EXPERIMENT_ARTIFACT_DIR_PATH)
    return artifact.model_dump(mode="json")


if __name__ == "__main__":
    print(main())
