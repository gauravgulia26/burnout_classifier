"""Fine-tune the experiment-selected best model and persist the tuned bundle.

The tuning stage trains the winning model directly (GridSearchCV refits the
best candidate on the full training split), so no separate training stage is
needed. The tuned estimator is saved together with the transformation
preprocessor as the inference-ready bundle consumed by prediction and the
model registry.
"""

import json
from pathlib import Path

import dagshub
import joblib

from src.components.model_tuner import ModelTuner
from src.configs.paths import (
    EXPERIMENT_ARTIFACT_DIR_PATH,
    MODEL_PATH,
    TRANSFORMATION_ARTIFACT_DIR_PATH,
    TUNING_ARTIFACT_DIR_PATH,
)
from src.entity.configs.mlflow_cfg import MLflowConfig
from src.entity.configs.model_tuning_cfg import ModelTuningConfig
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.file_utils import dump_model_to_json, load_json_artifact, load_yaml

LABELS = {0: "Low", 1: "Medium", 2: "High"}

dagshub.init(repo_owner="grvgulia007", repo_name="burnout_classifier", mlflow=True)


def main() -> dict:
    """Run GridSearchCV on the selected best model and persist the tuning artifact."""
    config_path = Path(__file__).parents[2] / "configs" / "experiment_params.yaml"
    config = load_yaml(config_path)
    selection = load_json_artifact(EXPERIMENT_ARTIFACT_DIR_PATH)

    model_name = selection["best_model_name"]
    model_config = config.models[model_name]
    if not model_config.enabled:
        raise ValueError(f"Best model '{model_name}' is disabled in experiment_params.yaml.")

    tracking = config.experiment.tracking
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro"}
    if tracking.selection_metric not in scoring:
        raise ValueError(f"Unsupported selection metric '{tracking.selection_metric}'.")

    tuner = ModelTuner(
        ModelTuningConfig(
            model_name=model_name,
            parameter_grid=model_config.parameters,
            artifact_path=TRANSFORMATION_ARTIFACT_DIR_PATH,
            scoring=scoring[tracking.selection_metric],
            cv_folds=tracking.cv_folds,
            random_state=tracking.random_state,
        )
    )
    tuning = tuner.run()

    tracker = MLflowTracker(
        MLflowConfig(
            tracking_uri=tracking.tracking_uri,
            experiment_name=tracking.experiment_name,
            tags={"pipeline_stage": "model_tuning"},
        )
    )
    with tracker.run("best-model-tuning"):
        tracker.log_params({"model_name": model_name, **tuning.best_parameters})
        tracker.log_metrics(
            {
                "best_score": tuning.best_score,
                "tuning_seconds": tuning.tuning_time,
            }
        )
        if tracking.log_model and tuner.best_estimator is not None:
            tracker.log_model(tuner.best_estimator)

    transformation = load_json_artifact(TRANSFORMATION_ARTIFACT_DIR_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = joblib.load(transformation["pipeline_preprocessor_path"])
    preprocessor.set_output(transform="pandas")
    joblib.dump(
        {
            "model": tuner.best_estimator,
            "preprocessor": preprocessor,
            "label_map": LABELS,
        },
        MODEL_PATH,
    )
    tuning = tuning.model_copy(update={"model_path": str(MODEL_PATH)})

    dump_model_to_json(tuning, TUNING_ARTIFACT_DIR_PATH)
    return tuning.model_dump(mode="json")


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
