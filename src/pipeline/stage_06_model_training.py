"""Train and persist exactly one model selected by the experiment stage."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.components.model_trainer import ModelTrainer
from src.configs.paths import (
    EXPERIMENT_ARTIFACT_DIR_PATH,
    MODEL_ARTIFACT_DIR_PATH,
    MODEL_PATH,
    TRANSFORMATION_ARTIFACT_DIR_PATH,
)
from src.entity.configs.mlflow_cfg import MLflowConfig
from src.entity.configs.model_trainer_cfg import ModelTrainerConfig
from src.experiment.model_factory import ModelFactory
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.file_utils import load_json_artifact, load_yaml

LABELS = {0: "Low", 1: "Medium", 2: "High"}

import dagshub

dagshub.init(repo_owner="grvgulia007", repo_name="burnout_classifier", mlflow=True)


def main() -> dict:
    """Fit the selected estimator once on all training data and save the bundle."""
    config_path = Path(__file__).parents[2] / "configs" / "experiment_params.yaml"
    config = load_yaml(config_path)
    selection = load_json_artifact(EXPERIMENT_ARTIFACT_DIR_PATH)
    transformation = load_json_artifact(TRANSFORMATION_ARTIFACT_DIR_PATH)

    model_name = selection["best_model_name"]
    parameters = selection["best_parameters"]
    model = ModelFactory.create(model_name, parameters)
    trainer = ModelTrainer(
        ModelTrainerConfig(
            model=model,
            model_name=model_name,
            artifact_path=TRANSFORMATION_ARTIFACT_DIR_PATH,
        )
    )
    training = trainer.run()

    x_test = pd.read_parquet(transformation["x_test_path"])
    y_test = pd.read_parquet(transformation["y_test_path"]).squeeze("columns")
    predictions = training.trained_model.predict(x_test)
    metrics = {
        "model_name": model_name,
        "training_seconds": round(training.training_time, 3),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=[LABELS[i] for i in sorted(LABELS)],
            output_dict=True,
        ),
    }

    tracking = config.experiment.tracking
    tracker = MLflowTracker(
        MLflowConfig(
            tracking_uri=tracking.tracking_uri,
            experiment_name=tracking.experiment_name,
            tags={"pipeline_stage": "final_training"},
        )
    )
    with tracker.run("best-model-training"):
        tracker.log_params({"model_name": model_name, **parameters})
        tracker.log_metrics(
            {"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]}
        )
        if tracking.log_feature_importance:
            tracker.log_feature_importance(training.trained_model, list(x_test.columns))
        if tracking.log_model:
            tracker.log_model(training.trained_model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = joblib.load(transformation["pipeline_preprocessor_path"])
    preprocessor.set_output(transform="pandas")
    joblib.dump(
        {
            "model": training.trained_model,
            "preprocessor": preprocessor,
            "label_map": LABELS,
        },
        MODEL_PATH,
    )
    MODEL_ARTIFACT_DIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACT_DIR_PATH.write_text(
        json.dumps(
            {**metrics, "model_path": str(MODEL_PATH), "selection": selection}, indent=2
        ),
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
