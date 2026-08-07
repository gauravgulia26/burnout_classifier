"""Train, evaluate, and persist the final burnout-risk classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.components.model_trainer import ModelTrainer
from src.configs.paths import (
    MODEL_ARTIFACT_DIR_PATH,
    MODEL_PATH,
    TRANSFORMATION_ARTIFACT_DIR_PATH,
)
from src.entity.configs.model_trainer_cfg import ModelTrainerConfig
from src.experiment.model_factory import ModelFactory
from src.utils.file_utils import load_json_artifact, load_yaml

LABELS = {0: "Low", 1: "Medium", 2: "High"}


def main() -> dict:
    """Fit the configured model and write a self-contained inference bundle."""
    experiment_path = Path(__file__).parents[2] / "configs" / "experiment_params.yaml"
    experiment = load_yaml(experiment_path).experiment
    transformation = load_json_artifact(TRANSFORMATION_ARTIFACT_DIR_PATH)

    model_name = experiment.model
    model = ModelFactory.create(model_name, dict(experiment.model_parameters))
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
            y_test, predictions, target_names=[LABELS[i] for i in sorted(LABELS)], output_dict=True
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = joblib.load(transformation["pipeline_preprocessor_path"])
    preprocessor.set_output(transform="pandas")
    bundle = {
        "model": training.trained_model,
        "preprocessor": preprocessor,
        "label_map": LABELS,
    }
    joblib.dump(bundle, MODEL_PATH)
    MODEL_ARTIFACT_DIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACT_DIR_PATH.write_text(
        json.dumps({**metrics, "model_path": str(MODEL_PATH)}, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
