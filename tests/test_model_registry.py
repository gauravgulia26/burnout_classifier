import json
from types import SimpleNamespace

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.components.model_registry import ModelRegistrar
from src.entity.configs.model_registry_cfg import ModelRegistryConfig


class FakeTracker:
    def __init__(self, run_id="run-123"):
        self.run_id = run_id
        self.parameter_logs = []
        self.metric_logs = []
        self.logged_model = None
        self.logged_signature = None
        self.logged_input_example = None
        self.datasets = []
        self.artifact_files = []
        self.registered = None
        self.alias = None

    def run(self, _run_name, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get_run_id(self):
        return self.run_id

    def log_params(self, parameters):
        self.parameter_logs.append(parameters)

    def log_metrics(self, metrics):
        self.metric_logs.append(metrics)

    def log_model(self, model, signature=None, input_example=None):
        self.logged_model = model
        self.logged_signature = signature
        self.logged_input_example = input_example

    def log_dataset(self, data, name, context="evaluation"):
        self.datasets.append((name, context))

    def log_artifact_file(self, local_path, artifact_path):
        self.artifact_files.append((str(local_path), artifact_path))
        return f"runs:/{self.run_id}/{artifact_path}/{local_path.name}"

    def register_model(self, model_uri, name):
        self.registered = (model_uri, name)
        return SimpleNamespace(version=7)

    def set_registered_model_alias(self, name, version, alias):
        self.alias = (name, version, alias)


def make_config(tmp_path):
    sample = pd.DataFrame({"feature_a": [0.1, 0.2, 0.3], "feature_b": [1, 2, 3]})
    with_target = sample.assign(Burnout_Risk_Level=[0, 1, 0])
    with_target.to_csv(tmp_path / "processed_data.csv", index=False)

    scaler = StandardScaler().fit(sample)
    preprocessor = Pipeline([("preprocessor", scaler)])
    preprocessor_path = tmp_path / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(scaler.transform(sample), with_target["Burnout_Risk_Level"])
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump(
        {"model": model, "preprocessor": preprocessor, "label_map": {0: "Low"}},
        bundle_path,
    )

    tuning_path = tmp_path / "model_tuning.json"
    tuning_path.write_text(
        json.dumps(
            {
                "model_name": "logistic_regression",
                "best_parameters": {"C": 0.5},
                "best_score": 0.52,
                "tuning_time": 10.0,
            }
        ),
        encoding="utf-8",
    )

    training_path = tmp_path / "model_training.json"
    training_path.write_text(
        json.dumps(
            {
                "model_name": "logistic_regression",
                "accuracy": 0.51,
                "macro_f1": 0.53,
                "classification_report": {},
            }
        ),
        encoding="utf-8",
    )

    feature_engineer_path = tmp_path / "feature_engineer.json"
    feature_engineer_path.write_text(
        json.dumps({"processed_data_path": str(tmp_path / "processed_data.csv")}),
        encoding="utf-8",
    )

    transformation_path = tmp_path / "data_transformation.json"
    transformation_path.write_text(
        json.dumps({"pipeline_preprocessor_path": str(preprocessor_path)}),
        encoding="utf-8",
    )

    return ModelRegistryConfig(
        tracking_uri="http://mlflow-test:5000",
        experiment_name="test_experiment",
        registered_model_name="burnout_classifier",
        model_path=bundle_path,
        tuning_artifact_path=tuning_path,
        training_artifact_path=training_path,
        transformation_artifact_path=transformation_path,
        feature_engineer_artifact_path=feature_engineer_path,
        alias="champion",
        tags={"framework": "sklearn"},
    )


def test_registrar_registers_preprocessor_pipeline_with_signature(tmp_path):
    tracker = FakeTracker()
    registrar = ModelRegistrar(make_config(tmp_path), tracker=tracker)

    artifact = registrar.run()

    assert isinstance(tracker.logged_model, Pipeline)
    assert [step[0] for step in tracker.logged_model.steps] == [
        "preprocessor",
        "classifier",
    ]
    assert artifact.model_name == "logistic_regression"
    assert artifact.registered_model_name == "burnout_classifier"
    assert artifact.registered_model_version == 7
    assert artifact.registered_model_uri == "models:/burnout_classifier/7"
    assert artifact.run_id == "run-123"
    assert artifact.alias == "champion"
    assert artifact.best_score == 0.52
    assert artifact.test_metrics == {"accuracy": 0.51, "macro_f1": 0.53}
    assert artifact.model_uri == "runs:/run-123/model"

    assert tracker.registered == ("runs:/run-123/model", "burnout_classifier")
    assert tracker.alias == ("burnout_classifier", 7, "champion")
    assert tracker.logged_signature is not None
    assert list(tracker.logged_input_example.columns) == ["feature_a", "feature_b"]
    assert len(tracker.logged_input_example) == 3
    assert len(tracker.datasets) == 1 and tracker.datasets[0][0] == "inference_sample"
    assert len(tracker.artifact_files) == 2
    assert tracker.parameter_logs[0]["model_name"] == "logistic_regression"
    assert tracker.parameter_logs[0]["pipeline_steps"] == "preprocessor,classifier"
    assert tracker.metric_logs[0]["best_tuning_score"] == 0.52
