from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

from src.experiment.model_factory import ModelFactory
from src.experiment.runner import ExperimentRunner
from src.features.make_features import FeatureEngineer
from src.validators.schema_validator import validate_schema


def test_schema_accepts_the_project_dataset():
    validate_schema(
        "data/raw/raw_data.csv",
        {
            "columns": {
                "Major_Category": {"type": "categorical"},
                "Pre_Semester_GPA": {"type": "numerical"},
                "Paid_Subscription": {"type": "boolean"},
            }
        },
    )


def test_feature_engineering_is_repeatable_for_inference_rows(tmp_path):
    row = pd.read_csv("data/raw/raw_data.csv", nrows=1).drop(columns="Burnout_Risk_Level")
    engineer = FeatureEngineer(
        tmp_path / "input.csv", tmp_path / "output.csv", tmp_path / "artifact.json"
    )
    result = engineer.engineer_dataframe(row)

    assert "Student_ID" not in result.columns
    assert {"gpa_change", "burnout_pressure_score", "ai_reliance_ratio"}.issubset(result.columns)
    assert result.shape[0] == 1


def test_model_factory_rejects_unknown_models():
    with pytest.raises(ValueError, match="Unsupported model"):
        ModelFactory.create("not-a-model")


def test_experiment_runner_logs_one_default_run_and_selects_one():
    class Tracker:
        def __init__(self):
            self.parameter_logs = []

        @contextmanager
        def run(self, *_args, **_kwargs):
            yield

        def log_params(self, parameters):
            self.parameter_logs.append(parameters)

        def log_metrics(self, _metrics):
            pass

    config = SimpleNamespace(
        experiment=SimpleNamespace(
            tracking=SimpleNamespace(
                selection_metric="macro_f1",
                cv_folds=2,
                random_state=42,
                experiment_name="test",
                tracking_uri="http://test",
            )
        ),
        models={
            "logistic_regression": SimpleNamespace(
                enabled=True,
                search_type="grid",
                parameters={"C": [0.1, 1.0], "max_iter": [100], "solver": ["lbfgs"]},
            )
        },
    )
    x = pd.DataFrame({"signal": [0, 0, 0, 1, 1, 1, 0, 1], "noise": range(8)})
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    tracker = Tracker()

    artifact = ExperimentRunner(config, tracker).run(x, y)

    assert len(tracker.parameter_logs) == 1
    assert artifact.best_model_name == "logistic_regression"
    assert artifact.best_parameters == {}
    assert tracker.parameter_logs[0]["parameter_mode"] == "defaults"
