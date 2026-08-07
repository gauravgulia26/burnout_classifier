import pandas as pd
import pytest

from src.experiment.model_factory import ModelFactory
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
