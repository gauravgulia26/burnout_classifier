import numpy as np
import pytest

from src.backend.schemas.prediction import BurnoutInput
from src.backend.services.model_service import MLflowModelService

RECORD = {
    "Major_Category": "STEM",
    "Year_of_Study": "Junior",
    "Pre_Semester_GPA": 3.2,
    "Weekly_GenAI_Hours": 5.0,
    "Primary_Use_Case": "Ideation",
    "Prompt_Engineering_Skill": "Intermediate",
    "Tool_Diversity": 3,
    "Paid_Subscription": False,
    "Traditional_Study_Hours": 12.0,
    "Perceived_AI_Dependency": 4,
    "Institutional_Policy": "Allowed_With_Citation",
    "Anxiety_Level_During_Exams": 5,
    "Post_Semester_GPA": 3.4,
    "Skill_Retention_Score": 80.0,
}


class FakeModel:
    def predict(self, frame):
        assert "gpa_change" in frame.columns
        assert len(frame) == 1
        return np.array([2])


def test_model_service_engineers_raw_records_and_decodes_prediction():
    service = MLflowModelService("http://mlflow.example", "models:/burnout_classifier@champion")
    service._model = FakeModel()

    assert service.predict([RECORD]) == ["High"]


def test_input_schema_rejects_invalid_values_and_unknown_fields():
    assert BurnoutInput.model_validate(RECORD).Tool_Diversity == 3

    with pytest.raises(ValueError):
        BurnoutInput.model_validate({**RECORD, "Tool_Diversity": 0})
    with pytest.raises(ValueError):
        BurnoutInput.model_validate({**RECORD, "unexpected": "field"})


def test_app_exposes_root_health_for_cloud_probes():
    from src.backend.core.config import Settings
    from src.backend.main import create_app

    app = create_app(Settings("http://mlflow.example", "models:/burnout_classifier@champion"))

    assert any(route.path == "/health" for route in app.routes)
