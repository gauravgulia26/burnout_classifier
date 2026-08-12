from typing import Any

from pydantic import BaseModel


class ModelTuningArtifact(BaseModel):
    """The hand-off from hyperparameter tuning to model registration."""

    model_name: str
    best_parameters: dict[str, Any]
    best_score: float
    tuning_time: float
    model_path: str
