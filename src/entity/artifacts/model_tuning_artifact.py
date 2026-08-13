from typing import Any

from pydantic import BaseModel


class ModelTuningArtifact(BaseModel):
    """The hand-off from hyperparameter tuning to model registration."""

    model_name: str
    best_parameters: dict[str, Any]
    best_score: float
    tuning_time: float
    # The tuner creates this intermediate object before the pipeline writes
    # the inference bundle. Stage 08 fills the path before persisting it.
    model_path: str | None = None
