from typing import Any

from pydantic import BaseModel


class ExperimentArtifact(BaseModel):
    """The sole hand-off from model selection to final model training."""

    best_model_name: str
    best_parameters: dict[str, Any]
