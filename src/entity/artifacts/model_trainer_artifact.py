from typing import Any

from pydantic import BaseModel


class ModelTrainerArtifact(BaseModel):
    training_time: int | float
    trained_model: Any
