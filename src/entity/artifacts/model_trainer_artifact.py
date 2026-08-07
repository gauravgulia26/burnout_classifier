from pydantic import BaseModel
from typing import Any


class ModelTrainerArtifact(BaseModel):
    training_time: int | float
    trained_model: Any
