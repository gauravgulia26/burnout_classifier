from dataclasses import field
from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(frozen=True, extra="forbid"))
class ModelRegistryConfig:
    """Configuration for registering the tuned best model into the MLflow Model Registry."""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    model_path: Path
    tuning_artifact_path: Path
    transformation_artifact_path: Path
    feature_engineer_artifact_path: Path
    input_sample_size: int = 5
    target_variable: str = "Burnout_Risk_Level"
    alias: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
