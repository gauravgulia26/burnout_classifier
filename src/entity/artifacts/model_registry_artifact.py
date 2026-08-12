from typing import Any

from pydantic import BaseModel


class ModelRegistryArtifact(BaseModel):
    """The hand-off from model registration to model serving."""

    model_name: str
    registered_model_name: str
    registered_model_version: int
    registered_model_uri: str
    run_id: str
    tracking_uri: str
    experiment_name: str
    model_signature: dict[str, Any]
    best_score: float
    test_metrics: dict[str, float]
    registered_at: str
    alias: str | None = None
    model_uri: str
    preprocessor_uri: str
    sample_data_uri: str
    local_model_path: str
