from pydantic import BaseModel
from pathlib import Path


class FeatureEngineerArtifact(BaseModel):
    processed_data_path: Path
