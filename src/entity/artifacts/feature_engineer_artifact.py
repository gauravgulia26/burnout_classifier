from pathlib import Path

from pydantic import BaseModel


class FeatureEngineerArtifact(BaseModel):
    processed_data_path: Path
