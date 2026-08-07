from pydantic import BaseModel
from pathlib import Path


class DataValidationArtifact(BaseModel):
    validation_status: bool
    raw_data_path: Path
