from pathlib import Path

from pydantic import BaseModel


class DataValidationArtifact(BaseModel):
    validation_status: bool
    raw_data_path: Path
