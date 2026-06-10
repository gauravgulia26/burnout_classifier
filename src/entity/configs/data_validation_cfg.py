from pathlib import Path
from pydantic import BaseModel


class DataValidationConfig(BaseModel):
    schema_path: Path
    component_name: str = "DataValidation"
