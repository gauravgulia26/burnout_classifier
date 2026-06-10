from pathlib import Path
from pydantic import BaseModel


class DataIngestionConfig(BaseModel):
    raw_data_path: Path
    component_name: str = "DataIngestion"
