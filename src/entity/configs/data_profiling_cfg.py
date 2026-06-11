from pydantic import BaseModel
from pathlib import Path


class DataProfilingConfig(BaseModel):

    component_name: str

    statistics_path: Path

    distributions_path: Path

    missing_values_path: Path

    profile_summary_path: Path

    profile_artifact_path: Path
