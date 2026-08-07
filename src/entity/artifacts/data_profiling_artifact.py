from pydantic import BaseModel
from pathlib import Path


class DataProfilingArtifact(BaseModel):

    statistics_path: Path

    distributions_path: Path

    missing_values_path: Path
    profile_summary_path: Path

    profiling_completed: bool
