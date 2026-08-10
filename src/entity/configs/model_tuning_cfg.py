from pathlib import Path
from typing import Any, List

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelTuningConfig:
    """Configuration for fine-tuning the experiment-selected best model."""

    model_name: str
    parameter_grid: dict[str, List[Any]]
    artifact_path: Path
    scoring: str
    cv_folds: int = 3
    random_state: int = 42
    n_jobs: int = -1
