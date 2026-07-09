from pydantic.dataclasses import dataclass
from sklearn.base import BaseEstimator
from pathlib import Path
from pydantic import ConfigDict


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelTrainerConfig:
    """Pass Model Through Model Factory"""

    model: BaseEstimator
    model_name: str
    artifact_path: Path
