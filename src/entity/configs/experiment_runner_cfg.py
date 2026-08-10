from pathlib import Path
from typing import Dict

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from rich.traceback import install
from sklearn.model_selection import ParameterGrid

install()
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ExperimentRunnerConfig:
    parameter_grid: Dict[str, ParameterGrid]
    parameter_yaml_path: Path
    artifact_path: Path
