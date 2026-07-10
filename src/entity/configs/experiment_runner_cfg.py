from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from pathlib import Path
from typing import Dict
from sklearn.model_selection import ParameterGrid
from rich.traceback import install

install()
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ExperimentRunnerConfig:
    parameter_grid: Dict[str, ParameterGrid]
    parameter_yaml_path: Path
    artifact_path: Path
