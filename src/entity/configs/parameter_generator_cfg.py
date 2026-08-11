from pathlib import Path
from typing import List

from pydantic.dataclasses import dataclass


@dataclass
class ParameterGeneratorConfig:
    model_name: List[str]
    parameter_file_path: Path
