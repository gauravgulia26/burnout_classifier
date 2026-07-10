from pydantic.dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class ParameterGeneratorConfig:
    model_name: List[str]
    parameter_file_path: Path