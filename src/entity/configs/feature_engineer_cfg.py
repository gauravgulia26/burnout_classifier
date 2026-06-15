from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeatureEngineerConfig:
    raw_data_path: Path
    output_data_path: Path
    artifact_save_path: Path
