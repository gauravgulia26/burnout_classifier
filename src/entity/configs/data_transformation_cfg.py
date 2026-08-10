from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    processed_data_path: Path
    target_variable: str
    transformation_schema_path: Path
    transformation_artifact_save_path: Path
    x_train_save_path: Path
    y_train_save_path: Path
    x_test_save_path: Path
    y_test_save_path: Path
