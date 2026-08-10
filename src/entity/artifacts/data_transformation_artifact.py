from pathlib import Path

from pydantic import BaseModel


class DataTransformationArtifact(BaseModel):
    x_train_path: Path
    x_test_path: Path
    y_train_path: Path
    y_test_path: Path
    pipeline_preprocessor_path: Path
