from pydantic import BaseModel
from pathlib import Path
from typing import List, NamedTuple


class DataIngestionArtifact(BaseModel):
    raw_data_path: Path

    num_rows: int
    num_columns: int

    feature_names: List[str]

    numerical_features: List[str]
    categorical_features: List[str]
    boolean_features: List[str]


class DataIngestionMetadataArtifact(NamedTuple):
    num_rows: int
    num_cols: int
    columns: list[str]
    numerical_features: list[str]
    boolean_features: list[str]
    categorical_features: list[str]
