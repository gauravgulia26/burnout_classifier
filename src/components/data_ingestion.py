from src.utils.decorators import handle_exceptions
from src.core.logger import get_logger
from src.configs.paths import INGESTION_ARTIFACT_DIR_PATH
from src.managers import DataIngestionConfig
from src.validators import validate_path
from src.entity.artifacts.data_ingestion_artifact import (
    DataIngestionArtifact,
    DataIngestionMetadataArtifact,
)
from src.utils.file_utils import dump_model_to_json
import sys
import pandas as pd


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.logger = get_logger(
            logger_name=data_ingestion_config.component_name,
        )

    @handle_exceptions
    def __validate(self):
        validate_path(self.data_ingestion_config.raw_data_path)

    def __save_object(self, object: DataIngestionArtifact):
        dump_model_to_json(
            model=object,
            output_path=INGESTION_ARTIFACT_DIR_PATH,
        )
        self.logger.info(f"Artifact Saved to: {INGESTION_ARTIFACT_DIR_PATH}")

    def __get_metadata(self) -> DataIngestionMetadataArtifact:
        df = pd.read_csv(self.data_ingestion_config.raw_data_path)

        return DataIngestionMetadataArtifact(
            num_rows=df.shape[0],
            num_cols=df.shape[1],
            columns=df.columns.tolist(),
            numerical_features=df.select_dtypes(
                exclude=["object", "bool"]
            ).columns.tolist(),
            boolean_features=df.select_dtypes(include="bool").columns.tolist(),
            categorical_features=df.select_dtypes(include="object").columns.tolist(),
        )

    def run(self) -> DataIngestionArtifact:
        self.logger.info("Data Ingestion Started")
        self.__validate()
        metadata = self.__get_metadata()
        obj = DataIngestionArtifact(
            raw_data_path=self.data_ingestion_config.raw_data_path,
            num_rows=metadata.num_rows,
            num_columns=metadata.num_cols,
            feature_names=metadata.columns,
            numerical_features=metadata.numerical_features,
            categorical_features=metadata.categorical_features,
            boolean_features=metadata.boolean_features,
        )
        self.__save_object(object=obj)
        return obj
