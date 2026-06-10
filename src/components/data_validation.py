from src.utils.decorators import handle_exceptions
from src.core.logger import get_logger
from src.configs.paths import SCHEMA_YAML_DIR_PATH, INGESTION_ARTIFACT_DIR_PATH
from src.validators.schema_validator import validate_schema
from src.validators import validate_path
from src.utils.file_utils import load_json_artifact, load_yaml
from src.entity.configs.data_validation_cfg import DataValidationConfig
from pathlib import Path


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        self.component_name = data_validation_config.component_name
        self.schema_path = data_validation_config.schema_path
        self.logger = get_logger(logger_name=data_validation_config.component_name)
        self.__post_init__()

    @handle_exceptions
    def __post_init__(self):
        validate_path(self.schema_path)

    @handle_exceptions
    def __read_ingestion_json(self, schema_key: str = "raw_data_path") -> Path:
        return Path(
            load_json_artifact(artifact_path=INGESTION_ARTIFACT_DIR_PATH)[schema_key]
        )

    @handle_exceptions
    def run(self):
        """
        Main Method to Start Validation

        Returns:
            int: Status Code (200)
        """
        schema_dict = load_yaml(SCHEMA_YAML_DIR_PATH)
        __df_path = self.__read_ingestion_json()
        validate_schema(df_path=__df_path, schema=schema_dict)
        self.logger.info("Data Validation Completed Successfully")
