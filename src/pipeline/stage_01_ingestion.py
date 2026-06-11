from src.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig,
    handle_exceptions,
)
from src.utils.file_utils import load_yaml
from src.configs.paths import COMPONENT_YAML_DIR_PATH


@handle_exceptions
def main():
    component_config = load_yaml(yaml_path=COMPONENT_YAML_DIR_PATH)
    cfg = DataIngestionConfig(
        raw_data_path=component_config.data_ingestion.raw_data_path
    )
    obj = DataIngestion(data_ingestion_config=cfg)
    ingestion_artifact = obj.run()
    return ingestion_artifact


if __name__ == "__main__":
    main()
