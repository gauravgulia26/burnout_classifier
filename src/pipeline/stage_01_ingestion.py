from src.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig,
    handle_exceptions,
)
from src.configs.paths import RAW_DATA_PATH


@handle_exceptions
def main():
    cfg = DataIngestionConfig(raw_data_path=RAW_DATA_PATH)
    obj = DataIngestion(data_ingestion_config=cfg)
    ingestion_artifact = obj.run()
    return ingestion_artifact


if __name__ == "__main__":
    main()
