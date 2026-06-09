from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.configs.paths import RAW_DATA_PATH
from src.core.exception import CustomException
import sys


def main():
    try:
        cfg = DataIngestionConfig(raw_data_path=RAW_DATA_PATH)
        obj = DataIngestion(data_ingestion_config=cfg)
        ingestion_artifact = obj.run()
    except Exception as e:
        err = CustomException(error_message=e, error_detail=sys)
        obj.logger.error(f"Error Occured:{e}")
        raise err
    return ingestion_artifact


if __name__ == "__main__":
    main()
