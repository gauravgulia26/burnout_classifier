from src.entity.configs.data_ingestion_cfg import DataIngestionConfig


class ConfigManager:
    def __init__(self):
        pass

    def get_data_ingestion_cfg(self) -> DataIngestionConfig:
        return DataIngestionConfig
