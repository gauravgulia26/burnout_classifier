from src.components.data_validation import DataValidation, DataValidationConfig
from src.configs.paths import INGESTION_ARTIFACT_DIR_PATH


def main():
    cfg = DataValidationConfig(schema_path=INGESTION_ARTIFACT_DIR_PATH)
    obj = DataValidation(data_validation_config=cfg)
    obj.run()


if __name__ == "__main__":
    main()
