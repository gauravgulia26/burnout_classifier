from src.configs.paths import (
    VALIDATION_ARTIFACT_DIR_PATH,
    PROCESSED_DATA_DIR_PATH,
    FEATURE_ENGINEER_ARTIFACT_DIR_PATH,
)
from src.utils.file_utils import load_json_artifact
from src.components.feature_engineer import MakeFeatures
from src.features.make_features import FeatureEngineer
from src.entity.configs.feature_engineer_cfg import FeatureEngineerConfig
from src.utils.decorators import handle_exceptions


def get_config() -> FeatureEngineerConfig:
    cfg = FeatureEngineerConfig(
        raw_data_path=load_json_artifact(VALIDATION_ARTIFACT_DIR_PATH)["raw_data_path"],
        output_data_path=PROCESSED_DATA_DIR_PATH / "processed_data.csv",
        artifact_save_path=FEATURE_ENGINEER_ARTIFACT_DIR_PATH,
    )
    return cfg


@handle_exceptions
def main():

    cfg = get_config()
    obj = MakeFeatures(
        engine=FeatureEngineer(
            raw_data_path=cfg.raw_data_path,
            output_data_path=cfg.output_data_path,
            artifact_save_path=cfg.artifact_save_path,
        )
    )
    obj.run()


if __name__ == "__main__":
    main()
