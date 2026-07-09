from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
    DataTransformationArtifact,
)
from src.utils.file_utils import load_json_artifact, load_yaml
from src.configs.paths import (
    FEATURE_ENGINEER_ARTIFACT_DIR_PATH,
    SCHEMA_YAML_DIR_PATH,
    TRANSFORMATION_YAML_DIR_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    TRANSFORMATION_ARTIFACT_DIR_PATH,
)
import pandas as pd


def main() -> pd.DataFrame:
    feature_json = load_json_artifact(FEATURE_ENGINEER_ARTIFACT_DIR_PATH)
    schema_yaml = load_yaml(SCHEMA_YAML_DIR_PATH)
    cfg = DataTransformationConfig(
        processed_data_path=feature_json["processed_data_path"],
        target_variable=schema_yaml["target_column"],
        transformation_schema_path=TRANSFORMATION_YAML_DIR_PATH,
        transformation_artifact_save_path=TRANSFORMATION_ARTIFACT_DIR_PATH,
        x_train_save_path=X_TRAIN_PATH,
        x_test_save_path=X_TEST_PATH,
        y_train_save_path=Y_TRAIN_PATH,
        y_test_save_path=Y_TEST_PATH,
    )
    artifact = DataTransformationArtifact(
        x_train_path=cfg.x_train_save_path,
        x_test_path=cfg.x_test_save_path,
        y_train_path=cfg.y_train_save_path,
        y_test_path=cfg.y_test_save_path,
        pipeline_preprocessor_path=cfg.processed_data_path,
    )
    obj = DataTransformation(data_transformation_config=cfg)
    results = obj.run()
    obj.save_artifacts(transformation_artifact=artifact)
    return results


if __name__ == "__main__":
    main()
