"""Register the tuned best model into the MLflow Model Registry.

The MLflow server is passed via ``--tracking-uri``, or defaults to the URI
configured in ``configs/registry_params.yaml``.
"""

import argparse
import json

from src.components.model_registry import ModelRegistrar
from src.configs.paths import (
    EXPERIMENT_YAML_DIR_PATH,
    FEATURE_ENGINEER_ARTIFACT_DIR_PATH,
    MODEL_PATH,
    MODEL_REGISTRY_ARTIFACT_DIR_PATH,
    REGISTRY_YAML_DIR_PATH,
    SCHEMA_YAML_DIR_PATH,
    TRANSFORMATION_ARTIFACT_DIR_PATH,
    TUNING_ARTIFACT_DIR_PATH,
)
from src.entity.configs.model_registry_cfg import ModelRegistryConfig
from src.utils.file_utils import dump_model_to_json, load_yaml


def main(tracking_uri: str | None = None) -> dict:
    """Register the best tuned model in MLflow and persist the registry artifact."""
    experiment_config = load_yaml(EXPERIMENT_YAML_DIR_PATH)
    registry_config = load_yaml(REGISTRY_YAML_DIR_PATH).registry
    schema_yaml = load_yaml(SCHEMA_YAML_DIR_PATH)

    tracking = experiment_config.experiment.tracking
    uri = tracking_uri or registry_config.get("tracking_uri") or tracking.tracking_uri
    experiment_name = registry_config.get("experiment_name") or tracking.experiment_name

    registrar = ModelRegistrar(
        ModelRegistryConfig(
            tracking_uri=uri,
            experiment_name=experiment_name,
            registered_model_name=registry_config.registered_model_name,
            model_path=MODEL_PATH,
            tuning_artifact_path=TUNING_ARTIFACT_DIR_PATH,
            transformation_artifact_path=TRANSFORMATION_ARTIFACT_DIR_PATH,
            feature_engineer_artifact_path=FEATURE_ENGINEER_ARTIFACT_DIR_PATH,
            target_variable=schema_yaml.target_column,
            input_sample_size=int(registry_config.get("input_sample_size", 5)),
            alias=registry_config.get("alias"),
            tags=dict(registry_config.get("tags") or {}),
        )
    )
    artifact = registrar.run()
    dump_model_to_json(artifact, MODEL_REGISTRY_ARTIFACT_DIR_PATH)
    return artifact.model_dump(mode="json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register the tuned best model into the MLflow Model Registry."
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking server URI (overrides configs/registry_params.yaml).",
    )
    args = parser.parse_args()
    print(json.dumps(main(args.tracking_uri), indent=2))
