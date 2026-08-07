"""Command-line entry point for the reproducible burnout-risk training pipeline."""

import json

from src.pipeline.stage_01_ingestion import main as ingest
from src.pipeline.stage_02_validation import main as validate
from src.pipeline.stage_03_profiling import main as profile
from src.pipeline.stage_04_feature_engineer import main as engineer
from src.pipeline.stage_05_data_transformation import main as transform
from src.pipeline.stage_06_experiment import main as experiment
from src.pipeline.stage_06_model_training import main as train


def main() -> dict:
    """Run every pipeline stage in dependency order and return model metrics."""
    ingest()
    validate()
    profile()
    engineer()
    transform()
    experiment()
    return train()


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
