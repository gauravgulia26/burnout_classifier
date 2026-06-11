from src.utils.get_root import get_proj_root

PROJ_ROOT = get_proj_root()
DATA_DIR = "data"
RAW_DATA_DIR = "raw"
INTERIM_DATA_DIR = "interim"
PROCESSED_DATA_DIR = "processed"
EXTERNAL_DATA_DIR = "external"

MODELS_DIR = "models"

REPORTS_DIR = "reports"
FIGURES_DIR = "figures"

RAW_DATA_NAME = "raw_data.csv"

# Artifacts Configs
ARTIFACT_DIR = "artifacts"

INGESTION_ARTIFACT_DIR = "ingestion_artifact"
INGESTION_ARTIFACT = "data_ingestion.json"

VALIDATION_ARTIFACT_DIR = "validation_artifact"
VALIDATION_ARTIFACT = "data_validation.json"

PROFILING_ARTIFACT_DIR = "profiling_artifacts"
PROFILING_STATS_ARTIFACT = "data_profiling_stats.json"
PROFILING_SUMMARY_ARTIFACT = "data_profiling_summary.json"
PROFILING_MISSING_VALUES_ARTIFACT = "data_profiling_missing_val.json"
PROFILING_DISTRIBUTION_ARTIFACT = "data_profiling_dist.json"

LOG_DIR = "logs"

YAML_DIR_NAME = "configs"
SCHEMA_YAML = "schema.yaml"
COMPONENT_YAML = "components_params.yaml"
