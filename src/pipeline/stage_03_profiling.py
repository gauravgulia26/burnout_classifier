from src.components.data_profiling import DataProfiling, DataProfilingConfig
from src.configs.paths import (
    PROFILING_ARTIFACT_DIST_DIR_PATH,
    PROFILING_ARTIFACT_MISSING_VALUES_DIR_PATH,
    PROFILING_ARTIFACT_STATS_DIR_PATH,
    PROFILING_ARTIFACT_SUMM_DIR_PATH,
    PROFILING_ARTIFACT_DIR_PATH,
)


def main():
    cfg = DataProfilingConfig(
        component_name="DataProfiling",
        statistics_path=PROFILING_ARTIFACT_STATS_DIR_PATH,
        distributions_path=PROFILING_ARTIFACT_DIST_DIR_PATH,
        missing_values_path=PROFILING_ARTIFACT_MISSING_VALUES_DIR_PATH,
        profile_summary_path=PROFILING_ARTIFACT_SUMM_DIR_PATH,
        profile_artifact_path=PROFILING_ARTIFACT_DIR_PATH,
    )

    obj = DataProfiling(profiling_config=cfg)
    obj.run()


if __name__ == "__main__":
    main()
