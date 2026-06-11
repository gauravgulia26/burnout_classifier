import pandas as pd

from src.core.logger import get_logger
from src.configs.paths import (
    LOG_DIR_PATH,
    VALIDATION_ARTIFACT_DIR_PATH,
)
from src.entity.artifacts.data_profiling_artifact import (
    DataProfilingArtifact,
)
from src.entity.configs.data_profiling_cfg import (
    DataProfilingConfig,
)
from src.profilers.validation_profiler import ValidationProfiler
from src.utils.decorators import (
    handle_exceptions,
)
from src.utils.file_utils import (
    dump_model_to_json,
    load_json_artifact,
)


class DataProfiler:

    def __init__(
        self,
        profiling_config: DataProfilingConfig,
    ):
        self.config = profiling_config

        self.logger = get_logger(
            logger_name=self.config.component_name,
            log_dir=LOG_DIR_PATH,
        )

    def _save_artifact(
        self,
        artifact: DataProfilingArtifact,
    ) -> None:

        dump_model_to_json(
            model=artifact,
            output_path=self.config.profile_artifact_path,
        )

    @handle_exceptions
    def run(
        self,
        key: str = None,
    ) -> DataProfilingArtifact:

        self.logger.info("Data Profiling Started")

        validation_artifact = load_json_artifact(
            artifact_path=VALIDATION_ARTIFACT_DIR_PATH
        )

        df = pd.read_csv(validation_artifact[key])

        profiler = ValidationProfiler(df)

        self.logger.info("Generating Statistics Report")

        profiler.generate_statistics(output_path=self.config.statistics_path)

        self.logger.info("Generating Distribution Report")

        profiler.generate_distributions(output_path=self.config.distributions_path)

        self.logger.info("Generating Missing Values Report")

        profiler.generate_missing_values(output_path=self.config.missing_values_path)

        self.logger.info("Generating Summary Report")

        profiler.generate_summary(output_path=self.config.profile_summary_path)

        artifact = DataProfilingArtifact(
            statistics_path=self.config.statistics_path,
            distributions_path=self.config.distributions_path,
            missing_values_path=self.config.missing_values_path,
            profile_summary_path=self.config.profile_summary_path,
            profiling_completed=True,
        )

        self._save_artifact(artifact=artifact)

        self.logger.info("Data Profiling Completed")

        return artifact
