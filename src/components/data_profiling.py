import json
import pandas as pd
import numpy as np

from src.core.logger import get_logger
from src.configs.paths import LOG_DIR_PATH, VALIDATION_ARTIFACT_DIR_PATH
from src.entity.configs.data_profiling_cfg import (
    DataProfilingConfig,
)
from src.entity.artifacts.data_profiling_artifact import (
    DataProfilingArtifact,
)
from src.utils.decorators import handle_exceptions
from src.utils.file_utils import dump_model_to_json, load_json_artifact


class DataProfiling:

    def __init__(
        self,
        profiling_config: DataProfilingConfig,
    ):
        self.config = profiling_config

        self.logger = get_logger(
            logger_name=self.config.component_name,
            log_dir=LOG_DIR_PATH,
        )

    def _generate_statistics(
        self,
        df: pd.DataFrame,
    ) -> None:

        self.logger.info("Generating Statistics Report")

        numerical_df = df.select_dtypes(include="number")

        statistics = {}

        for column in numerical_df.columns:

            statistics[column] = {
                "mean": float(numerical_df[column].mean()),
                "median": float(numerical_df[column].median()),
                "std": float(numerical_df[column].std()),
                "min": float(numerical_df[column].min()),
                "max": float(numerical_df[column].max()),
                "q25": float(numerical_df[column].quantile(0.25)),
                "q50": float(numerical_df[column].quantile(0.50)),
                "q75": float(numerical_df[column].quantile(0.75)),
                "skewness": float(numerical_df[column].skew()),
            }

        with open(
            self.config.statistics_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                statistics,
                file,
                indent=4,
            )

    def _generate_missing_values(
        self,
        df: pd.DataFrame,
    ) -> None:
        self.logger.info("Generating Missing Values Report")
        report = {}

        total_rows = len(df)

        for column in df.columns:

            missing_count = int(df[column].isna().sum())

            report[column] = {
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_count / total_rows * 100,
                    2,
                ),
            }

        with open(
            self.config.missing_values_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

    def _generate_distributions(
        self,
        df: pd.DataFrame,
    ) -> None:
        self.logger.info("Generating Distribution Report")
        report = {}

        numerical_df = df.select_dtypes(include="number")

        for column in numerical_df.columns:

            hist, bins = np.histogram(
                numerical_df[column],
                bins=10,
            )

            report[column] = {
                "histogram": hist.tolist(),
                "bins": bins.tolist(),
            }

        with open(
            self.config.distributions_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

    def _generate_summary(
        self,
        df: pd.DataFrame,
    ) -> None:
        self.logger.info("Generating Summary Report")
        summary = {
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_cells": int(df.isna().sum().sum()),
            "numerical_features": len(df.select_dtypes(include="number").columns),
            "categorical_features": len(
                df.select_dtypes(include=["object", "category"]).columns
            ),
        }

        with open(
            self.config.profile_summary_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
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
    def run(self, key="raw_data_path") -> DataProfilingArtifact:

        self.logger.info("Data Profiling Started")

        df = pd.read_csv(
            load_json_artifact(artifact_path=VALIDATION_ARTIFACT_DIR_PATH)[key]
        )

        self._generate_statistics(df)

        self._generate_distributions(df)

        self._generate_missing_values(df)

        self._generate_summary(df)

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
