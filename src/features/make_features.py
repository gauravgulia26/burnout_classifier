import pandas as pd

from src.entity.artifacts.feature_engineer_artifact import FeatureEngineerArtifact
from src.core.logger import get_logger
from src.utils.decorators import handle_exceptions
from src.utils.file_utils import dump_model_to_json
from pathlib import Path
from pydantic import BaseModel


class FeatureEngineer:
    """
    Creates domain-specific features for burnout prediction.
    """

    def __init__(
        self, raw_data_path: Path, output_data_path: Path, artifact_save_path: Path
    ):
        """
        Args:
            raw_data_path (Path): Posix Raw Data Path
            output_data_path (Path): Posix Output Data Path to Save Engineered df
            artifact_save_path (Path): Posix Path to Save Feature Engineer Artifact
        """
        self.raw_data_path = raw_data_path
        self.output_data_path = output_data_path
        self.logger = get_logger(logger_name="FeatureEngineer")
        self.artifact_save_path = artifact_save_path
        self.EPSILON = 1e-6

    @handle_exceptions
    def transform(self) -> FeatureEngineerArtifact:
        """
        Apply all feature engineering steps.
        """
        self.logger.info("Generating Custom Features")
        df = pd.read_csv(self.raw_data_path)
        df = df.copy()

        df = self._add_gpa_change(df)

        df = self._add_ai_dependency_gap(df)

        df = self._add_study_efficiency(df)

        df = self._add_ai_productivity_score(df)

        df = self._add_retention_efficiency(df)

        df = self._add_burnout_pressure_score(df)

        df = self._add_academic_resilience(df)

        df = self._add_tool_utilization_efficiency(df)

        df = self._add_ai_reliance_ratio(df)

        df = self._add_learning_retention_gap(df)

        self.logger.info("Saving DF")

        df.to_csv(self.output_data_path, index=False)

        self.logger.info(f"Df Saved to: {self.output_data_path}")

        obj = FeatureEngineerArtifact(processed_data_path=self.output_data_path)

        self._save_model(model=obj)
        return obj

    @handle_exceptions
    def _save_model(self, model: BaseModel):
        dump_model_to_json(model=model, output_path=self.artifact_save_path)
        self.logger.info(f"Artifact Saved At: {self.artifact_save_path}")

    def _add_gpa_change(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Gpa Change")

        df["gpa_change"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]

        return df

    def _add_ai_dependency_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Ai Dependency Gap")

        df["ai_dependency_gap"] = (
            df["Perceived_AI_Dependency"] - df["Weekly_GenAI_Hours"]
        )

        return df

    def _add_study_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Study Efficiency")

        df["study_efficiency"] = df["Post_Semester_GPA"] / (
            df["Traditional_Study_Hours"] + self.EPSILON
        )

        return df

    def _add_ai_productivity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Productivity Score")

        df["ai_productivity_score"] = df["gpa_change"] / (
            df["Weekly_GenAI_Hours"] + self.EPSILON
        )

        return df

    def _add_retention_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Retention Efficiency")

        df["retention_efficiency"] = df["Skill_Retention_Score"] / (
            df["Weekly_GenAI_Hours"] + self.EPSILON
        )

        return df

    def _add_burnout_pressure_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Burnout Pressure Score")

        df["burnout_pressure_score"] = (
            df["Anxiety_Level_During_Exams"]
            + df["Traditional_Study_Hours"]
            + df["Perceived_AI_Dependency"]
        ) / 3

        return df

    def _add_academic_resilience(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Academic Resilience")

        df["academic_resilience"] = df["Post_Semester_GPA"] / (
            df["Anxiety_Level_During_Exams"] + self.EPSILON
        )

        return df

    def _add_tool_utilization_efficiency(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.logger.info("Generating Tool Utilization Efficiency")
        df["tool_utilization_efficiency"] = df["Skill_Retention_Score"] / (
            df["Tool_Diversity"] + self.EPSILON
        )

        return df

    def _add_ai_reliance_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Generating Ai Reliance Ratio")

        df["ai_reliance_ratio"] = df["Weekly_GenAI_Hours"] / (
            df["Traditional_Study_Hours"] + self.EPSILON
        )

        return df

    def _add_learning_retention_gap(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        self.logger.info("Generating Add Learning Retention Gap")
        df["learning_retention_gap"] = (
            df["Post_Semester_GPA"] - df["Skill_Retention_Score"]
        )

        return df
