from src.entity.configs.model_trainer_cfg import ModelTrainerConfig
from src.core.logger import get_logger
from src.utils.file_utils import load_json_artifact
from src.configs.paths import TRANSFORMATION_ARTIFACT_DIR_PATH
from typing import Tuple
import pandas as pd
from src.entity.artifacts.model_trainer_artifact import ModelTrainerArtifact
import time


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.config = model_trainer_config
        self.model = model_trainer_config.model
        self.model_name = model_trainer_config.model_name
        self.logger = get_logger(logger_name="ModelTrainerLogger")

    def __read_artifact(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        artifact = load_json_artifact(self.config.artifact_path)
        x_train = pd.read_parquet(artifact["x_train_path"])
        y_train = pd.read_parquet(artifact["y_train_path"])
        return x_train, y_train

    def run(self) -> ModelTrainerArtifact:
        self.logger.info(f"Performing Training for {self.config.model_name}")
        x_train, y_train = self.__read_artifact()

        start = time.perf_counter()

        self.model.fit(x_train, y_train)

        end = time.perf_counter()

        return ModelTrainerArtifact(
            trained_model=self.model,
            training_time=end - start,
        )
