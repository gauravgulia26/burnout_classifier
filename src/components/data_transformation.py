from src.entity.configs import DataTransformationConfig
from src.entity.artifacts.data_transformation_artifact import DataTransformationArtifact
from src.core.logger import get_logger
from src.utils.file_utils import load_yaml, dump_model_to_json
from src.utils.decorators import handle_exceptions

from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import set_config

import pandas as pd


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.config = data_transformation_config
        self.logger = get_logger(logger_name="DataTransformation")
        self.yaml_file = load_yaml(
            data_transformation_config.transformation_schema_path
        )

    def __load_data(self) -> pd.DataFrame:
        self.logger.info("Reading Data")
        df = pd.read_csv(self.config.processed_data_path)
        return df

    def __perform_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        self.logger.info("Performing X/Y Split")
        df = self.__load_data()
        X = df.drop(columns=self.config.target_variable)
        Y = df[self.config.target_variable]
        return df, X, Y

    def __get_pipeline(self) -> Pipeline:
        self.logger.info("Generating Sklearn Pipleline")
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "nominal",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.yaml_file.nominal_features,
                ),
                (
                    "ordinal",
                    OrdinalEncoder(categories=self.yaml_file.ordinal_categories),
                    self.yaml_file.ordinal_features,
                ),
            ],
            remainder="passthrough",
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
            ]
        )

        return pipeline

    def __transform_y(self) -> pd.Series:
        self.logger.info("Transforming Y")
        df = self.__load_data()
        target_encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
        y = target_encoder.fit_transform(df[["Burnout_Risk_Level"]])
        return y

    def save_artifacts(self, transformation_artifact: DataTransformationArtifact):
        dump_model_to_json(
            model=transformation_artifact,
            output_path=self.config.transformation_artifact_save_path,
        )
        self.logger.info(
            f"Artifact Saved to {self.config.transformation_artifact_save_path}"
        )

    @handle_exceptions
    def run(self):
        set_config(display="diagram", transform_output="pandas")
        df, x, y = self.__perform_split()
        pipeline = self.__get_pipeline()
        transformed_x = pipeline.fit_transform(X=x)
        transformed_y = self.__transform_y()
        self.logger.info("Performing Train Test Split")
        x_train, x_test, y_train, y_test = train_test_split(
            transformed_x,
            transformed_y,
            test_size=self.yaml_file["test_size"],
            stratify=transformed_y,
        )
        try:
            x_train.to_parquet(self.config.x_train_save_path, index=False)
            y_train.to_parquet(self.config.y_train_save_path, index=False)
            x_test.to_parquet(self.config.x_test_save_path, index=False)
            y_test.to_parquet(self.config.y_test_save_path, index=False)
        except Exception as e:
            self.logger.error(f"Error Occured: {e}")
            raise e
        self.logger.info("Transformation Completed")
