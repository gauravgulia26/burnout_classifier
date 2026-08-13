"""Environment-based settings for the inference API."""

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings, intentionally kept independent of training configs."""

    mlflow_tracking_uri: str
    model_uri: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            mlflow_tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                "https://dagshub.com/grvgulia007/burnout_classifier.mlflow",
            ),
            model_uri=os.getenv("MLFLOW_MODEL_URI", "models:/burnout_classifier@champion"),
        )

