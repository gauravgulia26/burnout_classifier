"""
Factory for creating machine learning model instances.
"""

from typing import Any

from src.experiment.model_registry import MODEL_REGISTRY
from sklearn.base import BaseEstimator


class ModelFactory:
    """Factory class for creating ML models."""

    @staticmethod
    def create(
        model_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> BaseEstimator:
        """
        Create a model instance.

        Args:
            model_name: Name of the model in MODEL_REGISTRY.
            parameters: Hyperparameters for the model.

        Returns:
            Instantiated sklearn estimator.

        Raises:
            ValueError: If model_name is not registered.
        """

        if model_name not in MODEL_REGISTRY:
            available_models = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Available models: {available_models}"
            )

        model_class = MODEL_REGISTRY[model_name]

        return model_class(**(parameters or {}))
