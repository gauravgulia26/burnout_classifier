"""Exhaustive, MLflow-tracked model-selection runner."""

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.entity.artifacts.experiment_artifact import ExperimentArtifact
from src.experiment.model_factory import ModelFactory
from src.tracking.mlflow_tracker import MLflowTracker


class ExperimentRunner:
    """Evaluate each enabled model once with its estimator defaults."""

    def __init__(self, experiment_config: Any, tracker: MLflowTracker):
        self.config = experiment_config
        self.tracker = tracker

    def run(self, x_train: pd.DataFrame, y_train: pd.Series) -> ExperimentArtifact:
        tracking = self.config.experiment.tracking
        metric_name = tracking.selection_metric
        scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro"}
        if metric_name not in scoring:
            raise ValueError(f"Unsupported selection metric '{metric_name}'.")

        cv = StratifiedKFold(
            n_splits=tracking.cv_folds,
            shuffle=True,
            random_state=tracking.random_state,
        )
        best: tuple[float, str, dict[str, Any]] | None = None

        for model_name, model_config in self.config.models.items():
            if not model_config.enabled:
                continue
            # The configured parameter grids are intentionally ignored here.  A
            # single run per model keeps selection inexpensive while still
            # providing a fair cross-validated comparison.
            parameters: dict[str, Any] = {}
            estimator = ModelFactory.create(model_name)
            with self.tracker.run(f"{model_name}-default", nested=True):
                scores = cross_validate(
                    estimator,
                    x_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                )
                metrics = {
                    "cv_accuracy": float(scores["test_accuracy"].mean()),
                    "cv_macro_f1": float(scores["test_macro_f1"].mean()),
                    "cv_fit_seconds": float(scores["fit_time"].mean()),
                }
                # Record the actual sklearn defaults used by this run.
                self.tracker.log_params(
                    {
                        "model_name": model_name,
                        "parameter_mode": "defaults",
                        **estimator.get_params(deep=False),
                    }
                )
                self.tracker.log_metrics(metrics)

            score = metrics[f"cv_{metric_name}"]
            if best is None or score > best[0]:
                best = (score, model_name, parameters)

        if best is None:
            raise ValueError("No enabled models were configured for experimentation.")

        _score, model_name, parameters = best
        return ExperimentArtifact(
            best_model_name=model_name,
            best_parameters=parameters,
        )
