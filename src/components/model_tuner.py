from multiprocessing import Manager
import threading
import time
from typing import Tuple

import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold

from src.core.logger import get_logger
from src.entity.artifacts.model_tuning_artifact import ModelTuningArtifact
from src.entity.configs.model_tuning_cfg import ModelTuningConfig
from src.experiment.model_factory import ModelFactory
from src.utils.file_utils import load_json_artifact


class _ProgressEstimator(BaseEstimator):
    """Delegating estimator that bumps a shared counter after every fit.

    GridSearchCV clones this wrapper for every candidate/fold and dispatches
    the clones to loky workers, so it must stay picklable. ``__sklearn_clone__``
    therefore drops the Rich progress object (not picklable) and keeps only the
    ``multiprocessing.Manager`` counter, which the parent process polls to
    render the progress bar.
    """

    def __init__(self, estimator: BaseEstimator, counter=None):
        self.estimator = estimator
        self._counter = counter

    def __sklearn_clone__(self) -> "_ProgressEstimator":
        return _ProgressEstimator(
            estimator=clone(self.estimator),
            counter=self._counter,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {"estimator": self.estimator}

    def set_params(self, **params) -> "_ProgressEstimator":
        self.estimator.set_params(**params)
        return self

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        if self._counter is not None:
            self._counter.value += 1
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y=None):
        return self.estimator.score(X, y)

    def __getattr__(self, name: str):
        return getattr(self.estimator, name)


class ModelTuner:
    def __init__(self, model_tuning_config: ModelTuningConfig):
        self.config = model_tuning_config
        self.best_estimator: BaseEstimator | None = None
        self.logger = get_logger(logger_name="ModelTunerLogger")

    def __read_artifact(self) -> Tuple[pd.DataFrame, pd.Series]:
        artifact = load_json_artifact(self.config.artifact_path)
        x_train = pd.read_parquet(artifact["x_train_path"])
        y_train = pd.read_parquet(artifact["y_train_path"]).squeeze("columns")
        return x_train, y_train

    @staticmethod
    def __sync_progress(
        progress: Progress,
        task_id: int,
        counter,
        stop_event: threading.Event,
        n_combinations: int,
        cv_folds: int,
        model_name: str,
    ) -> None:
        while not stop_event.wait(0.1):
            completed = counter.value
            candidates_done = min(completed // cv_folds, n_combinations)
            progress.update(
                task_id,
                completed=completed,
                description=(
                    f"Tuning {model_name} ({n_combinations} candidates, "
                    f"{cv_folds}-fold CV) - candidate "
                    f"{min(candidates_done + 1, n_combinations)}/{n_combinations}"
                ),
            )

    def run(self) -> ModelTuningArtifact:
        self.logger.info(f"Tuning {self.config.model_name} with GridSearchCV")
        x_train, y_train = self.__read_artifact()

        estimator = ModelFactory.create(self.config.model_name)
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        n_combinations = len(list(ParameterGrid(self.config.parameter_grid)))
        total_fits = n_combinations * self.config.cv_folds + 1

        manager = Manager()
        counter = manager.Value("i", 0)
        stop_event = threading.Event()

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} fits"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(
                f"Tuning {self.config.model_name} "
                f"({n_combinations} candidates, {self.config.cv_folds}-fold CV)",
                total=total_fits,
            )
            search = GridSearchCV(
                estimator=_ProgressEstimator(estimator, counter),
                param_grid=dict(self.config.parameter_grid),
                scoring=self.config.scoring,
                cv=cv,
                n_jobs=self.config.n_jobs,
                refit=True,
                return_train_score=False,
                error_score="raise",
                verbose=0,
            )
            syncer = threading.Thread(
                target=self.__sync_progress,
                kwargs={
                    "progress": progress,
                    "task_id": task_id,
                    "counter": counter,
                    "stop_event": stop_event,
                    "n_combinations": n_combinations,
                    "cv_folds": self.config.cv_folds,
                    "model_name": self.config.model_name,
                },
                daemon=True,
            )
            syncer.start()
            try:
                start = time.perf_counter()
                search.fit(x_train, y_train)
                end = time.perf_counter()
            finally:
                stop_event.set()
                syncer.join()
            progress.update(task_id, completed=counter.value)

        self.best_estimator = search.best_estimator_.estimator

        self.logger.info(
            f"Best {self.config.scoring} for {self.config.model_name}: "
            f"{search.best_score_:.4f} with {search.best_params_}"
        )

        return ModelTuningArtifact(
            model_name=self.config.model_name,
            best_parameters=dict(search.best_params_),
            best_score=float(search.best_score_),
            tuning_time=end - start,
        )
