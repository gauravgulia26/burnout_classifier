from src.entity.artifacts.model_tuning_artifact import ModelTuningArtifact


def test_tuning_artifact_can_be_created_before_the_model_bundle_is_written():
    tuning = ModelTuningArtifact(
        model_name="random_forest",
        best_parameters={"n_estimators": 100},
        best_score=0.52,
        tuning_time=12.0,
    )

    assert tuning.model_path is None
    persisted = tuning.model_copy(update={"model_path": "models/burnout_classifier.joblib"})
    assert persisted.model_path == "models/burnout_classifier.joblib"
