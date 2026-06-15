from src.features.make_features import FeatureEngineer
from src.entity.artifacts.feature_engineer_artifact import FeatureEngineerArtifact


class MakeFeatures:
    def __init__(self, engine: FeatureEngineer):
        self.engine = engine

    def run(self) -> FeatureEngineerArtifact:
        obj = self.engine.transform()
        return obj
