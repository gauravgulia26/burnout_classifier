from src.entity.artifacts.feature_engineer_artifact import FeatureEngineerArtifact
from src.features.make_features import FeatureEngineer


class MakeFeatures:
    def __init__(self, engine: FeatureEngineer):
        self.engine = engine

    def run(self) -> FeatureEngineerArtifact:
        obj = self.engine.transform()
        return obj
