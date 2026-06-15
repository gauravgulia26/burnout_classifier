from .features_config import FeatureMetaData

ENGINEERED_FEATURES = {
    "gpa_change": FeatureMetaData(
        feature_name="gpa_change",
        custom_flag=True,
        d_type="numerical",
        description="Difference between pre and post semester GPA",
    ),
    "ai_dependency_gap": FeatureMetaData(
        feature_name="ai_dependency_gap",
        custom_flag=True,
        d_type="numerical",
        description="Difference between perceived dependency and actual AI usage",
    ),
    "study_efficiency": FeatureMetaData(
        feature_name="study_efficiency",
        custom_flag=True,
        d_type="numerical",
        description="Academic performance per study hour",
    ),
    "ai_productivity_score": FeatureMetaData(
        feature_name="ai_productivity_score",
        custom_flag=True,
        d_type="numerical",
        description="Academic improvement per AI usage hour",
    ),
    "retention_efficiency": FeatureMetaData(
        feature_name="retention_efficiency",
        custom_flag=True,
        d_type="numerical",
        description="Retention score per AI usage hour",
    ),
    "burnout_pressure_score": FeatureMetaData(
        feature_name="burnout_pressure_score",
        custom_flag=True,
        d_type="numerical",
        description="Composite burnout pressure indicator",
    ),
    "academic_resilience": FeatureMetaData(
        feature_name="academic_resilience",
        custom_flag=True,
        d_type="numerical",
        description="Academic performance despite anxiety",
    ),
    "tool_utilization_efficiency": FeatureMetaData(
        feature_name="tool_utilization_efficiency",
        custom_flag=True,
        d_type="numerical",
        description="Retention score per tool used",
    ),
    "ai_reliance_ratio": FeatureMetaData(
        feature_name="ai_reliance_ratio",
        custom_flag=True,
        d_type="numerical",
        description="AI study ratio",
    ),
    "learning_retention_gap": FeatureMetaData(
        feature_name="learning_retention_gap",
        custom_flag=True,
        d_type="numerical",
        description="Difference between GPA and retention score",
    ),
}
