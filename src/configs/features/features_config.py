from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class FeatureMetaData:

    feature_name: str = field(metadata={"description": "Name of the Feature"})

    custom_flag: bool = field(
        metadata={"description": "Whether feature is custom engineered"}
    )

    description: str = field(metadata={"description": "Feature explanation"})
    d_type: str = field(
        metadata={
            "description": "Data Type of the Feature - [bool, object, numerical, categorical]"
        }
    )
