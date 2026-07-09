from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(frozen=True, extra="forbid"))
class MLflowConfig:
    """
    Configuration for MLflow tracking.
    """

    tracking_uri: str
    experiment_name: str
    run_name: str | None = None
    experiment_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
