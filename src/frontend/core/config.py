"""Runtime configuration for the Streamlit application."""

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class FrontendSettings:
    backend_api_url: str

    @classmethod
    def from_environment(cls) -> "FrontendSettings":
        return cls(
            backend_api_url=os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/api/v1")
        )

