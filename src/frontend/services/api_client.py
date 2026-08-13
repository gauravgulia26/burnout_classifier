"""Small HTTP client for the FastAPI inference service."""

from typing import Any

import requests


class BackendAPIError(RuntimeError):
    """A user-safe error returned by the backend service."""


class BackendAPIClient:
    """Client for the versioned burnout-classifier API."""

    def __init__(self, base_url: str, timeout_seconds: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json={"records": [record]})

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise BackendAPIError("Could not reach the prediction service.") from exc

        if not response.ok:
            try:
                detail = response.json().get("detail", "Request failed")
            except ValueError:
                detail = "Request failed"
            raise BackendAPIError(str(detail))

        try:
            return response.json()
        except ValueError as exc:
            raise BackendAPIError("The prediction service returned an invalid response.") from exc

