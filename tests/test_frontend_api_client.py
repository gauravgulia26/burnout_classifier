import pytest
import requests

from src.frontend.services.api_client import BackendAPIClient, BackendAPIError


class FakeResponse:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, url, timeout, **kwargs):
        self.calls.append((method, url, timeout, kwargs))
        return self.response


def test_client_posts_the_backend_prediction_contract():
    client = BackendAPIClient("http://api.example/api/v1/")
    session = FakeSession(FakeResponse({"predictions": [{"burnout_risk": "Low"}]}))
    client._session = session

    response = client.predict({"Major_Category": "STEM"})

    assert response["predictions"][0]["burnout_risk"] == "Low"
    assert session.calls == [
        (
            "POST",
            "http://api.example/api/v1/predict",
            20.0,
            {"json": {"records": [{"Major_Category": "STEM"}]}},
        )
    ]


def test_client_adds_api_prefix_when_given_only_a_backend_host():
    client = BackendAPIClient("https://api.example")
    session = FakeSession(FakeResponse({"status": "ok"}))
    client._session = session

    client.health()

    assert session.calls[0][1] == "https://api.example/api/v1/health"


def test_client_returns_a_user_safe_error_for_connection_failures():
    client = BackendAPIClient("http://api.example")

    class BrokenSession:
        def request(self, *_args, **_kwargs):
            raise requests.ConnectionError("offline")

    client._session = BrokenSession()
    with pytest.raises(BackendAPIError, match="Could not reach"):
        client.health()
