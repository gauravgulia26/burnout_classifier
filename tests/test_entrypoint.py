import main
import pytest


class RunningProcess:
    returncode = None

    def poll(self):
        return self.returncode


class HealthResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_wait_for_backend_returns_only_after_a_successful_health_check(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return HealthResponse()

    monkeypatch.setattr(main, "urlopen", fake_urlopen)

    main.wait_for_backend(RunningProcess(), "9000")

    assert calls == [("http://127.0.0.1:9000/api/v1/health", 2)]


def test_wait_for_backend_stops_when_the_backend_process_exits():
    process = RunningProcess()
    process.returncode = 1

    with pytest.raises(RuntimeError, match="backend exited during startup"):
        main.wait_for_backend(process, "8000")
