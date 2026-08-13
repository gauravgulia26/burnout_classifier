"""Entry point for training or serving the burnout-risk application."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def train_pipeline() -> dict:
    """Run every pipeline stage in dependency order and return registry info."""
    from src.pipeline.stage_01_ingestion import main as ingest
    from src.pipeline.stage_02_validation import main as validate
    from src.pipeline.stage_03_profiling import main as profile
    from src.pipeline.stage_04_feature_engineer import main as engineer
    from src.pipeline.stage_05_data_transformation import main as transform
    from src.pipeline.stage_06_experiment import main as experiment
    from src.pipeline.stage_08_model_tuning import main as tune
    from src.pipeline.stage_09_model_registry import main as register

    ingest()
    validate()
    profile()
    engineer()
    transform()
    experiment()
    tune()
    return register()


def wait_for_backend(backend_process: subprocess.Popen, backend_port: str) -> None:
    """Block frontend startup until FastAPI reports that its model is ready."""
    timeout_seconds = float(os.getenv("API_STARTUP_TIMEOUT_SECONDS", "180"))
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://127.0.0.1:{backend_port}/api/v1/health"
    print("Waiting for the prediction API and registered model to load...", flush=True)

    while time.monotonic() < deadline:
        if backend_process.poll() is not None:
            raise RuntimeError(
                f"The backend exited during startup with code {backend_process.returncode}."
            )
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    print("Prediction API is ready. Starting the user interface.", flush=True)
                    return
        except URLError:
            pass
        time.sleep(1)

    raise TimeoutError(f"The prediction API was not ready after {timeout_seconds:.0f} seconds.")


def serve() -> None:
    """Run the FastAPI backend and Streamlit frontend as one application unit."""
    backend_port = os.getenv("BACKEND_PORT", "8000")
    frontend_port = os.getenv("FRONTEND_PORT", "8501")
    environment = os.environ.copy()
    environment.setdefault("BACKEND_API_URL", f"http://127.0.0.1:{backend_port}/api/v1")

    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.backend.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        backend_port,
    ]
    frontend_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/frontend/app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        frontend_port,
        "--server.headless",
        "true",
    ]
    backend_process = subprocess.Popen(backend_command, env=environment)
    processes = [backend_process]
    try:
        wait_for_backend(backend_process, backend_port)
        processes.append(subprocess.Popen(frontend_command, env=environment))
    except Exception:
        if backend_process.poll() is None:
            backend_process.terminate()
            backend_process.wait(timeout=10)
        raise

    def stop_processes(_signal_number: int, _frame: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    previous_sigterm = signal.signal(signal.SIGTERM, stop_processes)
    previous_sigint = signal.signal(signal.SIGINT, stop_processes)
    try:
        while True:
            exited = next((process for process in processes if process.poll() is not None), None)
            if exited is not None:
                raise RuntimeError(f"A service exited unexpectedly with code {exited.returncode}.")
            time.sleep(0.5)
    finally:
        stop_processes(signal.SIGTERM, None)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def main() -> None:
    """Parse commands for either complete training or the integrated web application."""
    parser = argparse.ArgumentParser(description="Burnout classifier application entry point.")
    parser.add_argument(
        "command",
        choices=("serve", "train"),
        nargs="?",
        default="serve",
        help="Run the web application (default) or the complete training pipeline.",
    )
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train_pipeline(), indent=2))
        return
    serve()


if __name__ == "__main__":
    main()
