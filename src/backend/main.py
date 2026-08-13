"""FastAPI application entry point for registered-model inference."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.backend.api.routes.inference import router as inference_router
from src.backend.core.config import Settings
from src.backend.services.model_service import MLflowModelService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API and load its configured MLflow registered model on startup."""
    runtime_settings = settings or Settings.from_environment()
    model_service = MLflowModelService(
        tracking_uri=runtime_settings.mlflow_tracking_uri,
        model_uri=runtime_settings.model_uri,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model_service.load()
        yield

    app = FastAPI(
        title="Burnout Classifier API",
        version="1.0.0",
        description="MLflow-registered model inference for student burnout risk.",
        lifespan=lifespan,
    )
    app.state.model_service = model_service
    app.include_router(inference_router, prefix="/api/v1")
    return app


app = create_app()
