"""Inference and operational endpoints."""

from fastapi import APIRouter, HTTPException, Request, status

from src.backend.schemas.prediction import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from src.backend.services.model_service import MLflowModelService, ModelNotLoadedError

router = APIRouter(tags=["inference"])


def _service(request: Request) -> MLflowModelService:
    return request.app.state.model_service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = _service(request)
    if not service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable")
    return HealthResponse(model_uri=service.model_uri)


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    service = _service(request)
    try:
        labels = service.predict([record.model_dump() for record in payload.records])
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Prediction failed") from exc

    return PredictionResponse(
        model_uri=service.model_uri,
        predictions=[PredictionResult(burnout_risk=label) for label in labels],
    )

