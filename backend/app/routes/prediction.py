from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.schemas import PredictionResponse
from app.services import prediction_service
from app.services.store import store

router = APIRouter(prefix="/api", tags=["prediction"])


@router.get("/prediction/{session_id}", response_model=PredictionResponse)
async def get_prediction(
    session_id: str,
    k: int = Query(5, ge=1, le=20, description="Number of forward windows to forecast"),
) -> PredictionResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id — upload a file first.")

    try:
        result = prediction_service.predict(session, k=k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(**result)
