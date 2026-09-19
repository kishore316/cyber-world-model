from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.schemas import AnalysisResponse
from app.services import report_service
from app.services.store import store

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def get_analysis(session_id: str) -> AnalysisResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id — upload a file first.")

    result = report_service.analyse(session)
    return AnalysisResponse(**result)
