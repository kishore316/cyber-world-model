from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import analysis, prediction, upload
from app.schemas.schemas import HealthResponse

app = FastAPI(
    title="Cyber World Model API",
    description="Predictive cyber defence via learned network state-transition dynamics.",
    version="0.1.0",
)

# Wide open for the hackathon demo (frontend runs on a different port with
# no build step). Tighten this to specific origins before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(prediction.router)
app.include_router(analysis.router)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model_path = os.environ.get("WORLD_MODEL_PATH", "models/world_model.pt")
    return HealthResponse(
        status="ok",
        service="cyber-world-model-backend",
        model_loaded=os.path.exists(model_path),
    )
