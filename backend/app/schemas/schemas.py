"""
Pydantic request/response models for the Cyber World Model API.

Kept deliberately flat and explicit (rather than deeply nested) so the
frontend can consume responses with plain `data.field` access and so the
ml/ world-model can be dropped in later without changing this contract:
`prediction_service.py` promises the exact same response shape whether it
is running the transparent heuristic scorer or a trained LSTM rollout.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# /api/upload
# --------------------------------------------------------------------------

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    source_type: str = Field(description="'csv' or 'pcap'")
    flow_count: int
    packet_count: int
    window_count: int
    window_seconds: int
    duration_seconds: float
    warnings: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# /api/prediction/{session_id}
# --------------------------------------------------------------------------

class TimelinePoint(BaseModel):
    window_index: int
    timestamp: float
    label: str
    probability: float
    stage: str
    is_forecast: bool


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float = Field(description="Signed: positive pushes risk up, negative pushes it down")
    direction: str = Field(description="'up' or 'down'")


class FlaggedFlow(BaseModel):
    flow_id: str
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    window_index: int
    risk_score: float
    reason: str


class BaselineComparison(BaseModel):
    metric: str
    world_model: float
    baseline: float


class PredictionResponse(BaseModel):
    session_id: str
    mode: str = Field(description="'heuristic' or 'model' depending on whether a trained checkpoint is loaded")
    current_probability: float
    current_stage: str
    timeline: List[TimelinePoint]
    top_drivers: List[FeatureContribution]
    flagged_flows: List[FlaggedFlow]
    baseline_comparison: List[BaselineComparison]


# --------------------------------------------------------------------------
# /api/analysis/{session_id}
# --------------------------------------------------------------------------

class TopTalker(BaseModel):
    ip: str
    flow_count: int
    total_bytes: float


class TopPort(BaseModel):
    port: int
    flow_count: int


class ProtocolShare(BaseModel):
    protocol: str
    flow_count: int
    percentage: float


class StageShare(BaseModel):
    stage: str
    window_count: int
    percentage: float


class AnalysisResponse(BaseModel):
    session_id: str
    total_flows: int
    total_packets: int
    total_windows: int
    unique_ip_pairs: int
    top_talkers: List[TopTalker]
    top_ports: List[TopPort]
    protocol_breakdown: List[ProtocolShare]
    stage_distribution: List[StageShare]
    flagged_flows: List[FlaggedFlow]


# --------------------------------------------------------------------------
# /api/health
# --------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
