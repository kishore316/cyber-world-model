from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.schemas import UploadResponse
from app.services import pcap_service
from app.services.store import store

router = APIRouter(prefix="/api", tags=["upload"])

WINDOW_SECONDS = 5
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_capture(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or "upload"
    lower = filename.lower()

    if lower.endswith(".pcap") or lower.endswith(".pcapng"):
        source_type = "pcap"
    elif lower.endswith(".csv"):
        source_type = "csv"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type — upload a .csv flow export or a .pcap/.pcapng capture.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large — 200 MB limit for this demo.")

    try:
        if source_type == "csv":
            flows, warnings = pcap_service.parse_csv(contents)
        else:
            flows, warnings = pcap_service.parse_pcap(contents)
    except RuntimeError as exc:
        # e.g. scapy not installed — a config problem, not a bad upload.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise HTTPException(status_code=422, detail=f"Unexpected parsing error: {exc}") from exc

    windows = pcap_service.build_time_windows(flows, window_seconds=WINDOW_SECONDS)
    if not windows:
        raise HTTPException(status_code=422, detail="No usable flows found after parsing.")

    session = store.create(
        filename=filename,
        source_type=source_type,
        flows=flows,
        windows=windows,
        window_seconds=WINDOW_SECONDS,
        warnings=warnings,
    )

    duration = windows[-1]["timestamp"] + WINDOW_SECONDS - windows[0]["timestamp"] if windows else 0.0

    return UploadResponse(
        session_id=session.session_id,
        filename=filename,
        source_type=source_type,
        flow_count=int(len(flows)),
        packet_count=int(flows["total_packets"].sum()),
        window_count=len(windows),
        window_seconds=WINDOW_SECONDS,
        duration_seconds=float(duration),
        warnings=warnings,
    )
