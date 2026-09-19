"""
API smoke tests. Needs backend/requirements.txt installed
(fastapi + httpx's TestClient support) — run from the repo root:

    pip install -r backend/requirements.txt httpx
    pytest tests/
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _sample_csv_bytes() -> bytes:
    rows = []
    t = 0.0
    for i in range(30):
        rows.append({
            "Flow ID": f"f{i}", "Timestamp": t, "Src IP": "10.0.0.5", "Dst IP": "10.0.0.1",
            "Src Port": 40000 + i, "Destination Port": 443, "Protocol": "TCP", "Label": "BENIGN",
            "Flow Duration": 500000, "Tot Fwd Pkts": 10, "Tot Bwd Pkts": 8,
            "TotLen Fwd Pkts": 1200, "TotLen Bwd Pkts": 900,
            "Flow Pkts/s": 20.0, "Flow Byts/s": 2000.0,
            "SYN Flag Cnt": 1, "ACK Flag Cnt": 9,
        })
        t += 1.0
    return pd.DataFrame(rows).to_csv(index=False).encode()


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_upload_rejects_bad_extension():
    res = client.post("/api/upload", files={"file": ("data.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_upload_predict_analysis_roundtrip():
    csv_bytes = _sample_csv_bytes()
    res = client.post("/api/upload", files={"file": ("sample.csv", csv_bytes, "text/csv")})
    assert res.status_code == 200
    payload = res.json()
    session_id = payload["session_id"]
    assert payload["flow_count"] == 30

    pred = client.get(f"/api/prediction/{session_id}")
    assert pred.status_code == 200
    body = pred.json()
    assert body["mode"] == "heuristic"
    assert 0.0 <= body["current_probability"] <= 1.0
    assert len(body["timeline"]) >= payload["window_count"]

    ana = client.get(f"/api/analysis/{session_id}")
    assert ana.status_code == 200
    assert ana.json()["total_flows"] == 30


def test_prediction_unknown_session_404():
    res = client.get("/api/prediction/does-not-exist")
    assert res.status_code == 404
