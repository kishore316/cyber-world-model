"""
Feature-window shape sanity checks. No FastAPI/torch dependency, so this
runs even before `pip install -r backend/requirements.txt` pulls in the
web-framework bits — only pandas/numpy are needed.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services import pcap_service  # noqa: E402


def _synthetic_csv_bytes(n_benign=20, n_attack=40) -> bytes:
    rows = []
    t = 0.0
    for i in range(n_benign):
        rows.append({
            "Flow ID": f"b{i}", "Timestamp": t, "Src IP": "10.0.0.5", "Dst IP": "10.0.0.1",
            "Src Port": 40000 + i, "Destination Port": 443, "Protocol": "TCP", "Label": "BENIGN",
            "Flow Duration": 500000, "Tot Fwd Pkts": 10, "Tot Bwd Pkts": 8,
            "TotLen Fwd Pkts": 1200, "TotLen Bwd Pkts": 900,
            "Flow Pkts/s": 20.0, "Flow Byts/s": 2000.0,
            "SYN Flag Cnt": 1, "ACK Flag Cnt": 9,
        })
        t += 1.0
    for i in range(n_attack):
        rows.append({
            "Flow ID": f"a{i}", "Timestamp": t, "Src IP": "203.0.113.9", "Dst IP": "10.0.0.1",
            "Src Port": 50000 + i, "Destination Port": 1000 + i, "Protocol": "TCP", "Label": "PortScan",
            "Flow Duration": 5000, "Tot Fwd Pkts": 2, "Tot Bwd Pkts": 0,
            "TotLen Fwd Pkts": 80, "TotLen Bwd Pkts": 0,
            "Flow Pkts/s": 400.0, "Flow Byts/s": 8000.0,
            "SYN Flag Cnt": 5, "ACK Flag Cnt": 0, "RST Flag Cnt": 1,
        })
        t += 0.1
    return pd.DataFrame(rows).to_csv(index=False).encode()


def test_parse_csv_produces_canonical_schema():
    flows, warnings = pcap_service.parse_csv(_synthetic_csv_bytes())
    for col in pcap_service.CANONICAL_COLUMNS:
        assert col in flows.columns
    assert len(flows) == 60


def test_parse_csv_handles_unrecognised_columns_gracefully():
    df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["x", "y", "z"]})
    flows, warnings = pcap_service.parse_csv(df.to_csv(index=False).encode())
    assert len(flows) == 3
    assert any("timestamp" in w.lower() for w in warnings)


def test_build_time_windows_shape():
    flows, _ = pcap_service.parse_csv(_synthetic_csv_bytes())
    windows = pcap_service.build_time_windows(flows, window_seconds=5)
    assert len(windows) > 0
    for w in windows:
        assert "window_index" in w and "state" in w and "flow_count" in w
        for feat in pcap_service.ALL_STATE_FEATURES:
            assert feat in w["state"]


def test_build_time_windows_no_gaps_in_index():
    flows, _ = pcap_service.parse_csv(_synthetic_csv_bytes())
    windows = pcap_service.build_time_windows(flows, window_seconds=5)
    indices = [w["window_index"] for w in windows]
    assert indices == list(range(len(indices)))


def test_empty_dataframe_returns_no_windows():
    empty = pd.DataFrame(columns=pcap_service.CANONICAL_COLUMNS)
    assert pcap_service.build_time_windows(empty) == []
