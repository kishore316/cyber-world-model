"""
report_service — session-wide summary statistics for the analysis panel:
top talkers, top ports, protocol mix, MITRE stage distribution across the
timeline, and the highest-risk flows. Reuses prediction_service.predict()
for the timeline/flagged-flows pieces rather than recomputing the scorer,
so the analysis and prediction endpoints can never disagree with each other.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from app.services import prediction_service
from app.services.store import SessionData


def _top_talkers(flows, top_n: int = 5) -> List[Dict[str, Any]]:
    counts: Counter = Counter()
    bytes_by_ip: Dict[str, float] = {}
    for _, row in flows.iterrows():
        for ip in (row["src_ip"], row["dst_ip"]):
            counts[ip] += 1
            bytes_by_ip[ip] = bytes_by_ip.get(ip, 0.0) + float(row.get("total_bytes", 0.0))
    top = counts.most_common(top_n)
    return [{"ip": ip, "flow_count": n, "total_bytes": bytes_by_ip.get(ip, 0.0)} for ip, n in top]


def _top_ports(flows, top_n: int = 5) -> List[Dict[str, Any]]:
    counts = Counter(int(p) for p in flows["dst_port"].tolist())
    return [{"port": port, "flow_count": n} for port, n in counts.most_common(top_n)]


def _protocol_breakdown(flows) -> List[Dict[str, Any]]:
    counts = Counter(str(p) for p in flows["protocol"].tolist())
    total = sum(counts.values()) or 1
    return [
        {"protocol": proto, "flow_count": n, "percentage": round(100.0 * n / total, 2)}
        for proto, n in counts.most_common()
    ]


def _stage_distribution(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observed = [t for t in timeline if not t["is_forecast"]]
    counts = Counter(t["stage"] for t in observed)
    total = sum(counts.values()) or 1
    return [
        {"stage": stage, "window_count": n, "percentage": round(100.0 * n / total, 2)}
        for stage, n in counts.most_common()
    ]


def analyse(session: SessionData) -> Dict[str, Any]:
    flows = session.flows
    prediction = prediction_service.predict(session)

    return {
        "session_id": session.session_id,
        "total_flows": int(len(flows)),
        "total_packets": int(flows["total_packets"].sum()),
        "total_windows": len(session.windows),
        "unique_ip_pairs": int(flows[["src_ip", "dst_ip"]].drop_duplicates().shape[0]),
        "top_talkers": _top_talkers(flows),
        "top_ports": _top_ports(flows),
        "protocol_breakdown": _protocol_breakdown(flows),
        "stage_distribution": _stage_distribution(prediction["timeline"]),
        "flagged_flows": prediction["flagged_flows"],
    }
