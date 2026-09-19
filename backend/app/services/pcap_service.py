"""
pcap_service — turns raw traffic (a flow-export CSV or a raw PCAP/PCAPNG
capture) into the canonical per-flow feature table, then windows that
table into the time-indexed network-state sequence `S_0, S_1, ..., S_n`
that the rest of the system (heuristic scorer today, the LSTM world model
later) actually reasons about.

Two entry points:
    parse_csv(file_bytes)   -> (DataFrame, warnings)
    parse_pcap(file_bytes)  -> (DataFrame, warnings)

Both return a DataFrame with the same canonical schema (one row per flow),
so `build_time_windows` and everything downstream never needs to know
which path the data came from.

`cybersecurity/pcap/parser.py` and `cybersecurity/flow/flow_extractor.py`
are thin re-exports of `parse_pcap`/`parse_csv` and `build_time_windows`
respectively — this module is the single source of truth for feature
extraction and windowing.
"""

from __future__ import annotations

import io
import re
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Canonical feature schema
# --------------------------------------------------------------------------

# ~20 flow-level features. Every state vector S_t is the mean of these
# across all flows that start inside that time window.
FLOW_FEATURES: List[str] = [
    "duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "total_bytes",
    "packets_per_second",
    "bytes_per_second",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "avg_packet_size",
    "packet_size_std",
    "avg_iat",
    "syn_ack_ratio",
    "retransmit_rate",
]

# 5 packet-level features. Real when parsed from a PCAP; derived as
# documented proxies when parsed from a flow-only CSV (flow-only datasets
# don't carry raw packets, so TTL/window-size variance is estimated from
# whatever flow-level signal correlates with it).
PACKET_FEATURES: List[str] = [
    "ttl_mean",
    "ttl_variance",
    "window_size_mean",
    "window_size_std",
    "flag_sequence_score",
]

# Aggregate features computed per time window rather than per flow (they
# only make sense across a group of flows).
WINDOW_LEVEL_FEATURES: List[str] = [
    "unique_dst_ports",
    "unique_src_dst_pairs",
    "flow_count",
]

ALL_STATE_FEATURES: List[str] = FLOW_FEATURES + PACKET_FEATURES + WINDOW_LEVEL_FEATURES

CANONICAL_COLUMNS: List[str] = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "label",
] + FLOW_FEATURES + PACKET_FEATURES


# --------------------------------------------------------------------------
# CSV column-name normalisation
# --------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """'Destination Port' -> 'destination_port' for robust matching."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


# Maps a set of *normalised* known column-name variants (CIC-IDS-2018,
# CTU-13, and generic NetFlow exports all differ) onto our canonical name.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "flow_id": ["flow_id", "flowid", "id"],
    "timestamp": ["timestamp", "flow_start", "start_time", "time", "date_first_seen"],
    "src_ip": ["src_ip", "source_ip", "srcaddr", "src_addr", "ipv4_src_addr"],
    "dst_ip": ["dst_ip", "destination_ip", "dstaddr", "dst_addr", "ipv4_dst_addr"],
    "src_port": ["src_port", "source_port", "l4_src_port", "sport"],
    "dst_port": ["dst_port", "destination_port", "l4_dst_port", "dport"],
    "protocol": ["protocol", "proto", "protocol_type"],
    "label": ["label", "class", "attack", "attack_type", "traffic_type"],
    "duration": ["flow_duration", "duration", "flow_duration_ms"],
    "total_fwd_packets": ["total_fwd_packets", "tot_fwd_pkts", "fwd_packets"],
    "total_bwd_packets": ["total_backward_packets", "tot_bwd_pkts", "bwd_packets"],
    "total_fwd_bytes": ["total_length_of_fwd_packets", "totlen_fwd_pkts", "fwd_bytes"],
    "total_bwd_bytes": ["total_length_of_bwd_packets", "totlen_bwd_pkts", "bwd_bytes"],
    "packets_per_second": ["flow_packets_s", "flow_pkts_s", "packets_per_second"],
    "bytes_per_second": ["flow_bytes_s", "bytes_per_second"],
    "syn_count": ["syn_flag_count", "syn_flag_cnt", "syn_count"],
    "ack_count": ["ack_flag_count", "ack_flag_cnt", "ack_count"],
    "fin_count": ["fin_flag_count", "fin_flag_cnt", "fin_count"],
    "rst_count": ["rst_flag_count", "rst_flag_cnt", "rst_count"],
    "psh_count": ["psh_flag_count", "psh_flag_cnt", "psh_count"],
    "urg_count": ["urg_flag_count", "urg_flag_cnt", "urg_count"],
    "avg_packet_size": ["average_packet_size", "pkt_size_avg", "avg_packet_size"],
    "packet_size_std": ["packet_length_std", "pkt_len_std", "packet_size_std"],
    "avg_iat": ["flow_iat_mean", "avg_iat"],
    "retransmit_rate": ["retransmission_rate", "retransmit_rate"],
}

_REVERSE_ALIASES: Dict[str, str] = {
    variant: canonical
    for canonical, variants in COLUMN_ALIASES.items()
    for variant in variants
}


def _map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Rename whatever columns we recognise onto the canonical schema.
    Returns the renamed frame and a {canonical: matched_original} map so
    callers can report exactly what was and wasn't found.
    """
    rename_map: Dict[str, str] = {}
    matched: Dict[str, str] = {}
    for col in df.columns:
        norm = _normalise(str(col))
        canonical = _REVERSE_ALIASES.get(norm)
        if canonical and canonical not in matched:
            rename_map[col] = canonical
            matched[canonical] = col
    return df.rename(columns=rename_map), matched


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def parse_csv(file_bytes: bytes) -> Tuple[pd.DataFrame, List[str]]:
    """Parse a CIC-IDS-2018 / CTU-13 style flow-export CSV into the
    canonical per-flow schema. Missing columns are filled with documented
    defaults/proxies rather than failing, so an unfamiliar export still
    produces a usable (if less precise) session.
    """
    warnings: List[str] = []
    raw = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    if raw.empty:
        raise ValueError("CSV has no rows.")

    df, matched = _map_columns(raw)
    n = len(df)

    out = pd.DataFrame(index=df.index)
    out["flow_id"] = df["flow_id"] if "flow_id" in matched else [f"flow-{i}" for i in range(n)]
    out["src_ip"] = df["src_ip"].astype(str) if "src_ip" in matched else "0.0.0.0"
    out["dst_ip"] = df["dst_ip"].astype(str) if "dst_ip" in matched else "0.0.0.0"
    out["src_port"] = _numeric(df["src_port"]).astype(int) if "src_port" in matched else 0
    out["dst_port"] = _numeric(df["dst_port"]).astype(int) if "dst_port" in matched else 0
    out["protocol"] = df["protocol"].astype(str) if "protocol" in matched else "TCP"
    out["label"] = df["label"].astype(str) if "label" in matched else "UNKNOWN"

    if "timestamp" in matched:
        # Flow exports store the timestamp either as a plain number (epoch
        # or already-relative seconds) or as a formatted date string
        # ("3/1/2018 8:47:38 AM"). Try numeric first — parsing a numeric
        # column with `to_datetime` silently reinterprets it as
        # nanoseconds-since-epoch and collapses an entire session into a
        # single instant, which is worse than just using the number as-is.
        numeric_ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if numeric_ts.notna().sum() >= n * 0.5:
            out["timestamp"] = (numeric_ts.ffill().fillna(0.0) - numeric_ts.min())
        else:
            parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if parsed_ts.notna().sum() >= n * 0.5:
                base = parsed_ts.min()
                out["timestamp"] = (parsed_ts - base).dt.total_seconds().ffill().fillna(0.0)
            else:
                warnings.append(
                    "Timestamp column present but mostly unparseable — synthesised "
                    "sequential timestamps (1 flow per 100ms) instead."
                )
                out["timestamp"] = np.arange(n) * 0.1
    else:
        warnings.append(
            "No recognisable timestamp column — synthesised sequential timestamps "
            "(1 flow per 100ms) so windowing still works."
        )
        out["timestamp"] = np.arange(n) * 0.1

    for feat in FLOW_FEATURES:
        if feat in matched:
            out[feat] = _numeric(df[feat])
        else:
            out[feat] = 0.0

    if "syn_count" not in matched and "syn_ack_ratio" not in matched:
        warnings.append(
            "Flag-count columns not found — syn_count/ack_count/etc default to 0; "
            "SYN-flood style signals will be muted for this session."
        )

    # Derive anything we can rather than leaving it at the 0.0 default.
    if (out["total_packets"] == 0).all():
        out["total_packets"] = out["total_fwd_packets"] + out["total_bwd_packets"]
    if (out["total_bytes"] == 0).all():
        out["total_bytes"] = out["total_fwd_bytes"] + out["total_bwd_bytes"]
    if (out["syn_ack_ratio"] == 0).all():
        out["syn_ack_ratio"] = out["syn_count"] / out["ack_count"].replace(0, np.nan)
        out["syn_ack_ratio"] = out["syn_ack_ratio"].fillna(0.0)

    # Packet-level features: flow-only exports don't carry raw packets, so
    # these are documented proxies derived from flow-level signal.
    proxy_note_needed = False
    for feat in PACKET_FEATURES:
        if feat in matched:
            out[feat] = _numeric(df[feat])
        else:
            proxy_note_needed = True
            if feat == "ttl_mean":
                out[feat] = 64.0  # common default TTL, used as a neutral baseline
            elif feat == "ttl_variance":
                # proxy: retransmit-heavy / erratic flows correlate with TTL churn
                out[feat] = out["retransmit_rate"] * 10.0
            elif feat == "window_size_mean":
                out[feat] = (out["avg_packet_size"] * 4.0).clip(lower=0)
            elif feat == "window_size_std":
                out[feat] = out["packet_size_std"]
            elif feat == "flag_sequence_score":
                # proxy: high SYN with low ACK looks like scan/flood sequencing
                denom = (out["syn_count"] + out["ack_count"]).replace(0, np.nan)
                out[feat] = (out["syn_count"] / denom).fillna(0.0)

    if proxy_note_needed:
        warnings.append(
            "One or more packet-level features (TTL/window-size/flag-sequencing) "
            "aren't present in a flow-only CSV — using documented proxy formulas "
            "(see pcap_service.parse_csv). Upload a .pcap for real packet-level features."
        )

    return out[CANONICAL_COLUMNS], warnings


def parse_pcap(file_bytes: bytes) -> Tuple[pd.DataFrame, List[str]]:
    """Parse a raw .pcap/.pcapng capture into the same canonical schema,
    computing real (not proxied) packet-level features.
    """
    try:
        from scapy.all import IP, TCP, UDP, PcapReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without scapy
        raise RuntimeError(
            "PCAP parsing requires the optional 'scapy' dependency. "
            "Install it (`pip install scapy`) or upload a CSV flow export instead."
        ) from exc

    warnings: List[str] = []
    flows: Dict[Tuple[str, str, int, int, str], Dict[str, Any]] = {}

    with tempfile.NamedTemporaryFile(suffix=".pcap") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        packet_count = 0
        for pkt in PcapReader(tmp.name):
            if IP not in pkt:
                continue
            packet_count += 1
            ip = pkt[IP]
            proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else str(ip.proto))
            sport = int(pkt[TCP].sport) if TCP in pkt else (int(pkt[UDP].sport) if UDP in pkt else 0)
            dport = int(pkt[TCP].dport) if TCP in pkt else (int(pkt[UDP].dport) if UDP in pkt else 0)
            key = (ip.src, ip.dst, sport, dport, proto)
            ts = float(pkt.time)
            size = len(pkt)
            ttl = int(ip.ttl)
            window = int(pkt[TCP].window) if TCP in pkt else 0
            flags = str(pkt[TCP].flags) if TCP in pkt else ""

            f = flows.setdefault(key, {
                "timestamps": [], "sizes": [], "ttls": [], "windows": [],
                "syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0, "urg": 0,
                "fwd_packets": 0, "fwd_bytes": 0,
            })
            f["timestamps"].append(ts)
            f["sizes"].append(size)
            f["ttls"].append(ttl)
            f["windows"].append(window)
            f["fwd_packets"] += 1
            f["fwd_bytes"] += size
            if "S" in flags:
                f["syn"] += 1
            if "A" in flags:
                f["ack"] += 1
            if "F" in flags:
                f["fin"] += 1
            if "R" in flags:
                f["rst"] += 1
            if "P" in flags:
                f["psh"] += 1
            if "U" in flags:
                f["urg"] += 1

    if not flows:
        raise ValueError("No IP packets found in capture.")

    min_ts = min(f["timestamps"][0] for f in flows.values())
    rows = []
    for i, ((src, dst, sport, dport, proto), f) in enumerate(flows.items()):
        ts = np.array(f["timestamps"])
        sizes = np.array(f["sizes"], dtype=float)
        ttls = np.array(f["ttls"], dtype=float)
        windows = np.array(f["windows"], dtype=float)
        duration = float(ts.max() - ts.min()) if len(ts) > 1 else 0.0
        iat = np.diff(ts) if len(ts) > 1 else np.array([0.0])
        # retransmits proxied as repeated (ts, size) pairs within a flow —
        # a real duplicate-segment detector needs sequence numbers, which
        # is out of scope for this fast path.
        retransmits = max(0, len(ts) - len(set(zip(ts.tolist(), sizes.tolist()))))

        rows.append({
            "flow_id": f"flow-{i}",
            "timestamp": float(ts.min() - min_ts),
            "src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport,
            "protocol": proto, "label": "UNKNOWN",
            "duration": duration,
            "total_fwd_packets": f["fwd_packets"], "total_bwd_packets": 0,
            "total_packets": f["fwd_packets"],
            "total_fwd_bytes": f["fwd_bytes"], "total_bwd_bytes": 0,
            "total_bytes": f["fwd_bytes"],
            "packets_per_second": f["fwd_packets"] / duration if duration > 0 else float(f["fwd_packets"]),
            "bytes_per_second": f["fwd_bytes"] / duration if duration > 0 else float(f["fwd_bytes"]),
            "syn_count": f["syn"], "ack_count": f["ack"], "fin_count": f["fin"],
            "rst_count": f["rst"], "psh_count": f["psh"], "urg_count": f["urg"],
            "avg_packet_size": float(sizes.mean()), "packet_size_std": float(sizes.std()),
            "avg_iat": float(iat.mean()),
            "syn_ack_ratio": (f["syn"] / f["ack"]) if f["ack"] else 0.0,
            "retransmit_rate": retransmits / len(ts),
            "ttl_mean": float(ttls.mean()), "ttl_variance": float(ttls.var()),
            "window_size_mean": float(windows.mean()), "window_size_std": float(windows.std()),
            "flag_sequence_score": (f["syn"] / (f["syn"] + f["ack"])) if (f["syn"] + f["ack"]) else 0.0,
        })

    warnings.append(f"Parsed {packet_count} packets into {len(rows)} flows via 5-tuple grouping.")
    return pd.DataFrame(rows)[CANONICAL_COLUMNS], warnings


# --------------------------------------------------------------------------
# Time windowing — the actual "network state" the world model reasons about
# --------------------------------------------------------------------------

def build_time_windows(df: pd.DataFrame, window_seconds: int = 5) -> List[Dict[str, Any]]:
    """Group flows into fixed windows and average each window into a
    single state vector S_t. Returns a list of:
        {window_index, timestamp, state: {feature: value}, flow_indices, flow_count, dominant_label}
    ordered by window_index, with no gaps (empty windows get a
    zero/hold state so the sequence stays evenly spaced for the model).
    """
    if df.empty:
        return []

    df = df.copy()
    t0 = df["timestamp"].min()
    df["_window"] = ((df["timestamp"] - t0) // window_seconds).astype(int)
    max_window = int(df["_window"].max())

    windows: List[Dict[str, Any]] = []
    last_state: Dict[str, float] = {f: 0.0 for f in ALL_STATE_FEATURES}

    for w in range(max_window + 1):
        group = df[df["_window"] == w]
        state: Dict[str, float] = {}
        if group.empty:
            # Hold the previous state (network went quiet) rather than a
            # hard zero, which would look like a discontinuous cliff to
            # the model / chart for no operational reason.
            state = dict(last_state)
            flow_indices: List[int] = []
            dominant_label = "UNKNOWN"
        else:
            for feat in FLOW_FEATURES + PACKET_FEATURES:
                state[feat] = float(group[feat].mean())
            state["unique_dst_ports"] = float(group["dst_port"].nunique())
            state["unique_src_dst_pairs"] = float(group[["src_ip", "dst_ip"]].drop_duplicates().shape[0])
            state["flow_count"] = float(len(group))
            flow_indices = group.index.tolist()
            dominant_label = group["label"].mode().iat[0] if not group["label"].mode().empty else "UNKNOWN"
            last_state = state

        windows.append({
            "window_index": w,
            "timestamp": float(t0 + w * window_seconds),
            "state": state,
            "flow_indices": flow_indices,
            "flow_count": len(flow_indices),
            "dominant_label": dominant_label,
        })

    return windows
