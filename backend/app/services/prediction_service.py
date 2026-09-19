"""
prediction_service — turns a session's windowed state sequence into an
infiltration-probability timeline, a predicted MITRE ATT&CK stage, a
K-step forecast, and signed feature-attribution explanations.

Two modes, same response shape:
  - "model":     ml/models/world_model.py rollout (LSTM + attention).
                 Only activates once a checkpoint exists at
                 models/world_model.pt (or $WORLD_MODEL_PATH) *and* the
                 ml/ package has been built — see `_try_load_model`.
  - "heuristic":  transparent, fully-documented weighted scorer below.
                 This is what runs today, before ml/ exists, and it's
                 what keeps the API, the frontend, and this whole demo
                 usable in the meantime. Every weight is a named,
                 inspectable constant — nothing here is a black box.

Swapping mode later requires no route or schema changes: `predict()`
already returns the same `mode` field the frontend can display either way.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from app.services.store import SessionData

# --------------------------------------------------------------------------
# MITRE ATT&CK stages this scorer can point to.
#
# NOTE: this is a working copy for the heuristic scorer. The canonical,
# fully-described version (tactic IDs + long-form descriptions) will live
# in cybersecurity/attack_mapping/mitre_mapping.py once that layer is
# built — this dict is written so it can be replaced by an import from
# there without changing anything below it.
# --------------------------------------------------------------------------

NORMAL_STAGE = "Normal"

STAGE_BY_DRIVER: Dict[str, str] = {
    "unique_dst_ports": "Reconnaissance",
    "syn_ack_ratio": "Initial Access",
    "unique_src_dst_pairs": "Lateral Movement",
    "flag_sequence_score": "Command & Control",
    "bytes_per_second": "Exfiltration",
}

STAGE_ORDER = ["Reconnaissance", "Initial Access", "Lateral Movement", "Command & Control", "Exfiltration"]

# --------------------------------------------------------------------------
# Heuristic scorer — documented weights + normalisation caps.
#
# Each weight reflects how strongly that signal, on its own, indicates an
# in-progress intrusion rather than ordinary traffic. Weights sum to 1.0
# so the raw weighted sum of normalised features is already a 0..1
# probability with no extra rescaling needed.
# --------------------------------------------------------------------------

HEURISTIC_WEIGHTS: Dict[str, float] = {
    "unique_dst_ports": 0.22,       # port-scan / fan-out reconnaissance
    "syn_count": 0.18,              # SYN-flood / brute-force bursts
    "syn_ack_ratio": 0.12,          # half-open connections, exploitation attempts
    "flag_sequence_score": 0.10,    # irregular flag sequencing (scan/flood signature)
    "retransmit_rate": 0.10,        # instability from aggressive scanning or floods
    "unique_src_dst_pairs": 0.08,   # host-to-host spread (lateral movement)
    "packets_per_second": 0.08,     # burst intensity
    "bytes_per_second": 0.06,       # sustained throughput (exfiltration signal)
    "ttl_variance": 0.06,           # route/host inconsistency across a flow window
}

# Values at/above the cap are treated as "maximally suspicious" (normalised to 1.0).
# Chosen as generous upper bounds for a single 5-second window on a small/medium LAN.
HEURISTIC_CAPS: Dict[str, float] = {
    "unique_dst_ports": 30.0,
    "syn_count": 50.0,
    "syn_ack_ratio": 5.0,
    "flag_sequence_score": 1.0,
    "retransmit_rate": 0.5,
    "unique_src_dst_pairs": 15.0,
    "packets_per_second": 500.0,
    "bytes_per_second": 1_000_000.0,
    "ttl_variance": 50.0,
}

INFILTRATION_THRESHOLD = 0.35  # below this, a window is reported as "Normal"


def _normalise(feature: str, value: float) -> float:
    cap = HEURISTIC_CAPS.get(feature, 1.0)
    if cap <= 0:
        return 0.0
    return float(min(1.0, max(0.0, value) / cap))


def _window_probability(state: Dict[str, float]) -> float:
    score = 0.0
    for feat, weight in HEURISTIC_WEIGHTS.items():
        score += weight * _normalise(feat, state.get(feat, 0.0))
    return float(min(1.0, max(0.0, score)))


def _window_contributions(state: Dict[str, float], baseline: Dict[str, float]) -> List[Dict[str, Any]]:
    """Signed, per-feature attribution: how far this window's normalised
    value sits from the session's own running-baseline value, scaled by
    that feature's weight. Positive = pushing risk up vs. this session's
    norm; negative = pushing it down.
    """
    contributions = []
    for feat, weight in HEURISTIC_WEIGHTS.items():
        norm_val = _normalise(feat, state.get(feat, 0.0))
        norm_base = _normalise(feat, baseline.get(feat, 0.0))
        contribution = weight * (norm_val - norm_base)
        contributions.append({
            "feature": feat,
            "value": float(state.get(feat, 0.0)),
            "contribution": float(contribution),
            "direction": "up" if contribution >= 0 else "down",
        })
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions


def _predict_stage(probability: float, contributions: List[Dict[str, Any]]) -> str:
    if probability < INFILTRATION_THRESHOLD:
        return NORMAL_STAGE
    positive_drivers = [c for c in contributions if c["contribution"] > 0 and c["feature"] in STAGE_BY_DRIVER]
    if not positive_drivers:
        return STAGE_ORDER[0]
    top = max(positive_drivers, key=lambda c: c["contribution"])
    return STAGE_BY_DRIVER[top["feature"]]


def _running_baseline(states: List[Dict[str, float]], upto: int) -> Dict[str, float]:
    """Mean of every feature over windows [0, upto) — 'what normal has
    looked like in this session so far'. Falls back to the current window
    itself for window 0, since there's no history yet.
    """
    if upto <= 0:
        return dict(states[0]) if states else {}
    window_slice = states[:upto]
    keys = window_slice[0].keys()
    return {k: float(np.mean([s.get(k, 0.0) for s in window_slice])) for k in keys}


def _try_load_model():
    """Hook for the trained LSTM world model. Returns None until ml/ has
    been built and a checkpoint exists — the caller falls back to the
    heuristic scorer whenever this returns None, so nothing here is
    load-bearing yet.
    """
    model_path = os.environ.get("WORLD_MODEL_PATH", "models/world_model.pt")
    if not os.path.exists(model_path):
        return None
    try:
        import torch  # noqa: F401
        from ml.models.world_model import WorldModel  # type: ignore
    except ImportError:
        return None
    try:
        model = WorldModel.load(model_path)  # type: ignore[attr-defined]
        return model
    except Exception:
        return None


def _rollout_probabilities(probabilities: List[float], k: int) -> List[float]:
    """Heuristic-mode forecast: linear-trend extrapolation over the last
    few observed windows. This is intentionally simple — it's a stand-in
    for `ml/models/world_model.py::rollout`'s learned K-step simulation,
    which feeds its own predicted state back in K times. Swap this
    function out once that exists; the response shape doesn't change.
    """
    if not probabilities:
        return [0.0] * k
    history = probabilities[-6:]
    if len(history) == 1:
        slope = 0.0
    else:
        x = np.arange(len(history))
        slope, _intercept = np.polyfit(x, np.array(history), 1)
    last = probabilities[-1]
    forecast = []
    for step in range(1, k + 1):
        forecast.append(float(min(1.0, max(0.0, last + slope * step))))
    return forecast


def _baseline_metrics(windows: List[Dict[str, Any]], probabilities: List[float]) -> List[Dict[str, float]]:
    """Illustrative benchmark panel for the dashboard: the same weighted
    scorer ('world model' slot, heuristic mode today) vs. a deliberately
    cruder single-signal classifier ('baseline' slot, standing in for
    ml/baseline/logistic_regression.py until that's trained). Ground
    truth uses the dataset's own Label column when present; otherwise
    falls back to the main scorer's own calls, which makes the panel
    illustrative rather than a rigorous benchmark. The authoritative,
    held-out-split benchmark is ml/evaluation/evaluate.py, to be added
    alongside the trained model.
    """
    labels = [w["dominant_label"] for w in windows]
    labels_known = any(l not in ("UNKNOWN", "", None) for l in labels)

    def is_attack_label(label: str) -> bool:
        return label.upper() not in ("BENIGN", "NORMAL", "UNKNOWN", "")

    if labels_known:
        y_true = np.array([1 if is_attack_label(l) else 0 for l in labels])
    else:
        y_true = np.array([1 if p >= INFILTRATION_THRESHOLD else 0 for p in probabilities])

    y_pred_main = np.array([1 if p >= INFILTRATION_THRESHOLD else 0 for p in probabilities])

    baseline_scores = []
    for w in windows:
        state = w["state"]
        s = 0.5 * _normalise("syn_count", state.get("syn_count", 0.0)) \
            + 0.5 * _normalise("unique_dst_ports", state.get("unique_dst_ports", 0.0))
        baseline_scores.append(s)
    y_pred_baseline = np.array([1 if s > 0.5 else 0 for s in baseline_scores])

    def metrics(y_pred: np.ndarray) -> Tuple[float, float, float, float]:
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return precision, recall, f1, fpr

    p_main, r_main, f1_main, fpr_main = metrics(y_pred_main)
    p_base, r_base, f1_base, fpr_base = metrics(y_pred_baseline)

    return [
        {"metric": "F1 Score", "world_model": f1_main, "baseline": f1_base},
        {"metric": "Precision", "world_model": p_main, "baseline": p_base},
        {"metric": "Recall", "world_model": r_main, "baseline": r_base},
        {"metric": "False Positive Rate", "world_model": fpr_main, "baseline": fpr_base},
    ]


def _flagged_flows(session: SessionData, windows: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    flows = session.flows
    flow_to_window = {}
    for w in windows:
        for idx in w["flow_indices"]:
            flow_to_window[idx] = w["window_index"]

    scored = []
    for idx, row in flows.iterrows():
        state = {feat: row.get(feat, 0.0) for feat in HEURISTIC_WEIGHTS}
        score = _window_probability(state)
        scored.append((idx, score))
    scored.sort(key=lambda t: t[1], reverse=True)

    out = []
    for idx, score in scored[:top_n]:
        row = flows.loc[idx]
        top_feat = max(HEURISTIC_WEIGHTS.keys(), key=lambda f: _normalise(f, row.get(f, 0.0)) * HEURISTIC_WEIGHTS[f])
        out.append({
            "flow_id": str(row["flow_id"]),
            "src_ip": str(row["src_ip"]),
            "dst_ip": str(row["dst_ip"]),
            "dst_port": int(row["dst_port"]),
            "protocol": str(row["protocol"]),
            "window_index": int(flow_to_window.get(idx, 0)),
            "risk_score": float(score),
            "reason": f"elevated {top_feat.replace('_', ' ')}",
        })
    return out


def predict(session: SessionData, k: int = 5) -> Dict[str, Any]:
    windows = session.windows
    if not windows:
        raise ValueError("Session has no windows to predict over.")

    states = [w["state"] for w in windows]
    model = _try_load_model()
    mode = "model" if model is not None else "heuristic"

    probabilities: List[float] = []
    stages: List[str] = []
    all_contributions: List[List[Dict[str, Any]]] = []

    if mode == "model":
        # Placeholder branch: exercised only once ml/ ships a real
        # checkpoint + WorldModel.load/.rollout implementation.
        result = model.rollout(states, k=k)  # type: ignore[attr-defined]
        probabilities = result["probabilities"]
        stages = result["stages"]
        all_contributions = result["contributions"]
    else:
        for i, state in enumerate(states):
            baseline = _running_baseline(states, i)
            prob = _window_probability(state)
            contributions = _window_contributions(state, baseline)
            probabilities.append(prob)
            stages.append(_predict_stage(prob, contributions))
            all_contributions.append(contributions)

    forecast_probs = _rollout_probabilities(probabilities, k) if mode == "heuristic" else result["forecast"]
    last_stage = stages[-1] if stages else NORMAL_STAGE
    window_seconds = session.window_seconds

    timeline = []
    for w, prob, stage in zip(windows, probabilities, stages):
        timeline.append({
            "window_index": w["window_index"],
            "timestamp": w["timestamp"],
            "label": f"t+{w['window_index'] * window_seconds}s",
            "probability": prob,
            "stage": stage,
            "is_forecast": False,
        })
    last_index = windows[-1]["window_index"] if windows else 0
    last_ts = windows[-1]["timestamp"] if windows else 0.0
    for step, prob in enumerate(forecast_probs, start=1):
        forecast_stage = last_stage if prob >= INFILTRATION_THRESHOLD else NORMAL_STAGE
        timeline.append({
            "window_index": last_index + step,
            "timestamp": last_ts + step * window_seconds,
            "label": f"t+{(last_index + step) * window_seconds}s",
            "probability": prob,
            "stage": forecast_stage,
            "is_forecast": True,
        })

    top_drivers = all_contributions[-1][:5] if all_contributions else []
    flagged = _flagged_flows(session, windows)
    baseline_comparison = _baseline_metrics(windows, probabilities)

    return {
        "session_id": session.session_id,
        "mode": mode,
        "current_probability": probabilities[-1] if probabilities else 0.0,
        "current_stage": last_stage,
        "timeline": timeline,
        "top_drivers": top_drivers,
        "flagged_flows": flagged,
        "baseline_comparison": baseline_comparison,
    }
