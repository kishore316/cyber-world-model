# Cyber World Model

### Predictive cyber defence through temporal network analysis

A collaborative Smart India Hackathon research prototype exploring how network states evolve over time and how future attack progression could be forecast and explained.

**[Open interactive demo](https://cyber-world-model-predictive-cyber-defence.ai.studio/)** · [Original team repository](https://github.com/punit-28/SIH-Project/) · [Kishore’s portfolio](https://kishore-portifolio.ai.studio/)

## At a glance

- Traffic ingestion from CSV and PCAP/PCAPNG captures.
- Five-second network-state windows and temporal analysis.
- A dashboard for infiltration timelines, attack-stage tracking and explanatory features.
- A FastAPI backend with heuristic prediction as the current repository fallback.
- A proposed LSTM-and-attention world model for future learned state transitions.

## Current implementation and demo

The repository’s existing build-status notes below describe an implemented frontend/backend using heuristic mode; a trained model checkpoint is not included. The separately hosted demo presents an embedded TypeScript ML workflow and opens with clearly labeled synthetic/demo traffic. The demo and this repository are separate prototype surfaces and may differ in implementation. No model accuracy or production-readiness claim is made here.

## Team and attribution

This is Kishore Karuturi’s portfolio fork of [punit-28/SIH-Project](https://github.com/punit-28/SIH-Project/). It preserves the original team’s source and commit history. Kishore participated in the collaborative Cyberworld prototype selected in NIT Srinagar’s internal Smart India Hackathon.

For setup, architecture, API routes, evaluation plans and limitations, see the original technical documentation preserved below.

---

> **Build status:** `frontend/` and `backend/` are implemented and working
> end-to-end right now, running in **heuristic mode** (see
> `backend/app/services/prediction_service.py`) since no trained
> checkpoint exists yet. `ml/` (the LSTM world model + training) and
> `cybersecurity/` (the standalone MITRE-mapping/feature-extraction
> re-export layer) are **not built yet** — that's next. The backend is
> written so dropping those in later needs no route or schema changes;
> see the "Known limitations" section (§12) below.


# Cyber World Model
### Predictive Cyber Defence via Learned Network State-Transition Dynamics


> A world-model-based AI system that learns how a network's traffic *evolves
> over time*, forecasts where an in-progress attack is heading, and explains
> *why* — before the compromise completes.


---


## 1. Problem statement (what this solves)


Traditional intrusion detection treats every network flow in isolation and
maps it to a binary benign/malicious label. That throws away the temporal
structure of a real intrusion — the order in which ports get probed, the
SYN-before-ACK-flood pattern, the timing between reconnaissance packets and
the lateral movement that follows.


An intrusion is a **process unfolding over time**, not a single anomalous
packet. This project builds a **World Model**: instead of classifying
traffic, it learns the transition dynamics


```
P(S_t+1 | S_t)
```


— given the current network state (active flows, flag distributions, port
activity, packet timing), what is the probability distribution over the
*next* state? That lets the system roll forward K steps and ask: **does this
trajectory converge on an infiltration state, and if so, which MITRE ATT&CK
stage is it heading toward?**


Full original hackathon brief: see `docs/problem_statement.md` (or the brief
you were given — this README assumes you already have it).


---


## 2. Core idea in one picture


```
┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ Raw traffic   │──▶│ Feature pipeline  │──▶│ Time-windowed      │──▶│ World model       │
│ PCAP / CSV    │   │ flow + packet     │   │ state vectors S_t  │   │ (LSTM + attention) │
