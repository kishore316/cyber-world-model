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
│ flow records  │   │ level features    │   │ (5s windows)       │   │                    │
└──────────────┘   └───────────────────┘   └────────────────────┘   └─────────┬─────────┘
                                                                                │
                                          K-step forward rollout: S_t+1 … S_t+K │
                                                                                ▼
                                          ┌────────────────────────────────────────────┐
                                          │ Infiltration probability timeline           │
                                          │ Predicted MITRE ATT&CK stage                │
                                          │ Top contributing features (explainability)  │
                                          └───────────────────┬──────────────────────────┘
                                                                │
                                                                ▼
                                                   ┌─────────────────────────┐
                                                   │ Dashboard (frontend)     │
                                                   │ timeline · stage tracker │
                                                   │ flagged flows · benchmark│
                                                   └─────────────────────────┘
```

---

## 3. Project structure

```
cyber-world-model/
│
├── frontend/                  # Self-contained responsive dashboard (no build step)
│   └── index.html             # Upload UI, infiltration timeline, MITRE tracker, explainability
│
├── backend/                   # FastAPI service — the API the frontend calls
│   ├── app/
│   │   ├── main.py            # App entrypoint, CORS, router registration
│   │   ├── routes/
│   │   │   ├── upload.py      # POST /api/upload
│   │   │   ├── prediction.py  # GET  /api/prediction/{session_id}
│   │   │   └── analysis.py    # GET  /api/analysis/{session_id}
│   │   ├── services/
│   │   │   ├── pcap_service.py        # PCAP/CSV parsing + feature extraction + windowing
│   │   │   ├── prediction_service.py  # Model rollout OR heuristic fallback scoring
│   │   │   ├── report_service.py      # Dashboard summary stats
│   │   │   └── store.py               # In-memory session store
│   │   └── schemas/schemas.py         # Pydantic request/response models
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── cybersecurity/              # Standalone feature-extraction + MITRE mapping layer
│   ├── pcap/parser.py                  # re-exports pcap_service parsing
│   ├── flow/flow_extractor.py          # re-exports windowing logic
│   ├── features/                       # flow/packet feature schemas + normalisation
│   └── attack_mapping/mitre_mapping.py # MITRE ATT&CK tactic IDs + descriptions
│
├── ml/                          # The actual world model
│   ├── models/world_model.py           # LSTM + attention + transition/infiltration/stage heads
│   ├── training/
│   │   ├── train.py                    # Supervised dynamics-learning training loop
│   │   ├── dataset.py                  # Sliding-window sequence dataset
│   │   └── config.yaml                 # Hyperparameters
│   ├── inference/
│   │   ├── predict.py                  # CLI inference (no server needed)
│   │   └── rollout.py                  # K-step forward simulation
│   ├── explainability/shap.py          # SHAP-based feature attribution (model mode)
│   ├── baseline/logistic_regression.py # Static classifier used for benchmarking
│   └── evaluation/
│       ├── metrics.py                  # F1 / precision / recall / FPR
│       └── evaluate.py                 # World model vs baseline benchmark script
│
├── data/                       # raw/ (datasets) and processed/ (cached features) — gitignored
├── models/                     # Trained weights land here: world_model.pt, baseline.pkl
├── notebooks/                  # Exploration / feature analysis / evaluation notebooks
├── docs/
│   └── demo.md                 # ≤2-minute demo script
├── tests/                      # pytest suite (features, model, API)
├── docker-compose.yml          # backend + static frontend together
└── README.md                   # you are here
```

---

## 4. How it actually works — the full flow

### Step 1 — Ingest (`/api/upload`)
You upload a **CIC-IDS-2018 / CTU-13 style CSV** flow export, or a raw
**`.pcap` / `.pcapng`** capture.

- CSV path (`pcap_service.parse_csv`): reads the file with pandas, maps
  common column-name variants (e.g. `Destination Port`, `SYN Flag Count`)
  onto a canonical schema, and derives proxy packet-level features (TTL
  variance, window size, retransmit rate) since flow-only datasets don't
  carry raw packets.
- PCAP path (`pcap_service.parse_pcap`, needs `scapy`): reads every packet,
  groups into 5-tuple flows, and computes both flow-level aggregates *and*
  real packet-level features (TTL per packet, TCP window size, flag
  sequencing).

Either way you end up with one row per flow across ~20 flow-level + 5
packet-level features (`FLOW_FEATURES` / `PACKET_FEATURES` in
`pcap_service.py`).

### Step 2 — Window into network states (`build_time_windows`)
Flows are grouped into fixed **5-second windows**. Each window is averaged
into a single **state vector `S_t`** — this is the "network state" the
world model actually reasons about, not individual flows. A session
becomes a sequence `S_0, S_1, S_2, …, S_n`.

### Step 3 — Forecast (`/api/prediction/{session_id}`)
The world model (`ml/models/world_model.py`) is an **LSTM encoder with
additive attention**:

- Feeds the last 6 windows in as a sequence.
- The **transition head** predicts `S_t+1` (the actual dynamics-learning
  part — not a classifier).
- The **infiltration head** outputs a probability that the trajectory is
  heading toward compromise.
- The **stage head** classifies which MITRE ATT&CK phase the trajectory is
  converging to (Reconnaissance → Initial Access → Lateral Movement →
  Command & Control → Exfiltration).
- `model.rollout()` feeds its own predicted state back in and repeats this
  **K = 5 times**, producing a forecast timeline instead of a single number.
- The **attention weights** double as explainability: they show which past
  windows mattered most for the current prediction.

**No trained checkpoint yet?** The API doesn't break — `prediction_service.py`
falls back to a **transparent heuristic scorer** using the same feature set
(SYN count, unique destination ports, retransmit rate, etc., each with an
explicit, documented weight — see `HEURISTIC_WEIGHTS`). This means the
entire pipeline, UI, and demo are usable **before training finishes**. Drop
trained weights at `models/world_model.pt` and the API automatically
switches to real LSTM rollout — no code changes required. Both modes return
the exact same response shape.

### Step 4 — Explain
Every prediction ships with **top contributing features** — attention-based
when a trained model is loaded, weighted-contribution-based in heuristic
mode. Never a black-box number with no reasoning attached (this was a hard
requirement in the brief).

### Step 5 — Summarise (`/api/analysis/{session_id}`)
Aggregates the whole session: total flows/packets, top talkers, top ports,
protocol breakdown, MITRE stage distribution across the timeline, and the
top 10 highest-risk flows.

### Step 6 — Visualise (frontend)
`frontend/index.html` is a single self-contained, fully responsive
dashboard (no build step) that:
1. Lets you drag-and-drop a file onto `/api/upload`.
2. Polls `/api/prediction/{id}` and `/api/analysis/{id}`.
3. Renders: KPI cards → infiltration-probability timeline chart (observed +
   forecast, visually distinguished) → MITRE stage tracker → explainability
   bar chart → flagged-flows table → world-model-vs-baseline benchmark bars.

---

## 5. Setup — step by step

### Prerequisites
- Python 3.10+
- (optional, for raw PCAP support) `scapy`
- (optional, for real model inference instead of heuristic mode) `torch`

### 5.1 Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd ..                                # run from the REPO ROOT so backend.*/ml.* imports resolve
uvicorn backend.app.main:app --reload --port 8000
```

Check it's alive: open `http://localhost:8000/docs` (interactive Swagger UI)
or `http://localhost:8000/api/health`.

### 5.2 Frontend

No build step — it's one HTML file.

```bash
cd frontend
python -m http.server 3000
# open http://localhost:3000
```

If your backend runs somewhere other than `localhost:8000`, edit the
`API_BASE` constant at the top of the `<script>` block in `index.html`.

### 5.3 Docker (both together)

```bash
docker compose up --build
# backend  → http://localhost:8000
# frontend → http://localhost:3000
```

---

## 6. Demo walkthrough (what to actually click)

1. Open the frontend. You'll see the connection badge confirm the backend
   is reachable, and an empty state below.
2. Drag in a flow CSV (get one from CIC-IDS-2018 / CTU-13, or use a slice
   containing a port scan for a good demo — see `data/README.md`) or a
   `.pcap` file.
3. Watch the upload progress bar → the dashboard populates:
   - **KPI row**: current infiltration probability, predicted stage, flow
     count, unique IP pairs.
   - **Timeline chart**: solid teal line = observed windows, dashed amber
     = the K-step forecast.
   - **MITRE stage tracker**: lights up whichever ATT&CK phases appeared
     across the timeline.
   - **Explainability panel**: horizontal bars showing which features
     (SYN count, unique dst ports, etc.) are driving the prediction, signed
     so you can see what's pushing risk up vs down.
   - **Flagged flows table**: highest-risk individual flows, colour-coded.
   - **Benchmark panel**: world model vs logistic-regression baseline on
     F1 / precision / recall / false-positive rate.
4. Drop a different file to re-run the whole pipeline on a new session.

Scripted 2-minute version: `docs/demo.md`.

---

## 7. API reference

| Endpoint | Method | Body / Params | Returns |
|---|---|---|---|
| `/api/health` | GET | — | `{status, service}` liveness check |
| `/api/upload` | POST | multipart file (`.csv`/`.pcap`) | `session_id`, flow/window counts |
| `/api/prediction/{session_id}` | GET | — | timeline, current probability + stage, top drivers, flagged flows, baseline comparison |
| `/api/analysis/{session_id}` | GET | — | flow/packet counts, top ports, protocol breakdown, stage distribution, flagged flows, metrics |

Full request/response schemas: `backend/app/schemas/schemas.py`, or just
open `/docs` on the running backend for interactive Swagger.

---

## 8. Training your own world model

```bash
# from repo root
python -m ml.training.train --csv data/raw/cic-ids-2018-friday.csv --epochs 20
```

- Ground-truth stage labels are derived from the dataset's own `Label`
  column via `ml/training/train.py::label_to_stage_index` (maps things like
  `PortScan` → Reconnaissance, `DDoS`/`Infiltration` → Command & Control,
  etc. — adjust the keyword rules for your specific dataset's label set).
- Trains the LSTM on three joint losses: next-state MSE (the actual world
  model / dynamics-learning objective), infiltration BCE, and stage
  cross-entropy.
- Saves a checkpoint to `models/world_model.pt`. The backend picks this up
  automatically on next request — no restart-time config needed beyond the
  file existing at that path (or set `WORLD_MODEL_PATH` env var).

Benchmark against the required baseline:

```bash
python -m ml.evaluation.evaluate --csv data/raw/cic-ids-2018-friday.csv
```

Prints F1 / precision / recall / false-positive rate for the logistic
regression baseline on a held-out split — compare directly against the
world model's rollout metrics from the same split.

---

## 9. MITRE ATT&CK stage mapping

| Predicted stage | Tactic ID | Signal it's driven by |
|---|---|---|
| Reconnaissance | TA0043 | wide fan-out across few ports/hosts, low per-connection depth |
| Initial Access | TA0001 | exploitation/brute-force bursts against an exposed service |
| Lateral Movement | TA0008 | rising internal host-to-host connections |
| Command & Control | TA0011 | persistent low-volume beaconing to few external destinations |
| Exfiltration | TA0010 | sustained high-throughput outbound transfer |

Full descriptions: `cybersecurity/attack_mapping/mitre_mapping.py`.

---

## 10. Testing

```bash
pip install pytest httpx
pytest tests/
```

Covers: feature-window shape sanity (`test_features.py`), world model
forward-pass shapes when torch is available (`test_model.py`, auto-skips
otherwise), and the `/api/health` endpoint (`test_api.py`).

---

## 11. Hackathon deliverables checklist

| Required | Where it is |
|---|---|
| Source code (GitHub/Drive) | this repo |
| README with setup instructions | this file, §5 |
| Architecture document (≤2 pages) | write from §2–§4 above, or ask for a formatted doc |
| Demo video (≤2 min) | script in `docs/demo.md` |
| Technical presentation (≤5 slides) | ask for a slide deck generated from this README |
| Feature extraction pipeline (flow + packet) | `backend/app/services/pcap_service.py` |
| Trained world model + training config | `ml/models/world_model.py`, `ml/training/` |
| K-step infiltration prediction engine | `ml/models/world_model.py::rollout`, `prediction_service.py` |
| Explainability output | attention weights (model mode) / weighted scorer (heuristic mode) |
| Working demo interface, offline | `frontend/index.html` + local FastAPI backend |
| Benchmark vs logistic regression baseline | `ml/baseline/`, `ml/evaluation/evaluate.py` |

---

## 12. Known limitations / what's left before a full production run

- **No checkpoint is trained yet** in this repo — the system runs in
  transparent heuristic mode by default. Run `ml/training/train.py` against
  a real labelled dataset to get real LSTM inference.
- PCAP parsing needs `scapy` installed; CSV always works with no extra deps.
- Session storage is in-memory (`store.py`) — fine for a hackathon demo,
  swap for Redis/Postgres for anything persistent.
- `label_to_stage_index` keyword rules are dataset-specific — tune them to
  whatever labelling scheme your chosen CIC-IDS-2018/CTU-13 slice uses.

---

## License

MIT (or your team's choice — update `LICENSE`).
