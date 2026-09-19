# Backend — Cyber World Model API

FastAPI service that ingests flow CSVs / PCAP captures, windows them into
network-state sequences, and serves infiltration predictions + analysis.

Runs today in **heuristic mode** (transparent, documented weighted
scorer — see `app/services/prediction_service.py`). It will automatically
switch to real LSTM rollout once `ml/` ships a trained checkpoint at
`models/world_model.pt` — no code changes needed here.

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd ..                                # repo root, so `app.*` imports resolve
uvicorn backend.app.main:app --reload --port 8000 --app-dir backend
```

Or, if you prefer running from inside `backend/`:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Check it's alive: `http://localhost:8000/docs` (Swagger UI) or
`http://localhost:8000/api/health`.

## Layout

```
app/
├── main.py               # FastAPI app, CORS, router registration
├── routes/
│   ├── upload.py          # POST /api/upload
│   ├── prediction.py      # GET  /api/prediction/{session_id}
│   └── analysis.py        # GET  /api/analysis/{session_id}
├── services/
│   ├── pcap_service.py         # CSV/PCAP parsing, feature schema, windowing
│   ├── prediction_service.py   # heuristic scorer + rollout forecast + explainability
│   ├── report_service.py       # session-wide summary stats
│   └── store.py                 # in-memory session store
└── schemas/schemas.py     # Pydantic request/response models
```

## Notes

- `store.py` is in-memory and single-process — restarting the server drops
  all sessions. Fine for a demo; swap for Redis/Postgres for anything else.
- PCAP parsing needs `scapy` (`pip install scapy`). CSV works with no extra deps.
- See the repo-root `README.md` §9 for the MITRE ATT&CK stage mapping the
  heuristic scorer uses, and §12 for known limitations.
