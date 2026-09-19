# 2-minute demo script (placeholder)

This will be filled in once `ml/` ships a trained checkpoint and
`cybersecurity/` is in place — for now, the frontend + backend can be
demoed on their own in heuristic mode:

1. Start the backend (`uvicorn app.main:app --reload --port 8000` from `backend/`).
2. Start the frontend (`python -m http.server 3000` from `frontend/`).
3. Open `http://localhost:3000`, confirm the "backend live" badge.
4. Drop a CIC-IDS-2018/CTU-13 CSV slice containing a port scan or flood.
5. Walk through: KPI row → timeline chart → MITRE tracker → explainability
   panel → flagged flows → benchmark panel.
6. Mention it's running in heuristic mode today and will switch to the
   trained LSTM automatically once `models/world_model.pt` exists.
