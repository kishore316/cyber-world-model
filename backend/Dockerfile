FROM python:3.11-slim

WORKDIR /repo

# System deps for scapy's optional pcap support (safe to keep even if
# scapy install is skipped later).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
# ml/ and cybersecurity/ are copied too (once they exist) so the backend's
# optional model-mode import path resolves inside the container.
COPY ml ml
COPY cybersecurity cybersecurity
COPY models models

ENV PYTHONPATH=/repo/backend:/repo
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
