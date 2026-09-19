"""
In-memory session store.

Fine for a hackathon demo / single-process dev server. Swap this module
for a Redis- or Postgres-backed implementation for anything persistent —
every other module only calls `get`, `put`, and `exists`, so the storage
backend can change without touching routes or services.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SessionData:
    session_id: str
    filename: str
    source_type: str
    created_at: float
    flows: pd.DataFrame
    windows: List[Dict[str, Any]]
    window_seconds: int
    warnings: List[str] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def create(
        self,
        filename: str,
        source_type: str,
        flows: pd.DataFrame,
        windows: List[Dict[str, Any]],
        window_seconds: int,
        warnings: Optional[List[str]] = None,
    ) -> SessionData:
        session_id = uuid.uuid4().hex[:12]
        data = SessionData(
            session_id=session_id,
            filename=filename,
            source_type=source_type,
            created_at=time.time(),
            flows=flows,
            windows=windows,
            window_seconds=window_seconds,
            warnings=warnings or [],
        )
        with self._lock:
            self._sessions[session_id] = data
        return data

    def get(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            return self._sessions.get(session_id)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


# Module-level singleton — imported by routes/services.
store = SessionStore()
