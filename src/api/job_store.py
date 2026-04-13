# -*- coding: utf-8 -*-
"""
Persistent job store backed by SQLite.

Replaces the in-memory dict so that job state survives server restarts.
The public API mimics a dict: store[job_id] = job_dict, job = store[job_id].
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Iterator


_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "src" / "data" / "jobs.db"


class JobStore:
    """Thread-safe SQLite-backed job store."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, timeout=10)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                data   TEXT NOT NULL
            )
        """)
        conn.commit()

    def __setitem__(self, job_id: str, job: Dict[str, Any]):
        conn = self._get_conn()
        data = json.dumps(job, ensure_ascii=False)
        conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, data) VALUES (?, ?)",
            (job_id, data),
        )
        conn.commit()

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def __contains__(self, job_id: str) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return row is not None

    def get(self, job_id: str, default: Any = None) -> Any:
        try:
            return self[job_id]
        except KeyError:
            return default

    def __delitem__(self, job_id: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()

    def __iter__(self) -> Iterator[str]:
        conn = self._get_conn()
        for row in conn.execute("SELECT job_id FROM jobs"):
            yield row[0]

    def __len__(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return row[0] if row else 0

    def update_field(self, job_id: str, field: str, value: Any):
        """Update a single field in the job dict without a full read-modify-write."""
        job = self[job_id]
        job[field] = value
        self[job_id] = job

    def update_step(self, job_id: str, step_index: int, updates: Dict[str, Any]):
        """Update fields in a specific step of the job."""
        job = self[job_id]
        for k, v in updates.items():
            job["steps"][step_index][k] = v
        self[job_id] = job

    def list_jobs(self, limit: int = 50) -> list:
        """Return recent jobs (most recent first) as a list of dicts."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT data FROM jobs ORDER BY rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(r[0]) for r in rows]
