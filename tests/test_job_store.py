# -*- coding: utf-8 -*-
"""
Tests for the SQLite-backed job store.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.api.job_store import JobStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_jobs.db"
    return JobStore(db_path=db_path)


class TestJobStore:
    def test_set_and_get(self, store):
        job = {"job_id": "j1", "status": "queued", "pid": "test_pid"}
        store["j1"] = job
        retrieved = store["j1"]
        assert retrieved["job_id"] == "j1"
        assert retrieved["status"] == "queued"

    def test_contains(self, store):
        store["j1"] = {"status": "queued"}
        assert "j1" in store
        assert "j2" not in store

    def test_get_missing(self, store):
        with pytest.raises(KeyError):
            _ = store["nonexistent"]

    def test_get_default(self, store):
        assert store.get("missing") is None
        assert store.get("missing", {"default": True}) == {"default": True}

    def test_update(self, store):
        store["j1"] = {"status": "queued", "count": 0}
        store["j1"] = {"status": "running", "count": 1}
        job = store["j1"]
        assert job["status"] == "running"
        assert job["count"] == 1

    def test_delete(self, store):
        store["j1"] = {"status": "queued"}
        del store["j1"]
        assert "j1" not in store

    def test_len(self, store):
        assert len(store) == 0
        store["j1"] = {"status": "queued"}
        store["j2"] = {"status": "running"}
        assert len(store) == 2

    def test_iter(self, store):
        store["j1"] = {"status": "queued"}
        store["j2"] = {"status": "running"}
        ids = list(store)
        assert "j1" in ids
        assert "j2" in ids

    def test_update_field(self, store):
        store["j1"] = {"status": "queued", "email": ""}
        store.update_field("j1", "email", "user@test.com")
        assert store["j1"]["email"] == "user@test.com"

    def test_update_step(self, store):
        store["j1"] = {
            "status": "running",
            "steps": [
                {"name": "step1", "status": "pending"},
                {"name": "step2", "status": "pending"},
            ],
        }
        store.update_step("j1", 0, {"status": "done"})
        job = store["j1"]
        assert job["steps"][0]["status"] == "done"
        assert job["steps"][1]["status"] == "pending"

    def test_list_jobs(self, store):
        store["j1"] = {"status": "queued"}
        store["j2"] = {"status": "running"}
        store["j3"] = {"status": "done"}
        jobs = store.list_jobs(limit=2)
        assert len(jobs) == 2

    def test_persistence(self, tmp_path):
        db_path = tmp_path / "persist.db"

        store1 = JobStore(db_path=db_path)
        store1["j1"] = {"status": "done", "result": "success"}

        store2 = JobStore(db_path=db_path)
        assert "j1" in store2
        assert store2["j1"]["status"] == "done"

    def test_complex_data(self, store):
        job = {
            "job_id": "j1",
            "pid": "test_proposal_abc123",
            "status": "running",
            "steps": [
                {"name": "prepare", "status": "done", "started_at": "2024-01-01"},
                {"name": "extract", "status": "running", "started_at": "2024-01-01"},
            ],
            "error": None,
            "report_path": "/path/to/report.md",
            "email": "test@example.com",
        }
        store["j1"] = job
        retrieved = store["j1"]
        assert retrieved["steps"][0]["status"] == "done"
        assert retrieved["report_path"] == "/path/to/report.md"
