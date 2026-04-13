# -*- coding: utf-8 -*-
"""
Pipeline Checkpoint / Caching Mechanism (checkpoint.py)
-------------------------------------------------------
Provides resumability for the multi-stage pipeline by:
  1. Recording which stages have completed successfully
  2. Storing intermediate outputs with content hashes
  3. Allowing the pipeline to skip already-completed stages on restart

Usage:
    from src.tools.checkpoint import PipelineCheckpoint

    ckpt = PipelineCheckpoint(proposal_id="my_proposal")

    if ckpt.is_stage_complete("extract_facts_by_chunk"):
        print("Skipping fact extraction - already done")
    else:
        run_extraction(...)
        ckpt.mark_complete("extract_facts_by_chunk", {"facts_count": 42})

    # Get checkpoint summary
    print(ckpt.summary())

Storage:
    src/data/checkpoints/<proposal_id>/checkpoint.json
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = BASE_DIR / "src" / "data" / "checkpoints"


class PipelineCheckpoint:
    """Manages checkpoint state for a single proposal's pipeline execution."""

    STAGE_ORDER = [
        "prepare_proposal_text",
        "extract_facts_by_chunk",
        "verify_facts",
        "build_dimensions_from_facts",
        "generate_questions",
        "llm_answering",
        "post_processing",
        "ai_expert_opinion",
        "generate_final_report",
    ]

    def __init__(self, proposal_id: str):
        self.proposal_id = proposal_id
        self._dir = CHECKPOINT_DIR / proposal_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "checkpoint.json"
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "proposal_id": self.proposal_id,
            "stages": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
        }

    def _save(self) -> None:
        self._state["last_updated"] = datetime.now().isoformat()
        self._path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage has been completed successfully."""
        stage_info = self._state.get("stages", {}).get(stage_name, {})
        return stage_info.get("status") == "complete"

    def mark_complete(
        self,
        stage_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        output_files: Optional[List[str]] = None,
    ) -> None:
        """Mark a stage as complete with optional metadata."""
        if "stages" not in self._state:
            self._state["stages"] = {}

        file_hashes = {}
        if output_files:
            for fp in output_files:
                p = Path(fp)
                if p.exists():
                    file_hashes[str(p)] = self._file_hash(p)

        self._state["stages"][stage_name] = {
            "status": "complete",
            "completed_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "output_file_hashes": file_hashes,
        }
        self._save()

    def mark_failed(self, stage_name: str, error: str = "") -> None:
        """Mark a stage as failed."""
        if "stages" not in self._state:
            self._state["stages"] = {}

        self._state["stages"][stage_name] = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": error,
        }
        self._save()

    def invalidate_from(self, stage_name: str) -> None:
        """
        Invalidate this stage and all subsequent stages.
        Useful when an earlier stage is re-run, making downstream results stale.
        """
        if stage_name not in self.STAGE_ORDER:
            return

        idx = self.STAGE_ORDER.index(stage_name)
        for s in self.STAGE_ORDER[idx:]:
            if s in self._state.get("stages", {}):
                del self._state["stages"][s]
        self._save()

    def get_last_completed_stage(self) -> Optional[str]:
        """Return the name of the last successfully completed stage."""
        last = None
        for stage in self.STAGE_ORDER:
            if self.is_stage_complete(stage):
                last = stage
            else:
                break
        return last

    def get_next_stage(self) -> Optional[str]:
        """Return the name of the next stage to run."""
        for stage in self.STAGE_ORDER:
            if not self.is_stage_complete(stage):
                return stage
        return None

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of checkpoint state."""
        stages_summary = {}
        for stage in self.STAGE_ORDER:
            info = self._state.get("stages", {}).get(stage, {})
            status = info.get("status", "pending")
            stages_summary[stage] = {
                "status": status,
                "completed_at": info.get("completed_at"),
            }

        return {
            "proposal_id": self.proposal_id,
            "last_completed": self.get_last_completed_stage(),
            "next_stage": self.get_next_stage(),
            "stages": stages_summary,
        }

    def reset(self) -> None:
        """Reset all checkpoint state."""
        self._state = {
            "proposal_id": self.proposal_id,
            "stages": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
        }
        self._save()

    def verify_outputs(self, stage_name: str) -> bool:
        """
        Verify that output files from a completed stage still exist
        and haven't been modified (by checking file hashes).
        """
        stage_info = self._state.get("stages", {}).get(stage_name, {})
        if stage_info.get("status") != "complete":
            return False

        file_hashes = stage_info.get("output_file_hashes", {})
        if not file_hashes:
            return True

        for fp, expected_hash in file_hashes.items():
            p = Path(fp)
            if not p.exists():
                return False
            if self._file_hash(p) != expected_hash:
                return False

        return True

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Compute SHA256 hash of a file (first 64KB for large files)."""
        h = hashlib.sha256()
        try:
            with path.open("rb") as f:
                chunk = f.read(65536)
                h.update(chunk)
        except IOError:
            return ""
        return h.hexdigest()[:16]
