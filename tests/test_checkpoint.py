# -*- coding: utf-8 -*-
"""Tests for the checkpoint/caching module."""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.checkpoint import PipelineCheckpoint

TEST_PID = "__test_checkpoint_pid__"


def _cleanup():
    base = Path(__file__).resolve().parents[1] / "src" / "data" / "checkpoints" / TEST_PID
    shutil.rmtree(base, ignore_errors=True)


def test_initial_state():
    _cleanup()
    ckpt = PipelineCheckpoint(TEST_PID)
    assert ckpt.get_last_completed_stage() is None
    assert ckpt.get_next_stage() == "prepare_proposal_text"
    _cleanup()


def test_mark_complete():
    _cleanup()
    ckpt = PipelineCheckpoint(TEST_PID)
    ckpt.mark_complete("prepare_proposal_text")
    assert ckpt.is_stage_complete("prepare_proposal_text")
    assert not ckpt.is_stage_complete("extract_facts_by_chunk")
    assert ckpt.get_next_stage() == "extract_facts_by_chunk"
    _cleanup()


def test_mark_multiple_complete():
    _cleanup()
    ckpt = PipelineCheckpoint(TEST_PID)
    ckpt.mark_complete("prepare_proposal_text")
    ckpt.mark_complete("extract_facts_by_chunk")
    ckpt.mark_complete("verify_facts")
    assert ckpt.get_last_completed_stage() == "verify_facts"
    assert ckpt.get_next_stage() == "build_dimensions_from_facts"
    _cleanup()


def test_invalidate_from():
    _cleanup()
    ckpt = PipelineCheckpoint(TEST_PID)
    for stage in PipelineCheckpoint.STAGE_ORDER[:5]:
        ckpt.mark_complete(stage)
    assert ckpt.is_stage_complete("generate_questions")

    ckpt.invalidate_from("verify_facts")
    assert ckpt.is_stage_complete("extract_facts_by_chunk")
    assert not ckpt.is_stage_complete("verify_facts")
    assert not ckpt.is_stage_complete("build_dimensions_from_facts")
    _cleanup()


def test_persistence():
    _cleanup()
    ckpt1 = PipelineCheckpoint(TEST_PID)
    ckpt1.mark_complete("prepare_proposal_text", {"pages": 10})

    ckpt2 = PipelineCheckpoint(TEST_PID)
    assert ckpt2.is_stage_complete("prepare_proposal_text")
    _cleanup()


def test_summary():
    _cleanup()
    ckpt = PipelineCheckpoint(TEST_PID)
    ckpt.mark_complete("prepare_proposal_text")
    summary = ckpt.summary()
    assert summary["proposal_id"] == TEST_PID
    assert summary["last_completed"] == "prepare_proposal_text"
    assert summary["next_stage"] == "extract_facts_by_chunk"
    assert "stages" in summary
    _cleanup()


def test_reset():
    _cleanup()
    ckpt = PipelineCheckpoint(TEST_PID)
    ckpt.mark_complete("prepare_proposal_text")
    ckpt.reset()
    assert not ckpt.is_stage_complete("prepare_proposal_text")
    assert ckpt.get_next_stage() == "prepare_proposal_text"
    _cleanup()


def test_all_stages_complete():
    _cleanup()
    ckpt = PipelineCheckpoint(TEST_PID)
    for stage in PipelineCheckpoint.STAGE_ORDER:
        ckpt.mark_complete(stage)
    assert ckpt.get_next_stage() is None
    assert ckpt.get_last_completed_stage() == "generate_final_report"
    _cleanup()


if __name__ == "__main__":
    tests = [
        test_initial_state,
        test_mark_complete,
        test_mark_multiple_complete,
        test_invalidate_from,
        test_persistence,
        test_summary,
        test_reset,
        test_all_stages_complete,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
