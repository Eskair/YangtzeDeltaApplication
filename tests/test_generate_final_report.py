# -*- coding: utf-8 -*-
"""generate_final_report：一页摘要与 metrics 来源说明。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.generate_final_report import build_executive_summary


def _minimal_expert(overall_score_echo=0.75, confidence_echo=0.70, raw_s=0.50, raw_c=0.48):
    return {
        "overall_opinion": {
            "verdict": "HOLD",
            "overall_score_echo": overall_score_echo,
            "confidence_echo": confidence_echo,
            "overall_score_raw_echo": raw_s,
            "confidence_raw_echo": raw_c,
            "key_strengths": [],
            "key_risks": [],
            "recommendations": [],
            "basis": ["依据"],
        }
    }


def test_executive_summary_raw_suffix_when_metrics_missing():
    md = build_executive_summary(
        _minimal_expert(),
        None,
        metrics_json_loaded=False,
    )
    assert "未加载 `postproc/metrics.json`" in md
    assert "ai_expert_opinion.json" in md
    assert "与 `postproc/metrics.json` 中 `overall_*_raw` 一致" not in md


def test_executive_summary_raw_suffix_aligns_metrics_when_present():
    metrics = {
        "overall": {
            "overall_score_raw": 0.51,
            "overall_confidence_raw": 0.49,
        }
    }
    md = build_executive_summary(
        _minimal_expert(),
        metrics,
        metrics_json_loaded=True,
    )
    assert "postproc/metrics.json" in md
    assert "0.51" in md
    assert "0.49" in md


def test_executive_summary_placeholder_when_expert_json_empty():
    md = build_executive_summary({}, None, metrics_json_loaded=False, pid="TestPid")
    assert "## 0. 一页摘要" in md
    assert "expert_reports/TestPid/ai_expert_opinion.json" in md
    assert "ai_expert_opinion.md" in md
    assert "final_payload.json" in md


def test_executive_summary_metrics_file_but_no_raw_fields():
    md = build_executive_summary(
        _minimal_expert(0.8, 0.7, 0.55, 0.52),
        {"overall": {"overall_score": 0.8}},
        metrics_json_loaded=True,
    )
    assert "已加载但未包含" in md
    assert "与 `ai_expert_opinion.json` 回显字段一致" in md
