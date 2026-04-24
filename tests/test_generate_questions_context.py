# -*- coding: utf-8 -*-
"""Tests for generate_questions payload context helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.generate_questions import _payload_suggests_healthcare_context


def test_healthcare_context_detects_clinical_zh():
    assert _payload_suggests_healthcare_context(
        {
            "summary": "",
            "key_points": ["计划开展三期临床试验，入组约 200 例患者"],
            "risks": [],
            "mitigations": [],
        }
    )


def test_healthcare_context_detects_en():
    assert _payload_suggests_healthcare_context(
        {
            "summary": "Phase 3 clinical trial design",
            "key_points": [],
            "risks": [],
            "mitigations": [],
        }
    )


def test_healthcare_context_false_for_general_commercial():
    assert not _payload_suggests_healthcare_context(
        {
            "summary": "未来三年营收与毛利目标及区域渠道扩张",
            "key_points": ["华东区新增门店 50 家", "与头部商超签订年度框架协议"],
            "risks": ["原材料价格波动"],
            "mitigations": [],
        }
    )


def test_healthcare_context_false_when_empty():
    assert not _payload_suggests_healthcare_context(
        {"summary": "", "key_points": [], "risks": [], "mitigations": []}
    )
