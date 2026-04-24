# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.llm_answering import dedup_nearby


def test_dedup_keeps_distinct_variants_same_opening():
    """不同 variant_id 时，即使前段相同也不应合并为一条。"""
    body_a = "- 要点一\n- 要点二\n- 要点三"
    cands = [
        {"variant_id": "default", "answer": body_a},
        {"variant_id": "risk", "answer": body_a},
        {"variant_id": "implementation", "answer": body_a},
    ]
    out = dedup_nearby(cands)
    assert len(out) == 3


def test_dedup_same_variant_identical_answer_once():
    cands = [
        {"variant_id": "default", "answer": "相同正文。"},
        {"variant_id": "default", "answer": "相同正文。"},
    ]
    out = dedup_nearby(cands)
    assert len(out) == 1


def test_dedup_same_variant_different_tail_keeps_both():
    """全文指纹：前 256 字可相同，尾部不同则保留两条。"""
    prefix = "段落开头" * 90
    cands = [
        {"variant_id": "default", "answer": prefix + "\n尾部甲"},
        {"variant_id": "default", "answer": prefix + "\n尾部乙"},
    ]
    out = dedup_nearby(cands)
    assert len(out) == 2
