# -*- coding: utf-8 -*-
"""Tests for the fact verification module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.verify_facts import (
    extract_numbers,
    extract_key_entities,
    compute_text_overlap,
    compute_entity_coverage,
    compute_numeric_accuracy,
    verify_single_fact,
    _expand_cn_amount_tokens,
)


def test_extract_numbers():
    nums = extract_numbers("The market is worth $2.5 billion with 15% CAGR")
    assert "2.5" in nums
    assert "15%" in nums or "15" in nums


def test_extract_key_entities():
    entities = extract_key_entities("Dr. John Smith at Harvard University")
    # Should find multi-word capitalized names
    assert any("John Smith" in e for e in entities)
    assert any("Harvard University" in e for e in entities)


def test_compute_text_overlap_high():
    fact = "The team has expertise in AI and drug discovery"
    source = "Our team has deep expertise in AI and drug discovery research"
    overlap = compute_text_overlap(fact, source)
    assert overlap > 0.5


def test_compute_text_overlap_low():
    fact = "The team focuses on quantum computing"
    source = "Our project addresses climate change solutions"
    overlap = compute_text_overlap(fact, source)
    assert overlap < 0.4


def test_entity_coverage_all_present():
    fact = "Harvard University and MIT collaborate on this project"
    source = "Harvard University and MIT announced a new collaboration on this research project"
    coverage, missing = compute_entity_coverage(fact, source)
    assert coverage > 0.5
    assert len(missing) <= 1


def test_entity_coverage_fabricated():
    fact = "Stanford University leads the initiative"
    source = "The project is led by Harvard University"
    coverage, missing = compute_entity_coverage(fact, source)
    assert coverage < 1.0


def test_numeric_accuracy_all_match():
    fact = "The budget is 2.5 million with 30% allocated to R&D"
    source = "Total budget: 2.5 million USD. R&D allocation: 30%"
    accuracy, suspect = compute_numeric_accuracy(fact, source)
    assert accuracy >= 0.8


def test_numeric_accuracy_fabricated():
    fact = "The market size is 50 billion"
    source = "The market analysis shows growth in this sector"
    accuracy, suspect = compute_numeric_accuracy(fact, source)
    assert accuracy < 1.0
    assert suspect is True


def test_numeric_accuracy_wan_vs_arabic():
    """中文「万」与阿拉伯金额在原文中混写时仍应判为匹配。"""
    fact = "本轮融资额为3000万元，对应投后估值约3亿元。"
    source = "公司披露融资30000000元，投后估值300000000人民币。"
    accuracy, suspect = compute_numeric_accuracy(fact, source)
    assert accuracy >= 0.5


def test_expand_cn_amount_tokens():
    assert "30000000" in _expand_cn_amount_tokens("融资3000万元")
    assert "300000000" in _expand_cn_amount_tokens("估值3亿元")


def test_verify_single_fact_verified():
    fact = {
        "text": "The team includes AI experts with 20 years experience",
        "meta": {"char_start": 0, "char_end": 100},
    }
    full_text = "The team includes AI experts with 20 years of experience in machine learning and drug discovery"
    result = verify_single_fact(fact, full_text)
    assert "verification" in result
    assert result["verification"]["status"] in ("verified", "partially_verified")
    assert result["verification"]["score"] > 0.3


def test_verify_single_fact_unverified():
    fact = {
        "text": "Microsoft and Google jointly invested $500 million in this project",
        "meta": {"char_start": 0, "char_end": 50},
    }
    full_text = "The project received initial seed funding from local investors"
    result = verify_single_fact(fact, full_text)
    assert "verification" in result
    assert result["verification"]["score"] < 0.7


def test_verify_empty_fact():
    fact = {"text": "", "meta": {}}
    result = verify_single_fact(fact, "some source text")
    assert result["verification"]["status"] == "unverified"


def test_verify_no_source():
    fact = {
        "text": "Important finding about the project",
        "meta": {},
    }
    result = verify_single_fact(fact, "")
    assert result["verification"]["status"] == "unverified"


if __name__ == "__main__":
    tests = [
        test_extract_numbers,
        test_extract_key_entities,
        test_compute_text_overlap_high,
        test_compute_text_overlap_low,
        test_entity_coverage_all_present,
        test_entity_coverage_fabricated,
        test_numeric_accuracy_all_match,
        test_numeric_accuracy_fabricated,
        test_verify_single_fact_verified,
        test_verify_single_fact_unverified,
        test_verify_empty_fact,
        test_verify_no_source,
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
