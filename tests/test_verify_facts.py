# -*- coding: utf-8 -*-
"""
Tests for verify_facts.py — the fact verification module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.tools.verify_facts import (
    _extract_entities,
    _extract_numbers,
    _fuzzy_find,
    verify_single_fact,
)


class TestExtractEntities:
    def test_english_entities(self):
        text = "The FDA approved Pfizer's mRNA vaccine for COVID-19."
        entities = _extract_entities(text)
        assert any("FDA" in e for e in entities)
        assert any("Pfizer" in e for e in entities)

    def test_chinese_entities(self):
        text = "复旦大学附属中山医院的研究团队开发了新型AI诊断平台"
        entities = _extract_entities(text)
        assert any("中山医院" in e for e in entities)

    def test_numbers(self):
        text = "The market size is $2.5 billion with CAGR of 12.3%"
        entities = _extract_entities(text)
        assert any("2.5" in e or "12.3" in e for e in entities)

    def test_acronyms(self):
        text = "GMP and GCP compliance is required by NMPA"
        entities = _extract_entities(text)
        assert any("GMP" in e for e in entities)
        assert any("GCP" in e for e in entities)
        assert any("NMPA" in e for e in entities)


class TestExtractNumbers:
    def test_percentages(self):
        nums = _extract_numbers("Growth rate is 15.2% annually")
        assert any("15.2" in n for n in nums)

    def test_currency(self):
        nums = _extract_numbers("Budget of $3,500,000")
        assert len(nums) > 0

    def test_chinese_units(self):
        nums = _extract_numbers("市场规模约50亿元")
        assert any("50" in n for n in nums)

    def test_no_numbers(self):
        nums = _extract_numbers("No numbers here")
        assert len(nums) == 0


class TestFuzzyFind:
    def test_exact_match(self):
        assert _fuzzy_find("FDA", "The FDA approved the drug") is True

    def test_case_insensitive(self):
        assert _fuzzy_find("fda", "The FDA approved the drug") is True

    def test_not_found(self):
        assert _fuzzy_find("NMPA", "The FDA approved the drug") is False

    def test_short_string(self):
        assert _fuzzy_find("AI", "AI-based platform") is True

    def test_fuzzy_match(self):
        assert _fuzzy_find("clinicaltrials.gov", "clinical trials gov data") is True


class TestVerifySingleFact:
    SOURCE_TEXT = (
        "Our team at Fudan University has developed a novel mRNA vaccine platform. "
        "The Phase I clinical trial enrolled 120 patients with a response rate of 85%. "
        "The global mRNA vaccine market is projected to reach $45 billion by 2030, "
        "growing at a CAGR of 12.3%. Our partnership with Sinopharm provides "
        "manufacturing capabilities for 500 million doses annually."
    )

    def test_verified_fact(self):
        fact = {
            "text": "Phase I clinical trial enrolled 120 patients with 85% response rate",
            "dimensions": ["feasibility"],
            "type": "clinical_design",
            "meta": {"char_start": 0, "char_end": len(self.SOURCE_TEXT)},
        }
        _, verification = verify_single_fact(fact, self.SOURCE_TEXT)
        assert verification["status"] in ("verified", "partially_verified")
        assert verification["entity_coverage"] > 0.3

    def test_hallucinated_fact(self):
        fact = {
            "text": "The team secured a $200 million grant from the Bill Gates Foundation to build a factory in Singapore",
            "dimensions": ["feasibility"],
            "type": "funding_source",
            "meta": {"char_start": 0, "char_end": len(self.SOURCE_TEXT)},
        }
        _, verification = verify_single_fact(fact, self.SOURCE_TEXT)
        assert verification["entity_coverage"] < 0.5

    def test_empty_fact(self):
        fact = {"text": "", "meta": {}}
        _, verification = verify_single_fact(fact, self.SOURCE_TEXT)
        assert verification["status"] == "unverified"

    def test_numeric_verification(self):
        fact = {
            "text": "Market projected to reach $45 billion by 2030 with CAGR of 12.3%",
            "dimensions": ["strategy"],
            "type": "market",
            "meta": {"char_start": 0, "char_end": len(self.SOURCE_TEXT)},
        }
        _, verification = verify_single_fact(fact, self.SOURCE_TEXT)
        assert verification["numeric_verified"] is True

    def test_wrong_numbers(self):
        fact = {
            "text": "Market projected to reach $99 billion by 2035 with CAGR of 25%",
            "dimensions": ["strategy"],
            "type": "market",
            "meta": {"char_start": 0, "char_end": len(self.SOURCE_TEXT)},
        }
        _, verification = verify_single_fact(fact, self.SOURCE_TEXT)
        assert verification["numeric_verified"] is False
