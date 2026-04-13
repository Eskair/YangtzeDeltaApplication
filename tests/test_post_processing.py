# -*- coding: utf-8 -*-
"""
Tests for post_processing.py scoring logic.

Covers the core functions that were previously untested:
  - norm01, safe_float, tokenize, jaccard
  - looks_structured, overclaim_score
  - score_candidate, _bad_candidate_with_reason
  - sanitize_for_scoring, sanitize_for_display
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.tools.post_processing import (
    norm01,
    safe_float,
    tokenize,
    jaccard,
    looks_structured,
    overclaim_score,
    score_candidate,
    _bad_candidate_with_reason,
    sanitize_for_scoring,
    sanitize_for_display,
    bar,
    authority_ratio,
    coverage_score,
    has_redline,
    DEFAULT_CONF,
    _alignment_ratio,
    _apply_aliases,
)


class TestNorm01:
    def test_within_range(self):
        assert norm01(5, 0, 10) == pytest.approx(0.5)

    def test_at_boundaries(self):
        assert norm01(0, 0, 10) == 0.0
        assert norm01(10, 0, 10) == 1.0

    def test_below_range(self):
        assert norm01(-5, 0, 10) == 0.0

    def test_above_range(self):
        assert norm01(15, 0, 10) == 1.0

    def test_equal_bounds(self):
        assert norm01(5, 5, 5) == 0.0

    def test_inverted_bounds(self):
        assert norm01(5, 10, 0) == 0.0


class TestSafeFloat:
    def test_valid_float(self):
        assert safe_float(0.5) == 0.5

    def test_string_float(self):
        assert safe_float("0.8") == 0.8

    def test_clamped_above(self):
        assert safe_float(1.5) == 1.0

    def test_clamped_below(self):
        assert safe_float(-0.5) == 0.0

    def test_nan(self):
        assert safe_float(float("nan")) == 0.6

    def test_invalid(self):
        assert safe_float("abc") == 0.6

    def test_none(self):
        assert safe_float(None) == 0.6


class TestTokenize:
    def test_english_tokens(self):
        tokens = tokenize("FDA approval process for ISO 13485")
        assert "fda" in tokens
        assert "approval" in tokens
        assert "iso" in tokens
        assert "13485" in tokens

    def test_chinese_tokens(self):
        tokens = tokenize("该团队具有丰富的临床经验和研究能力")
        assert any(len(t) >= 2 for t in tokens)

    def test_empty(self):
        assert tokenize("") == []


class TestJaccard:
    def test_identical(self):
        t = ["a", "b", "c"]
        assert jaccard(t, t) == 1.0

    def test_disjoint(self):
        assert jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_partial(self):
        result = jaccard(["a", "b", "c"], ["b", "c", "d"])
        assert result == pytest.approx(0.5)

    def test_empty(self):
        assert jaccard([], ["a"]) == 0.0
        assert jaccard([], []) == 0.0


class TestLooksStructured:
    def test_well_structured(self):
        answer = """1. First point about the team
2. Second point about objectives
3. Third point about strategy
4. Fourth point about innovation"""
        score = looks_structured(answer)
        assert score > 0.3

    def test_unstructured(self):
        answer = "This is a single paragraph with no structure at all."
        score = looks_structured(answer)
        # Single short line scores on the shortish ratio (6-64 chars),
        # so unstructured single-liners still get a moderate score.
        assert score <= 0.6

    def test_empty(self):
        assert looks_structured("") == 0.0


class TestOverclaimScore:
    def test_no_overclaim(self):
        text = "The team has relevant experience in clinical trials."
        assert overclaim_score(text) < 0.15

    def test_overclaim(self):
        text = "绝对保证100%成功，完全零风险，必须一定能实现"
        assert overclaim_score(text) > 0.0

    def test_empty(self):
        assert overclaim_score("") == 0.0


class TestSanitize:
    def test_scoring_whitespace(self):
        result = sanitize_for_scoring("  hello   world  ")
        assert result == "hello world"

    def test_display_domain_fix(self):
        result = sanitize_for_display("see clinicaltrials  gov for details")
        assert "clinicaltrials" in result.lower()

    def test_empty(self):
        assert sanitize_for_scoring("") == ""
        assert sanitize_for_display("") == ""


class TestBar:
    def test_full(self):
        result = bar(1.0, 10)
        assert result == "██████████"

    def test_empty(self):
        result = bar(0.0, 10)
        assert result == "░░░░░░░░░░"

    def test_half(self):
        result = bar(0.5, 10)
        assert len(result) == 10


class TestAuthorityRatio:
    def test_with_authority(self):
        hints = ["FDA guidance 2023", "ISO 13485 compliance", "random note"]
        ratio = authority_ratio(hints)
        assert ratio >= 0.5

    def test_no_authority(self):
        hints = ["random text", "another random"]
        ratio = authority_ratio(hints)
        assert ratio == 0.0

    def test_empty(self):
        assert authority_ratio([]) == 0.0


class TestCoverageScore:
    def test_mixed_coverage(self):
        hints = ["clinicaltrials.gov NCT12345678", "pubmed article", "USPTO patent"]
        score = coverage_score(hints)
        assert score > 0.0

    def test_no_coverage(self):
        hints = ["random text"]
        assert coverage_score(hints) == 0.0


class TestHasRedline:
    def test_date(self):
        assert has_redline("Completed on 2024-03-15") is True

    def test_trial(self):
        assert has_redline("See NCT12345678 for details") is True

    def test_patent(self):
        assert has_redline("US12345678 patent filed") is True

    def test_no_redline(self):
        assert has_redline("General statement about the project") is False


class TestApplyAliases:
    def test_iso_alias(self):
        result = _apply_aliases("ISO13485")
        assert "iso" in result.lower()
        assert "13485" in result

    def test_ich_alias(self):
        result = _apply_aliases("ICHQ10")
        assert "ich" in result.lower()


class TestScoreCandidate:
    def _make_candidate(self, answer="", claims=None, evidence_hints=None,
                        topic_tags=None, confidence=0.7):
        return {
            "answer": answer,
            "claims": claims or [],
            "evidence_hints": evidence_hints or [],
            "topic_tags": topic_tags or [],
            "confidence": confidence,
            "diag": {},
        }

    def test_good_candidate(self):
        cand = self._make_candidate(
            answer="1. Strong team with proven track record\n2. Clear objectives and milestones\n3. Well-defined strategy",
            claims=["Team has 10+ years experience", "Objectives are measurable"],
            confidence=0.8,
        )
        result = score_candidate(cand, DEFAULT_CONF, dim="team")
        assert result["total"] > 0.0
        assert "scores" in result
        assert "penalties" in result

    def test_empty_candidate(self):
        cand = self._make_candidate(answer="", confidence=0.0)
        result = score_candidate(cand, DEFAULT_CONF, dim="team")
        assert result["total"] < 0.3


class TestBadCandidate:
    def test_error_candidate(self):
        cand = {"answer": "Error occurred", "error": True}
        bad, reason = _bad_candidate_with_reason(cand, DEFAULT_CONF, "team", [])
        assert bad is True
        assert reason == "error"

    def test_short_candidate(self):
        cand = {"answer": "Too short"}
        bad, reason = _bad_candidate_with_reason(cand, DEFAULT_CONF, "team", [])
        assert bad is True
        assert reason == "too_short"

    def test_good_candidate(self):
        # The sanitize_for_display function joins lines at alphanumeric
        # boundaries, so we use Chinese line-starters to preserve structure.
        cand = {
            "answer": (
                "- 核心团队拥有超过15年的mRNA疫苗开发经验，包括在Nature和Science期刊发表的LNP递送系统研究\n"
                "- 项目负责人已成功指导两个候选药物通过II期临床试验\n"
                "- 团队结构涵盖实验科学家和计算生物学家，形成多学科互补\n"
                "- 风险因素包括对主要研究者的关键人员依赖\n"
                "- 建议聘请专职法规事务专家，并建立关键岗位继任计划\n"
            ),
            "claims": [
                "Team has 15+ years mRNA experience",
                "Two Phase II trials completed",
                "Interdisciplinary team structure",
            ],
            "evidence_hints": ["published in Nature", "Phase II trials"],
            "topic_tags": ["team", "mRNA", "clinical"],
        }
        bad, reason = _bad_candidate_with_reason(cand, DEFAULT_CONF, "team", [])
        assert bad is False, f"Expected candidate to pass but got: bad={bad}, reason={reason}"


class TestAlignmentRatio:
    def test_with_hints(self):
        score = _alignment_ratio(
            "team",
            ["FDA regulatory experience", "clinical trial design"],
            "The team has extensive FDA regulatory experience and clinical trial design expertise.",
            ["team", "regulatory"],
            [],
        )
        assert score > 0.0

    def test_no_hints(self):
        score = _alignment_ratio("team", [], "Some answer text", [], [])
        assert score >= 0.0

    def test_empty_everything(self):
        score = _alignment_ratio("team", [], "", [], [])
        assert score == 0.0
