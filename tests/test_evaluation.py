# -*- coding: utf-8 -*-
"""Tests for the evaluation scaffold module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.evaluation import (
    HeuristicEvaluator,
    Evaluator,
    EvaluationResult,
)


def test_evaluation_result_to_dict():
    r = EvaluationResult(
        factual_grounding=0.8,
        completeness=0.7,
        insight_quality=0.6,
        actionability=0.5,
        total_score=0.65,
        mode="test",
    )
    d = r.to_dict()
    assert d["factual_grounding"] == 0.8
    assert d["mode"] == "test"


def test_heuristic_evaluator_basic():
    evaluator = HeuristicEvaluator()
    result = evaluator.evaluate(
        answer="1. The team has strong AI expertise.\n2. The project addresses unmet clinical needs.\n3. Budget is well-planned.",
        question="How would you assess the overall project quality?",
        dimension="team",
        proposal_context="Our team includes leading AI researchers. The project targets critical unmet needs.",
        claims=["Strong AI team", "Addresses unmet needs", "Good budget planning"],
    )
    assert isinstance(result, EvaluationResult)
    assert result.mode == "heuristic"
    assert 0.0 <= result.total_score <= 1.0
    assert 0.0 <= result.factual_grounding <= 1.0
    assert 0.0 <= result.completeness <= 1.0


def test_heuristic_evaluator_empty_answer():
    evaluator = HeuristicEvaluator()
    result = evaluator.evaluate(
        answer="",
        question="Some question",
        dimension="team",
    )
    assert result.total_score == 0.0


def test_heuristic_evaluator_long_structured_answer():
    evaluator = HeuristicEvaluator()
    answer = """1. 团队优势分析：项目负责人具有20年AI和药物发现经验，主导过多个成功项目。
2. 组织结构：团队采用矩阵式管理，AI团队和临床团队紧密协作。
3. 风险评估：存在人才流失风险，建议制定留人计划。
4. 建议：应该加强跨学科合作，考虑引入更多产业化经验。
5. 对比基准：与行业领先团队相比，在临床转化方面仍有差距。"""
    result = evaluator.evaluate(
        answer=answer,
        question="请评估团队的综合能力",
        dimension="team",
        proposal_context="项目团队包括AI专家和临床研究者。负责人有20年经验。",
        claims=["团队经验丰富", "存在人才流失风险", "需要更多产业化经验"],
    )
    assert result.total_score > 0.2
    assert result.completeness > 0.0
    assert result.insight_quality > 0.0


def test_evaluator_heuristic_mode():
    evaluator = Evaluator(mode="heuristic")
    result = evaluator.evaluate(
        answer="The team is strong.",
        question="How is the team?",
        dimension="team",
    )
    assert result.mode == "heuristic"


def test_evaluator_hybrid_mode_fallback():
    evaluator = Evaluator(mode="hybrid")
    result = evaluator.evaluate(
        answer="The team is strong.",
        question="How is the team?",
        dimension="team",
    )
    # Without API key, should fall back to heuristic
    assert result.mode in ("hybrid", "hybrid_fallback", "heuristic")


if __name__ == "__main__":
    tests = [
        test_evaluation_result_to_dict,
        test_heuristic_evaluator_basic,
        test_heuristic_evaluator_empty_answer,
        test_heuristic_evaluator_long_structured_answer,
        test_evaluator_heuristic_mode,
        test_evaluator_hybrid_mode_fallback,
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
