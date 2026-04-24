# -*- coding: utf-8 -*-
"""
LLM-Based Evaluation Scaffold (evaluation.py)
----------------------------------------------
Provides a clean interface for evaluating proposal review answers
along four quality axes:
  1. Factual grounding - how well the answer is grounded in proposal facts
  2. Completeness - coverage of the question's scope
  3. Insight quality - depth and actionability of analysis
  4. Actionability - whether recommendations are specific and executable

This module is designed to:
  - Coexist with the existing heuristic scorer in post_processing.py
  - Be gradually swapped in as the primary scoring mechanism
  - Support both LLM-based and rule-based evaluation modes

Usage:
    from src.tools.evaluation import Evaluator

    evaluator = Evaluator(mode="hybrid")  # "llm", "heuristic", or "hybrid"
    result = evaluator.evaluate(
        answer="...",
        question="...",
        dimension="team",
        proposal_context="...",
        claims=["..."],
    )
    # result.total_score, result.factual_grounding, etc.
"""

import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvaluationResult:
    """Result of evaluating a single answer."""
    factual_grounding: float = 0.0
    completeness: float = 0.0
    insight_quality: float = 0.0
    actionability: float = 0.0
    total_score: float = 0.0
    mode: str = "heuristic"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HeuristicEvaluator:
    """
    Rule-based evaluator that assesses answer quality using text analysis.
    This preserves the existing scoring logic from post_processing.py
    but exposes it through a clean interface.
    """

    def evaluate(
        self,
        answer: str,
        question: str,
        dimension: str,
        proposal_context: str = "",
        claims: Optional[List[str]] = None,
        evidence_hints: Optional[List[str]] = None,
    ) -> EvaluationResult:
        claims = claims or []
        evidence_hints = evidence_hints or []

        factual = self._score_factual_grounding(answer, proposal_context, claims)
        completeness = self._score_completeness(answer, question)
        insight = self._score_insight_quality(answer, claims)
        actionability = self._score_actionability(answer)

        total = (
            0.30 * factual
            + 0.25 * completeness
            + 0.25 * insight
            + 0.20 * actionability
        )

        return EvaluationResult(
            factual_grounding=round(factual, 3),
            completeness=round(completeness, 3),
            insight_quality=round(insight, 3),
            actionability=round(actionability, 3),
            total_score=round(total, 3),
            mode="heuristic",
            details={
                "answer_length": len(answer),
                "claims_count": len(claims),
                "evidence_count": len(evidence_hints),
            },
        )

    def _score_factual_grounding(
        self, answer: str, context: str, claims: List[str]
    ) -> float:
        if not answer:
            return 0.0

        score = 0.0
        answer_lower = answer.lower()
        context_lower = (context or "").lower()

        # Check how many answer words appear in the context
        if context_lower:
            answer_words = set(re.findall(r"[a-z]+|[\u4e00-\u9fff]+", answer_lower))
            context_words = set(re.findall(r"[a-z]+|[\u4e00-\u9fff]+", context_lower))
            if answer_words:
                overlap = len(answer_words & context_words) / len(answer_words)
                score += 0.5 * min(1.0, overlap * 1.5)

        # Claims quality
        if claims:
            score += 0.3 * min(1.0, len(claims) / 4.0)
            # Check if claims reference context
            grounded = sum(
                1 for c in claims
                if any(
                    w in context_lower
                    for w in re.findall(r"[a-z]+|[\u4e00-\u9fff]+", c.lower())
                    if len(w) > 2
                )
            )
            if claims:
                score += 0.2 * (grounded / len(claims))

        return min(1.0, score)

    def _score_completeness(self, answer: str, question: str) -> float:
        if not answer:
            return 0.0

        score = 0.0

        # Length-based
        length = len(answer)
        if length >= 200:
            score += 0.3
        elif length >= 100:
            score += 0.2
        elif length >= 50:
            score += 0.1

        # Structure (bullet points / numbered lists)
        lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
        bullet_lines = [
            ln for ln in lines
            if re.match(r"^\d+[\.\)]\s+|^[-*•]\s+", ln)
        ]
        if len(bullet_lines) >= 3:
            score += 0.3
        elif len(bullet_lines) >= 1:
            score += 0.15

        # Question keyword coverage
        q_words = set(
            re.findall(r"[a-z]+|[\u4e00-\u9fff]+", question.lower())
        )
        a_words = set(
            re.findall(r"[a-z]+|[\u4e00-\u9fff]+", answer.lower())
        )
        if q_words:
            coverage = len(q_words & a_words) / len(q_words)
            score += 0.4 * min(1.0, coverage * 1.5)

        return min(1.0, score)

    def _score_insight_quality(self, answer: str, claims: List[str]) -> float:
        if not answer:
            return 0.0

        score = 0.0

        # Analysis depth indicators
        depth_markers_zh = [
            "分析", "评估", "建议", "优势", "不足", "风险",
            "改进", "对比", "基准", "趋势",
        ]
        depth_markers_en = [
            "analysis", "assessment", "recommend", "strength",
            "weakness", "risk", "improve", "benchmark", "trend",
        ]
        answer_lower = answer.lower()
        depth_count = sum(
            1 for m in depth_markers_zh + depth_markers_en
            if m in answer_lower
        )
        score += 0.4 * min(1.0, depth_count / 5.0)

        # Claims as proxy for structured insights
        if len(claims) >= 3:
            score += 0.3
        elif len(claims) >= 1:
            score += 0.15

        # Presence of comparison/contrast language
        comparison_markers = [
            "相比", "对比", "优于", "劣于", "不如", "compared",
            "versus", "unlike", "however", "但是", "然而",
        ]
        has_comparison = any(m in answer_lower for m in comparison_markers)
        if has_comparison:
            score += 0.15

        # Penalty for vague language
        vague_markers = ["一般来说", "通常", "可能", "也许", "perhaps", "maybe"]
        vague_count = sum(1 for m in vague_markers if m in answer_lower)
        score -= 0.05 * min(3, vague_count)

        return max(0.0, min(1.0, score))

    def _score_actionability(self, answer: str) -> float:
        if not answer:
            return 0.0

        score = 0.0
        answer_lower = answer.lower()

        # Action-oriented language
        action_markers_zh = [
            "建议", "应当", "需要", "可以考虑", "应该",
            "推荐", "步骤", "计划", "执行", "实施",
        ]
        action_markers_en = [
            "recommend", "should", "need to", "consider",
            "action", "step", "plan", "implement", "execute",
        ]
        action_count = sum(
            1 for m in action_markers_zh + action_markers_en
            if m in answer_lower
        )
        score += 0.5 * min(1.0, action_count / 4.0)

        # Specific next steps
        next_step_markers = [
            "下一步", "接下来", "后续", "next step",
            "follow-up", "todo", "action item",
        ]
        has_next = any(m in answer_lower for m in next_step_markers)
        if has_next:
            score += 0.25

        # Timeline references
        time_markers = [
            "月", "季度", "年", "month", "quarter", "year",
            "周", "week", "阶段", "phase",
        ]
        has_timeline = any(m in answer_lower for m in time_markers)
        if has_timeline:
            score += 0.15

        return min(1.0, score)


class LLMEvaluator:
    """
    LLM-based evaluator that uses a language model to assess answer quality.
    Designed as a drop-in replacement for HeuristicEvaluator.

    Currently a scaffold; can be activated by setting ENABLE_LLM_EVAL=true.
    """

    def __init__(self):
        self._client = None
        self._model = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if key:
                self._client = OpenAI(api_key=key)
                self._model = os.getenv("OPENAI_MODEL", "gpt-5.2")
        except ImportError:
            pass

    def evaluate(
        self,
        answer: str,
        question: str,
        dimension: str,
        proposal_context: str = "",
        claims: Optional[List[str]] = None,
        evidence_hints: Optional[List[str]] = None,
    ) -> EvaluationResult:
        self._ensure_client()
        if not self._client:
            return HeuristicEvaluator().evaluate(
                answer, question, dimension, proposal_context, claims, evidence_hints
            )

        prompt = self._build_eval_prompt(
            answer, question, dimension, proposal_context, claims or []
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert evaluator assessing the quality of "
                            "project review answers. Rate each dimension 0.0-1.0. "
                            "Return only JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)

            return EvaluationResult(
                factual_grounding=float(data.get("factual_grounding", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                insight_quality=float(data.get("insight_quality", 0.5)),
                actionability=float(data.get("actionability", 0.5)),
                total_score=float(data.get("total_score", 0.5)),
                mode="llm",
                details={"raw_response": data},
            )
        except Exception as e:
            # Fall back to heuristic on any LLM error
            result = HeuristicEvaluator().evaluate(
                answer, question, dimension, proposal_context, claims, evidence_hints
            )
            result.details["llm_error"] = str(e)
            return result

    def _build_eval_prompt(
        self,
        answer: str,
        question: str,
        dimension: str,
        context: str,
        claims: List[str],
    ) -> str:
        claims_text = "\n".join(f"- {c}" for c in claims) if claims else "(none)"
        context_snippet = context[:1500] if context else "(no context provided)"

        return f"""Evaluate this project review answer on four dimensions (0.0 to 1.0 each):

**Question**: {question}
**Dimension**: {dimension}

**Proposal Context** (excerpt):
{context_snippet}

**Answer**:
{answer[:2000]}

**Claims made**:
{claims_text}

Score each:
1. factual_grounding: Is the answer grounded in the proposal facts? (0=fabricated, 1=fully grounded)
2. completeness: Does it cover the question's scope? (0=missing everything, 1=comprehensive)
3. insight_quality: Depth of analysis, not just surface-level? (0=trivial, 1=expert-level)
4. actionability: Are recommendations specific and executable? (0=vague, 1=highly actionable)
5. total_score: weighted average (0.30*factual + 0.25*completeness + 0.25*insight + 0.20*actionability)

Return JSON:
{{"factual_grounding": 0.0, "completeness": 0.0, "insight_quality": 0.0, "actionability": 0.0, "total_score": 0.0}}"""


class Evaluator:
    """
    Unified evaluation interface. Supports three modes:
      - "heuristic": Rule-based scoring (fast, no API calls)
      - "llm": LLM-based scoring (higher quality, requires API key)
      - "hybrid": Runs both and averages (recommended for production)
    """

    def __init__(self, mode: str = "heuristic"):
        self.mode = mode
        self._heuristic = HeuristicEvaluator()
        self._llm = LLMEvaluator() if mode in ("llm", "hybrid") else None

    def evaluate(
        self,
        answer: str,
        question: str,
        dimension: str,
        proposal_context: str = "",
        claims: Optional[List[str]] = None,
        evidence_hints: Optional[List[str]] = None,
    ) -> EvaluationResult:
        kwargs = dict(
            answer=answer,
            question=question,
            dimension=dimension,
            proposal_context=proposal_context,
            claims=claims,
            evidence_hints=evidence_hints,
        )

        if self.mode == "heuristic":
            return self._heuristic.evaluate(**kwargs)

        if self.mode == "llm":
            return self._llm.evaluate(**kwargs)

        # Hybrid mode: run both and blend
        h_result = self._heuristic.evaluate(**kwargs)
        l_result = self._llm.evaluate(**kwargs)

        if l_result.mode == "llm":
            # LLM succeeded, blend 60/40 LLM/heuristic
            blended = EvaluationResult(
                factual_grounding=round(0.6 * l_result.factual_grounding + 0.4 * h_result.factual_grounding, 3),
                completeness=round(0.6 * l_result.completeness + 0.4 * h_result.completeness, 3),
                insight_quality=round(0.6 * l_result.insight_quality + 0.4 * h_result.insight_quality, 3),
                actionability=round(0.6 * l_result.actionability + 0.4 * h_result.actionability, 3),
                total_score=round(
                    0.6 * l_result.total_score + 0.4 * h_result.total_score, 3
                ),
                mode="hybrid",
                details={
                    "heuristic": h_result.to_dict(),
                    "llm": l_result.to_dict(),
                },
            )
            return blended

        # LLM failed, fall back to heuristic
        h_result.mode = "hybrid_fallback"
        return h_result
