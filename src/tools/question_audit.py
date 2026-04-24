# -*- coding: utf-8 -*-
"""
Post-generation audit for generate_questions.py:

1) Heuristic grounding: question terms vs full_text (prepared/<pid>/full_text.txt).
2) LLM pass (默认开启，见 generate_questions / QUESTION_AUDIT_LLM): industry-normative vs off-topic; rephrase or drop.

When LLM returns action=remove, QUESTION_AUDIT_AUTO_DROP defaults to applying the drop (set 0 to tag only).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.tools.review_text_snippets import extract_retrieval_terms


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def load_full_text(project_root: Path, proposal_id: str) -> str:
    p = project_root / "src" / "data" / "prepared" / proposal_id / "full_text.txt"
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def grounding_heuristic(question_zh: str, question_en: str, full_text: str) -> Dict[str, Any]:
    """
    Term hit ratio against full_text + longest shared substring (CJK) hint.
    label: grounded | weak | ungrounded | no_source_text
    """
    blob = full_text or ""
    if not blob.strip():
        return {
            "score": None,
            "label": "no_source_text",
            "matched_terms": 0,
            "term_total": 0,
            "lc_substr_len": 0,
        }

    q = f"{question_zh or ''} {question_en or ''}".strip()
    terms = extract_retrieval_terms(q, max_terms=100)
    if not terms:
        return {"score": 0.0, "label": "ungrounded", "matched_terms": 0, "term_total": 0, "lc_substr_len": 0}

    hits = 0
    blob_lower = blob.lower()
    for t in terms:
        if len(t) >= 2 and t in blob:
            hits += 1
        elif not ("\u4e00" <= t[0] <= "\u9fff") and t.lower() in blob_lower:
            hits += 1
    ratio = hits / max(len(terms), 1)

    # longest common substring (naive O(n*m) cap) for short question
    zh = (question_zh or "").strip()
    lc = 0
    if len(zh) >= 4 and len(blob) > 0:
        cap_zh = zh[:200]
        cap_blob = blob[:80000]
        for i in range(len(cap_zh)):
            for L in range(min(32, len(cap_zh) - i), 3, -1):
                sub = cap_zh[i : i + L]
                if sub in cap_blob:
                    lc = max(lc, L)
                    break

    if ratio >= 0.32 or lc >= 10:
        label = "grounded"
    elif ratio >= 0.12 or lc >= 6:
        label = "weak"
    else:
        label = "ungrounded"

    return {
        "score": round(float(ratio), 4),
        "label": label,
        "matched_terms": int(hits),
        "term_total": len(terms),
        "lc_substr_len": int(lc),
    }


AUDIT_LLM_SYSTEM = (
    "你是资深商业/投融资材料评审秘书，只做「问题清单审核」，不回答业务本身。"
    "输入含：维度名、该维度的摘要要点、提案全文前部摘录、以及若干已生成的问题（中英）。"
    "请逐题判断：\n"
    "1）material_grounded：题干核心可在「摘录或要点」中找到明显依据；\n"
    "2）industry_normative：原文/要点未充分写清，但该问题属于**行业评审通识下几乎应问**的要点"
    "（如现金流与用途、关键人依赖、合规边界、里程碑可测性等），保留有利于下游形成结构化答案；\n"
    "3）off_topic：与材料或该维度评审目标明显无关，或极易诱发模型编造具体机构/数字。\n"
    "对 off_topic 优先 action=rephrase 给出更贴材料的问题；无法挽救则 remove。\n"
    "不得编造材料中不存在的新事实；rephrase 只能使用摘录/要点已出现的实体与数字，或改为不点名的一般性问法。\n"
    "仅输出 JSON。"
)


def _audit_llm_prompt(
    dimension: str,
    dim_brief: str,
    full_text_excerpt: str,
    items: List[Dict[str, Any]],
) -> str:
    lines = []
    for i, it in enumerate(items):
        lines.append(
            json.dumps(
                {
                    "index": i,
                    "qid": it.get("qid", ""),
                    "question_zh": it.get("question_zh", ""),
                    "question_en": it.get("question_en", ""),
                },
                ensure_ascii=False,
            )
        )
    return (
        f"维度：{dimension}\n\n"
        f"【该维度摘要与要点（节选）】\n{dim_brief[:3500]}\n\n"
        f"【提案全文摘录（非全文，仅前部）】\n{full_text_excerpt[:6500]}\n\n"
        f"【待审核问题】（每行一个 JSON 对象）\n"
        + "\n".join(lines)
        + "\n\n"
        "请输出唯一 JSON 对象，格式：\n"
        '{"items":[{"index":0,"action":"keep|rephrase|remove","category":"material_grounded|industry_normative|off_topic",'
        '"question_zh_new":"","question_en_new":"","reason_zh":""}, ...]}\n'
        "要求：items 长度与问题数一致；index 对应输入顺序；keep 时 question_*_new 可为空字符串。"
    )


def _call_llm_audit(
    client: Any,
    model: str,
    dimension: str,
    dim_brief: str,
    full_text: str,
    items: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not items or client is None:
        return out
    excerpt = (full_text or "")[:12000]
    user = _audit_llm_prompt(dimension, dim_brief, excerpt, items)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AUDIT_LLM_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception as e:
        print(f"[WARN] LLM 问题审核失败（{dimension}）: {e}")
        return out

    arr = data.get("items") if isinstance(data, dict) else None
    if not isinstance(arr, list):
        return out
    for row in arr:
        if not isinstance(row, dict):
            continue
        try:
            ix = int(row.get("index", -1))
        except Exception:
            continue
        if ix < 0:
            continue
        out[ix] = {
            "action": str(row.get("action") or "keep").strip().lower(),
            "category": str(row.get("category") or "").strip(),
            "question_zh_new": str(row.get("question_zh_new") or "").strip(),
            "question_en_new": str(row.get("question_en_new") or "").strip(),
            "reason_zh": str(row.get("reason_zh") or "").strip(),
        }
    return out


def _dim_brief(dim_data: Dict[str, Any]) -> str:
    parts = [str(dim_data.get("summary") or "")]
    for k in ("key_points", "risks", "mitigations"):
        v = dim_data.get(k) or []
        if isinstance(v, list):
            parts.extend(str(x) for x in v if isinstance(x, str))
    return "\n".join(p for p in parts if p).strip()


def audit_generated_questions_for_proposal(
    *,
    project_root: Path,
    proposal_id: str,
    dimension_names: List[str],
    dimensions: Dict[str, Any],
    all_dim_questions: Dict[str, Any],
    openai_client: Optional[Any],
    openai_model: str,
    use_llm_audit: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Mutates copies of question dicts inside all_dim_questions with ['audit'].
    May drop questions when QUESTION_AUDIT_AUTO_DROP=1 and LLM says remove+off_topic (conservative).

    Returns (updated_all_dim_questions, audit_report_meta).
    """
    full_text = load_full_text(project_root, proposal_id)
    report: Dict[str, Any] = {
        "proposal_id": proposal_id,
        "full_text_chars": len(full_text),
        "use_llm_audit": use_llm_audit,
        "dimensions": {},
    }

    # 与 LLM 审核配合：默认对 action=remove 的题执行删除（可用 QUESTION_AUDIT_AUTO_DROP=0 关闭）
    auto_drop = _env_bool("QUESTION_AUDIT_AUTO_DROP", True)

    for dim in dimension_names:
        block = all_dim_questions.get(dim)
        if not isinstance(block, dict):
            continue
        qs = block.get("questions")
        if not isinstance(qs, list):
            continue

        dim_data = dimensions.get(dim) or {}
        brief = _dim_brief(dim_data)

        audited_list: List[Dict[str, Any]] = []
        for q in qs:
            if not isinstance(q, dict):
                continue
            item = dict(q)
            zh = str(item.get("question_zh") or "")
            en = str(item.get("question_en") or "")
            g = grounding_heuristic(zh, en, full_text)
            item["audit"] = {"grounding": g, "llm": None}
            audited_list.append(item)

        llm_map: Dict[int, Dict[str, Any]] = {}
        if use_llm_audit and openai_client is not None and audited_list:
            llm_map = _call_llm_audit(
                openai_client, openai_model, dim, brief, full_text, audited_list
            )

        kept: List[Dict[str, Any]] = []
        for i, item in enumerate(audited_list):
            patch = llm_map.get(i) or {}
            if patch:
                item["audit"]["llm"] = patch
                if patch.get("action") == "rephrase" and patch.get("question_zh_new"):
                    item["question_zh"] = patch["question_zh_new"]
                if patch.get("action") == "rephrase" and patch.get("question_en_new"):
                    item["question_en"] = patch["question_en_new"]
            action = (patch.get("action") or "keep").lower()

            drop = bool(auto_drop and action == "remove")

            if drop:
                continue
            kept.append(item)

        # re-assign qid sequential after drops
        for j, item in enumerate(kept, start=1):
            item["qid"] = f"{dim}_q{j:02d}"

        all_dim_questions[dim] = {"dimension": dim, "questions": kept}
        report["dimensions"][dim] = {
            "input_count": len(audited_list),
            "output_count": len(kept),
            "dropped": max(0, len(audited_list) - len(kept)),
            "grounding_labels": _count_labels(audited_list),
            "llm_applied": bool(llm_map),
        }

    return all_dim_questions, report


def _count_labels(items: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for it in items:
        lab = ((it.get("audit") or {}).get("grounding") or {}).get("label")
        if not lab:
            continue
        c[lab] = c.get(lab, 0) + 1
    return c
