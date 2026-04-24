# -*- coding: utf-8 -*-
"""
RAG-style excerpts from prepared proposal text for question generation and answering.

- Prefers pages.json (per-page text) when present; otherwise chunks full_text.txt.
- Scores chunks by simple token / substring overlap with a retrieval query built from
  dimensions payload (summary + key_points + risks, etc.).
- Budget scales with key_points_count: sparse summaries get more chunks/chars.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _extract_terms(text: str, max_terms: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    terms: List[str] = []
    try:
        import jieba  # type: ignore

        for x in jieba.cut(text):
            x = x.strip()
            if len(x) >= 2 and not x.isspace():
                terms.append(x)
    except Exception:
        terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}", text))
        terms.extend(t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", text))
    out: List[str] = []
    seen = set()
    for t in terms:
        key = t if "\u4e00" <= t[0] <= "\u9fff" else t.lower()
        if key in seen or len(t) > 48:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def extract_retrieval_terms(text: str, max_terms: int = 100) -> List[str]:
    """Public wrapper for term extraction (question audit, diagnostics, etc.)."""
    return _extract_terms(text, max_terms)


def _split_long_page(text: str, max_chunk: int, stride: int) -> List[str]:
    text = text or ""
    if len(text) <= max_chunk:
        return [text] if text else []
    out: List[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + max_chunk])
        i += stride
        if i >= len(text):
            break
    return out


def _load_chunks(prepared_dir: Path) -> List[Tuple[str, int]]:
    """
    Return list of (chunk_text, page_index) where page_index is 0 if unknown.
    """
    pages_path = prepared_dir / "pages.json"
    if pages_path.exists():
        try:
            pages = json.loads(pages_path.read_text(encoding="utf-8"))
        except Exception:
            pages = []
        if isinstance(pages, list):
            max_chunk = _env_int("RAG_PAGE_CHUNK_CHARS", 1400)
            stride = _env_int("RAG_PAGE_CHUNK_STRIDE", 900)
            chunks: List[Tuple[str, int]] = []
            for p in pages:
                if not isinstance(p, dict):
                    continue
                pi = int(p.get("page_index") or 0)
                txt = str(p.get("text") or "")
                for piece in _split_long_page(txt, max_chunk, stride):
                    if piece.strip():
                        chunks.append((piece, pi))
            if chunks:
                return chunks

    ft = prepared_dir / "full_text.txt"
    if not ft.exists():
        return []
    try:
        full = ft.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    win = _env_int("RAG_WINDOW_CHARS", 1400)
    step = _env_int("RAG_WINDOW_STRIDE", 700)
    out: List[Tuple[str, int]] = []
    i = 0
    while i < len(full):
        piece = full[i : i + win]
        if piece.strip():
            out.append((piece, 0))
        i += step
        if i >= len(full):
            break
    return out


def _score_chunk(chunk: str, terms: List[str]) -> int:
    if not chunk or not terms:
        return 0
    score = 0
    for t in terms:
        if len(t) >= 2 and t in chunk:
            score += len(t) if "\u4e00" <= t[0] <= "\u9fff" else max(3, len(t))
    return score


def _budget_for_kp(key_points_count: int) -> Tuple[int, int]:
    """
    Returns (top_k, max_total_chars).
    Few key_points -> more retrieval budget; many -> lean on summary.
    """
    base_k = _env_int("RAG_TOP_K", 5)
    base_budget = _env_int("RAG_MAX_CHARS", 4500)
    if key_points_count <= 0:
        return min(base_k + 3, 9), min(int(base_budget * 1.35), 8000)
    if key_points_count < 4:
        return min(base_k + 2, 8), min(int(base_budget * 1.25), 7000)
    if key_points_count < 8:
        return base_k, base_budget
    return max(base_k - 1, 3), int(base_budget * 0.88)


def format_snippets_for_prompt(
    project_root: Path,
    proposal_id: str,
    retrieval_query: str,
    key_points_count: int,
) -> str:
    """
    Build a markdown section for prompts, or "" if no prepared text / no overlap.
    """
    prepared_dir = project_root / "src" / "data" / "prepared" / proposal_id
    if not prepared_dir.is_dir():
        return ""

    chunks = _load_chunks(prepared_dir)
    if not chunks:
        return ""

    terms = _extract_terms(retrieval_query, max_terms=120)
    if not terms:
        terms = _extract_terms(re.sub(r"\s+", " ", retrieval_query[:2000]), max_terms=80)

    # (score, chunk_index, text, page_index)
    scored: List[Tuple[int, int, str, int]] = []
    for idx, (text, page) in enumerate(chunks):
        sc = _score_chunk(text, terms)
        if sc > 0:
            scored.append((sc, idx, text, page))

    top_k, max_chars = _budget_for_kp(key_points_count)

    if not scored:
        scored = [
            (1, i, t, p)
            for i, (t, p) in enumerate(chunks[: max(top_k, 3)])
        ]

    scored.sort(key=lambda x: (-x[0], x[1]))
    picked: List[str] = []
    used = 0
    seen_prefix = set()
    for sc, _i, text, page in scored:
        head = text.strip()[:120]
        if head in seen_prefix:
            continue
        seen_prefix.add(head)
        block = text.strip()
        if not block:
            continue
        page_note = f"（约第 {page} 页）" if page > 0 else ""
        piece = f"【摘录{len(picked)+1}{page_note}】\n{block}"
        if used + len(piece) > max_chars:
            remain = max_chars - used - 40
            if remain > 200:
                piece = f"【摘录{len(picked)+1}{page_note}】\n{block[:remain]}…（截断）"
            else:
                break
        picked.append(piece)
        used += len(piece)
        if len(picked) >= top_k:
            break

    if not picked:
        return ""

    header = (
        "【原文摘录（由摘要要点检索 full_text/pages，非全文；用于与维度摘要交叉核对）】\n"
        "说明：仅可引用以下摘录中实际出现的词句与数字；摘录未覆盖不代表全文不存在；"
        "不得以摘录之外的具体机构名、数字或结论编造。\n"
    )
    return header + "\n\n".join(picked)
