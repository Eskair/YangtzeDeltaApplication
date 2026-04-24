# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.review_text_snippets import format_snippets_for_prompt


def test_snippets_from_full_text(tmp_path):
    root = tmp_path / "proj"
    pid = "demo_pid"
    prep = root / "src" / "data" / "prepared" / pid
    prep.mkdir(parents=True)
    long_txt = (
        "无关前言。" * 50
        + "本公司计划在长三角地区建设智能仓储中心，投资约三亿元，2026年投产。"
        + "结尾其他内容。" * 50
    )
    (prep / "full_text.txt").write_text(long_txt, encoding="utf-8")

    q = "长三角 智能仓储 三亿元 2026"
    out = format_snippets_for_prompt(root, pid, q, key_points_count=2)
    assert "摘录" in out
    assert "长三角" in out or "仓储" in out


def test_snippets_prefers_pages_json(tmp_path):
    root = tmp_path / "proj2"
    pid = "p2"
    prep = root / "src" / "data" / "prepared" / pid
    prep.mkdir(parents=True)
    pages = [
        {"page_index": 1, "text": "第一页无关键词。"},
        {"page_index": 2, "text": "本页披露：目标客户为大型连锁商超，年采购额超过五亿元。"},
    ]
    (prep / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    (prep / "full_text.txt").write_text("fallback", encoding="utf-8")

    out = format_snippets_for_prompt(root, pid, "连锁商超 五亿元", key_points_count=6)
    assert "第 2 页" in out or "商超" in out
