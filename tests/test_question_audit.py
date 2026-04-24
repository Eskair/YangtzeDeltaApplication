# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.question_audit import grounding_heuristic, load_full_text


def test_grounding_heuristic_hits():
    ft = "本公司在上海设立研发中心，注册资本五千万元，主营工业软件。"
    g = grounding_heuristic("上海研发中心的注册资本是多少？", "", ft)
    assert g["label"] in ("grounded", "weak")
    assert (g.get("score") or 0) > 0.1


def test_grounding_ungrounded():
    ft = "仅讨论天气与旅游。"
    g = grounding_heuristic("请详细说明贵司在火星基地的供电方案。", "", ft)
    assert g["label"] == "ungrounded"


def test_load_full_text(tmp_path):
    pid = "x"
    prep = tmp_path / "src" / "data" / "prepared" / pid
    prep.mkdir(parents=True)
    (prep / "full_text.txt").write_text("hello", encoding="utf-8")
    assert load_full_text(tmp_path, pid) == "hello"
