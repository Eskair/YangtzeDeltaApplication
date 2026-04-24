# -*- coding: utf-8 -*-
"""post_processing 输出校准与聚合字段的回归测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.post_processing import (
    DEFAULT_CONF,
    DIM_ORDER,
    _apply_output_calibration_scalar,
    aggregate_dimensions,
    build_grounding_maps_from_detail,
    grounding_multiplier_for_entry,
    load_config,
    lookup_grounding_entry,
)
from src.tools.postproc_lexicon import load_postproc_lexicon
from src.tools.generate_final_report import (
    _output_calibration_enabled,
    _overall_raw_from_metrics,
)


def test_apply_output_calibration_raises_mid_scores():
    cfg = DEFAULT_CONF
    x = 0.471
    y = _apply_output_calibration_scalar(x, "score", cfg)
    assert y > 0.69
    c = _apply_output_calibration_scalar(0.512, "confidence", cfg)
    assert c > 0.69


def test_apply_output_calibration_respects_apply_below():
    cfg = DEFAULT_CONF
    th = float((cfg.get("output_calibration") or {}).get("apply_above", 0.0))
    low = max(0.0, th - 0.05)
    assert _apply_output_calibration_scalar(low, "score", cfg) == low


def test_aggregate_dimensions_exposes_raw_and_calibrated():
    """五维各一题、单候选；断言 metrics 风格字段齐全。"""

    def _item(dim: str, qidx: int):
        body = "说明文字" * 20
        ans = f"- {body}\n- {body}\n- {body}"
        return {
            "dimension": dim,
            "q_index": qidx,
            "question": "请结合提案分析该维度下的主要优势、风险与改进建议。",
            "candidates": [
                {
                    "provider": "openai",
                    "model": "gpt-5.2",
                    "answer": ans,
                    "claims": [
                        "要点一：与问题相关的结论",
                        "要点二：与问题相关的结论",
                        "要点三：与问题相关的结论",
                    ],
                    "evidence_hints": ["提案中关于该维度的描述"],
                    "topic_tags": [dim],
                    "confidence": 0.52,
                    "variant_id": "t",
                }
            ],
        }

    items = [_item(d, i + 1) for i, d in enumerate(DIM_ORDER)]
    cfg = load_config()
    _, per_d, ov = aggregate_dimensions(items, cfg, {})

    assert "overall_score" in ov and "overall_score_raw" in ov
    assert "overall_confidence" in ov and "overall_confidence_raw" in ov
    for d in DIM_ORDER:
        assert "avg" in per_d[d] and "avg_raw" in per_d[d]

    # 校准不应降低展示分（仿射参数在默认配置下为抬升）
    assert ov["overall_score"] + 1e-9 >= ov["overall_score_raw"]
    assert ov["overall_confidence"] + 1e-9 >= ov["overall_confidence_raw"]


def test_grounding_maps_lookup_and_multiplier():
    detail = {
        "dimensions": {
            "team": {
                "questions": [
                    {
                        "question_zh": "  A 题面  ",
                        "audit": {"grounding": {"label": "weak", "score": 0.4}},
                    }
                ]
            }
        }
    }
    maps = build_grounding_maps_from_detail(detail)
    assert maps is not None
    e = lookup_grounding_entry(maps, "team", 1, "A 题面")
    assert e["label"] == "weak"
    block = {"enabled": True, "by_label": {"weak": 0.95}, "multiplier_bounds": [0.5, 1.2]}
    assert grounding_multiplier_for_entry(e, block) == 0.95


def test_aggregate_dimensions_grounding_weight_adjusts_final():
    """有 detail 索引时 grounded 乘子>1 应抬高 score.final 与 overall_score_raw。"""

    def _one_team():
        body = "说明文字" * 20
        ans = f"- {body}\n- {body}\n- {body}"
        return {
            "dimension": "team",
            "q_index": 1,
            "question": "请结合提案分析该维度下的主要优势、风险与改进建议。",
            "candidates": [
                {
                    "provider": "openai",
                    "model": "gpt-5.2",
                    "answer": ans,
                    "claims": ["要点一", "要点二", "要点三"],
                    "evidence_hints": ["提案中关于该维度的描述"],
                    "topic_tags": ["team"],
                    "confidence": 0.52,
                    "variant_id": "t",
                }
            ],
        }

    items = [_one_team()]
    cfg = dict(load_config())
    cfg["question_grounding_weight"] = {
        "enabled": True,
        "by_label": {
            "grounded": 1.1,
            "weak": 1.0,
            "ungrounded": 0.92,
            "no_source_text": 1.0,
            "unknown": 1.0,
        },
        "multiplier_bounds": [0.65, 1.15],
    }
    maps = {"by_pos": {("team", 1): {"label": "grounded"}}, "by_text": {}}

    pq0, _, ov0 = aggregate_dimensions(items, cfg, {}, None)
    pq1, _, ov1 = aggregate_dimensions(items, cfg, {}, maps)

    assert float(pq1[0]["score"]["final"]) > float(pq0[0]["score"]["final"]) + 1e-6
    assert float(ov1["overall_score_raw"]) > float(ov0["overall_score_raw"]) + 1e-6
    assert pq1[0].get("question_grounding", {}).get("label") == "grounded"
    assert (ov1.get("question_grounding") or {}).get("detail_loaded") is True


def test_postproc_authority_profiles_yaml():
    cfg_d = dict(load_config())
    cfg_d["authority_profile"] = "default"
    lx_d = load_postproc_lexicon(cfg_d)
    assert lx_d.get("strong_alignment_bonus") is False
    assert "fda" not in (lx_d.get("authority_tokens") or [])

    cfg_r = dict(load_config())
    cfg_r["authority_profile"] = "regulated_products"
    lx_r = load_postproc_lexicon(cfg_r)
    assert lx_r.get("strong_alignment_bonus") is True
    assert "fda" in (lx_r.get("authority_tokens") or [])


def test_generate_final_report_metrics_helpers():
    assert not _output_calibration_enabled(None)
    assert not _output_calibration_enabled({})
    m = {"config_used": {"output_calibration": {"enabled": True}}}
    assert _output_calibration_enabled(m)
    assert _overall_raw_from_metrics({}) is None
    m2 = {"overall": {"overall_score": 0.8, "overall_score_raw": 0.5, "overall_confidence": 0.7, "overall_confidence_raw": 0.5}}
    pair = _overall_raw_from_metrics(m2)
    assert pair == (0.5, 0.5)
