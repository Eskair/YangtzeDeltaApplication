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
)
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
                    "model": "gpt-4o-mini",
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
    _, per_d, ov = aggregate_dimensions(items, DEFAULT_CONF, {})

    assert "overall_score" in ov and "overall_score_raw" in ov
    assert "overall_confidence" in ov and "overall_confidence_raw" in ov
    for d in DIM_ORDER:
        assert "avg" in per_d[d] and "avg_raw" in per_d[d]

    # 校准不应降低展示分（仿射参数在默认配置下为抬升）
    assert ov["overall_score"] + 1e-9 >= ov["overall_score_raw"]
    assert ov["overall_confidence"] + 1e-9 >= ov["overall_confidence_raw"]


def test_generate_final_report_metrics_helpers():
    assert not _output_calibration_enabled(None)
    assert not _output_calibration_enabled({})
    m = {"config_used": {"output_calibration": {"enabled": True}}}
    assert _output_calibration_enabled(m)
    assert _overall_raw_from_metrics({}) is None
    m2 = {"overall": {"overall_score": 0.8, "overall_score_raw": 0.5, "overall_confidence": 0.7, "overall_confidence_raw": 0.5}}
    pair = _overall_raw_from_metrics(m2)
    assert pair == (0.5, 0.5)
