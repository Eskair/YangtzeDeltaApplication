# -*- coding: utf-8 -*-
"""
阶段 9：Final Report Generator（v3 · 一页摘要 + 专家评审 + 精简问答 · 基于 final_payload）

功能：
  1) 自动读取：
       - refined_answers/<pid>/postproc/final_payload.json
       - refined_answers/<pid>/postproc/metrics.json（若存在：一页摘要与第 2 节标尺说明）
       - expert_reports/<pid>/ai_expert_opinion.json
       - expert_reports/<pid>/ai_expert_opinion.md
  2) 生成综合 Markdown 报告：
       - 顶部：项目综合评审报告（含生成时间与说明）
       - 0. 一页摘要（Executive Summary）：
           · verdict + 一句话结论依据
           · 综合评分、信心度 + 区间解释
           · Top 3 优势 & Top 3 主要风险
           · 关键补充材料建议
       - 1. AI 专家总体评审（来自 ai_expert_opinion.md，自动降一级标题）
       - 2. 维度问答与打分依据（基于 final_payload）：
             · 按五个维度展示
             · 每个维度：综合得分 + 打分依据摘要（rationales）+ 行业通识要点（general_insights）
             · 每道题：问题、选中回答、来源模型、置信度、对齐度、漂移等元信息
             · 每个维度仅展示前 MAX_QA_PER_DIM 条高价值问答（按置信度 + 对齐度排序）

用法：
  cd 到项目根目录（包含 src/）
  python -m src.tools.generate_final_report
  python -m src.tools.generate_final_report --pid Ebovir_LNP
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# ========= 路径配置 =========
BASE_DIR = Path(__file__).resolve().parents[2]   # 项目根（包含 src/）
DATA_DIR = BASE_DIR / "src" / "data"
REFINED_ROOT = DATA_DIR / "refined_answers"
EXPERT_ROOT = DATA_DIR / "expert_reports"
REPORT_ROOT = DATA_DIR / "reports"
PROGRESS_FILE = DATA_DIR / "step_progress.json"


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass

def _get_domain_config():
    """Load config from centralized config system."""
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from src.config import get_config
        return get_config()
    except Exception:
        return None


def _init_dim_order():
    cfg = _get_domain_config()
    if cfg:
        return cfg.dimension_names
    return ["team", "objectives", "strategy", "innovation", "feasibility"]


def _init_dim_labels():
    cfg = _get_domain_config()
    if cfg:
        return cfg.dimension_labels_zh
    return {
        "team": "团队与治理",
        "objectives": "项目目标",
        "strategy": "实施路径与战略",
        "innovation": "技术与产品创新",
        "feasibility": "资源与可行性",
    }


DIM_ORDER = _init_dim_order()
DIM_LABELS_ZH = _init_dim_labels()

# 每个维度最多展示多少条问答（从高置信度 / 高对齐度往下选）
MAX_QA_PER_DIM = 6


# ========= 工具函数 =========
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _output_calibration_enabled(metrics: Optional[Dict[str, Any]]) -> bool:
    cu = (metrics or {}).get("config_used") or {}
    oc = cu.get("output_calibration") or {}
    return bool(oc.get("enabled"))


def _overall_raw_from_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[tuple]:
    """若 metrics 含 overall_*_raw，返回 (score_raw, conf_raw)，否则 None。"""
    if not metrics:
        return None
    mo = metrics.get("overall") or {}
    if "overall_score_raw" not in mo and "overall_confidence_raw" not in mo:
        return None
    try:
        rs = float(mo.get("overall_score_raw", mo.get("overall_score", 0.0)) or 0.0)
        rc = float(mo.get("overall_confidence_raw", mo.get("overall_confidence", 0.0)) or 0.0)
        return rs, rc
    except (TypeError, ValueError):
        return None


def detect_latest_pid() -> str:
    """
    从 refined_answers 下自动选一个“最新且有 postproc/final_payload.json + 对应专家评审”的 pid
    """
    if not REFINED_ROOT.exists():
        return ""
    cands = []
    for d in REFINED_ROOT.iterdir():
        if not d.is_dir():
            continue
        postproc_dir = d / "postproc"
        fp_path = postproc_dir / "final_payload.json"
        expert_md_path = EXPERT_ROOT / d.name / "ai_expert_opinion.md"
        if fp_path.exists() and expert_md_path.exists():
            cands.append((d.name, fp_path.stat().st_mtime))
    if not cands:
        return ""
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0]


def adjust_expert_markdown(md_text: str) -> str:
    """
    将 ai_expert_opinion.md 的标题全部降一级：
      #  → ##
      ## → ###
      ###→ ####
    避免和顶层报告标题冲突。
    """
    lines = md_text.splitlines()
    adjusted = []
    for line in lines:
        if line.startswith("# "):
            adjusted.append("#" + line)      # '# ' → '## '
        elif line.startswith("## "):
            adjusted.append("#" + line)      # '## ' → '### '
        elif line.startswith("### "):
            adjusted.append("#" + line)      # '### '→ '#### '
        else:
            adjusted.append(line)
    return "\n".join(adjusted)


def _fmt_float(v, ndigits=1, default="0.0"):
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return default


# ========= 一页摘要生成 =========
def build_executive_summary(
    expert_json: dict,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    基于 ai_expert_opinion.json 生成“0. 一页摘要（Executive Summary）”Markdown 文本。
    若传入 postproc/metrics.json 解析结果，可与 output_calibration 对齐并回显权威 raw 分。
    如果 expert_json 为空，则返回空字符串，调用方自己判断是否插入。
    """
    if not expert_json:
        return ""

    overall = expert_json.get("overall_opinion", {}) or {}
    score = overall.get("overall_score_echo", 0.0)       # 0–1 区间
    conf = overall.get("confidence_echo", 0.0)           # 0–1 区间
    verdict = (overall.get("verdict") or "").strip()
    summary = (overall.get("summary") or "").strip()
    key_strengths = overall.get("key_strengths") or []
    key_risks = overall.get("key_risks") or []
    recs = overall.get("recommendations") or []
    basis = overall.get("basis") or []

    # verdict 补充中文解释
    verdict_zh = ""
    if verdict == "GO":
        verdict_zh = "（建议在可控风险前提下推进）"
    elif verdict == "HOLD":
        verdict_zh = "（建议补充材料后再决策）"
    elif verdict == "NO-GO":
        verdict_zh = "（当前条件下不建议立项）"

    # 一句话结论依据：优先用 basis[0]，其次用 summary
    brief_reason = ""
    if basis:
        brief_reason = basis[0]
    elif summary:
        brief_reason = summary

    lines = []
    lines.append("## 0. 一页摘要（Executive Summary）")
    lines.append("")
    if verdict:
        lines.append(f"- **总体结论（verdict）**：{verdict} {verdict_zh}".strip())
    else:
        lines.append("- **总体结论（verdict）**：暂无明确结论")
    if brief_reason:
        lines.append(f"- **结论依据简述**：{brief_reason}")
    try:
        rs = float(overall.get("overall_score_raw_echo", score) or score)
        rc = float(overall.get("confidence_raw_echo", conf) or conf)
    except (TypeError, ValueError):
        rs, rc = score, conf
    mr = _overall_raw_from_metrics(metrics)
    if mr is not None:
        rs, rc = mr[0], mr[1]
    calibrated_note = abs(rs - score) > 0.02 or abs(rc - conf) > 0.02
    scale_help = calibrated_note or _output_calibration_enabled(metrics)
    if calibrated_note:
        lines.append(f"- **综合评分**：{_fmt_float(score, 3)}（0–1 区间，输出校准后读数）")
        lines.append(f"- **信心度**：{_fmt_float(conf, 3)}（输出校准后读数）")
        lines.append(
            f"- **原始尺度**（verdict 判定依据，未做输出校准）：综合评分 {_fmt_float(rs, 3)}，信心度 {_fmt_float(rc, 3)}"
        )
    else:
        lines.append(f"- **综合评分**：{_fmt_float(score, 3)}（0–1 区间）")
        lines.append(f"- **信心度**：{_fmt_float(conf, 3)}")
    lines.append("")
    lines.append("> **评分区间说明（供非技术评审参考）**：")
    if scale_help:
        lines.append(
            "> - 下列 **0.62 / 0.45** 阈值针对本页「综合评分」行中的 **0–1 读数**；"
            "若该行已标注为「输出校准后读数」，请按**校准后的分**对照区间。"
            "**verdict** 仍以「原始尺度」一行为准（与 `metrics.json` 中 `overall_*_raw` 一致）。"
        )
    lines.append("> - ≥ 0.62：整体条件较好，可在控制风险前提下推进；")
    lines.append("> - 0.45–0.62：信息不充分或优劣并存，建议补充材料后再决策；")
    lines.append("> - < 0.45：关键维度存在明显短板或高不确定性，一般不建议立项。")
    lines.append("")

    # Top 优势 / 风险
    if key_strengths:
        lines.append("**Top 项目优势（按重要性排序，最多列出 3 条）**")
        for s in key_strengths[:3]:
            lines.append(f"- {s}")
        lines.append("")
    if key_risks:
        lines.append("**Top 主要风险 / 不足（最多列出 3 条）**")
        for r in key_risks[:3]:
            lines.append(f"- {r}")
        lines.append("")

    # 关键补充材料 / 建议：直接从 recommendations 抽几条
    if recs:
        lines.append("**关键后续建议 / 需补充材料要点（节选 3–5 条）**")
        for r in recs[:5]:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


# ========= 维度问答部分 =========
def build_qa_section_from_final_payload(
    final_payload: dict,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    输入：postproc/final_payload.json（dict）
    输出：Markdown 文本（每个维度下：
          综合得分 + 打分依据摘要 + 行业通识要点 + 精选问题/回答明细）
    - 每个维度只展示前 MAX_QA_PER_DIM 条问答
      （按 confidence + alignment 排序，从高到低截断）
    - 逐题不再重复展示 general_insights（避免和维度级/专家意见重复啰嗦）
    """
    dims = final_payload.get("dimensions", {}) or {}

    lines = []
    lines.append("## 2. 维度问答与打分依据")
    lines.append("")
    lines.append("> 本部分基于 post_processing 选中的问答结果生成，用于支撑专家评审结论的溯源。")
    lines.append("> 如阅读时间有限，可主要关注「0. 一页摘要」与「1. AI 专家总体评审」，本部分主要面向技术评审与审计。")
    if _output_calibration_enabled(metrics):
        lines.append(
            "> 各维度标题中的「综合得分 /100」来自 `final_payload`，与 post_processing 写入的**校准后**维度均值一致；"
            "逐题「置信度」「对齐度」等元字段为模型/后处理原始信号，**未**做输出校准仿射。"
        )
    lines.append("")

    for dim in DIM_ORDER:
        block = dims.get(dim) or {}
        qas = block.get("qas") or []
        if not qas:
            continue

        # 按置信度 + 对齐度排序，截断为前 MAX_QA_PER_DIM 条
        def _qa_key(qa_item):
            try:
                conf = float(qa_item.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            try:
                align = float(qa_item.get("alignment") or 0.0)
            except Exception:
                align = 0.0
            return (conf + align)

        qas_sorted = sorted(qas, key=_qa_key, reverse=True)
        qas_sorted = qas_sorted[:MAX_QA_PER_DIM]

        zh_label = DIM_LABELS_ZH.get(dim, dim)
        score = block.get("score", 0.0)  # 已是 0–100 之间的小数
        rationales = block.get("rationales") or []
        gi_dim = block.get("general_insights") or []

        # 维度标题 + 分数
        lines.append(f"### 维度：{zh_label}（{dim}） · 综合得分：{_fmt_float(score, 1)} / 100")
        lines.append("")

        # 打分依据（来源于 post_processing 的 strengths 聚合）
        if rationales:
            lines.append("**打分依据（摘要）**")
            for r in rationales:
                r = str(r).strip()
                if r:
                    lines.append(f"- {r}")
            lines.append("")

        # 维度级 industry general insights（来自 general_insights 聚合层）
        if gi_dim:
            lines.append("**行业通识要点（不代表本项目已达成，仅作参照）**")
            for g in gi_dim[:8]:
                g = str(g).strip()
                if g:
                    lines.append(f"- {g}")
            lines.append("")

        # 逐题问答（已按重要性排序 + 截断）
        for idx, qa in enumerate(qas_sorted, start=1):
            q_text = (qa.get("q") or "").strip()
            ans = (qa.get("answer") or "").strip()
            provider = (qa.get("provider") or "").strip()
            model = (qa.get("model") or "").strip()
            conf = qa.get("confidence", 0.0)
            align = qa.get("alignment", 0.0)
            drift = qa.get("dimension_drift", 0.0)
            claims = qa.get("claims") or []
            evids = qa.get("evidence_hints") or []

            lines.append("")
            lines.append(f"#### Q{idx}. {q_text}")
            lines.append("")
            meta_parts = []
            if provider:
                meta_parts.append(f"来源模型：{provider}")
            if model:
                meta_parts.append(model)
            meta_parts.append(f"置信度：{_fmt_float(conf, 2)}")
            meta_parts.append(f"对齐度：{_fmt_float(align, 2)}")
            meta_parts.append(f"维度漂移：{_fmt_float(drift, 2)}")

            lines.append("_" + " ｜ ".join(meta_parts) + "_")
            lines.append("")

            # 核心要点（claims）
            if claims:
                lines.append("**核心要点（claims）**")
                for c in claims:
                    c = str(c).strip()
                    if c:
                        lines.append(f"- {c}")
                lines.append("")

            # 选中回答
            lines.append("**选中回答**")
            lines.append("")
            if ans:
                lines.append(ans)
            else:
                lines.append("_（无有效回答）_")
            lines.append("")

            # 证据线索（evidence_hints）
            if evids:
                lines.append("**证据线索（evidence_hints）**")
                for e in evids:
                    e = str(e).strip()
                    if e:
                        lines.append(f"- {e}")
                lines.append("")

            # ⚠️ 不再展示逐题 general_insights，避免与维度级 / 专家意见重复啰嗦
            # gi_q = qa.get("general_insights") or []
            # if gi_q:
            #     lines.append("**行业通识经验（general_insights）**")
            #     for g in gi_q[:6]:
            #         g = str(g).strip()
            #         if g:
            #             lines.append(f"- {g}")
            #     lines.append("")

    return "\n".join(lines)


# ========= 主流程 =========
def main():
    ap = argparse.ArgumentParser(description="Generate final integrated markdown report (executive summary + expert opinion + Q&A, v3).")
    ap.add_argument("--pid", type=str, default="", help="提案 ID；若缺省则自动检测最新的一个。")
    args = ap.parse_args()

    pid = args.pid.strip() or detect_latest_pid()
    if not pid:
        raise RuntimeError(
            "未检测到可用项目：需要至少存在 "
            "`refined_answers/<pid>/postproc/final_payload.json` 与 "
            "`expert_reports/<pid>/ai_expert_opinion.md`。"
        )

    # ---------- 路径 ----------
    refined_dir = REFINED_ROOT / pid
    postproc_dir = refined_dir / "postproc"
    fp_path = postproc_dir / "final_payload.json"

    expert_dir = EXPERT_ROOT / pid
    expert_md_path = expert_dir / "ai_expert_opinion.md"
    expert_json_path = expert_dir / "ai_expert_opinion.json"

    if not fp_path.exists():
        raise FileNotFoundError(f"未找到 final_payload.json：{fp_path}")
    if not expert_md_path.exists():
        raise FileNotFoundError(f"未找到专家评审 Markdown：{expert_md_path}")
    # ai_expert_opinion.json 缺失时不会报错，只是无法生成一页摘要
    expert_json = {}
    if expert_json_path.exists():
        expert_json = load_json(expert_json_path)

    metrics: Dict[str, Any] = {}
    metrics_json_path = postproc_dir / "metrics.json"
    if metrics_json_path.exists():
        metrics = load_json(metrics_json_path)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_ROOT / f"{pid}_final_report.md"

    # ---------- 读取数据 ----------
    final_payload = load_json(fp_path)
    expert_md_raw = expert_md_path.read_text(encoding="utf-8")

    # ---------- 组装报告 ----------
    _write_progress(0, 4, pid)
    report_lines = []

    # 顶层标题
    report_lines.append(f"# 项目综合评审报告 · {pid}")
    report_lines.append("")
    report_lines.append(f"_生成时间：{now_str()}_")
    report_lines.append("")
    report_lines.append("_本报告由 AI 辅助评审系统自动生成，供内部专家和决策委员会参考使用。_")
    report_lines.append("")

    # 0. 一页摘要（如果有 ai_expert_opinion.json）
    exec_summary_md = build_executive_summary(expert_json, metrics)
    if exec_summary_md:
        report_lines.append(exec_summary_md)
        report_lines.append("")
    _write_progress(1, 4, pid)

    # 1. AI 专家总体评审（来自 ai_expert_opinion.md）
    report_lines.append("## 1. AI 专家总体评审（详细版）")
    report_lines.append("")
    report_lines.append(adjust_expert_markdown(expert_md_raw))
    report_lines.append("")
    _write_progress(2, 4, pid)

    # 2. 维度问答与打分依据（来自 final_payload.json 的问题 + 选中的一个回答）
    qa_section = build_qa_section_from_final_payload(final_payload, metrics)
    report_lines.append(qa_section)
    report_lines.append("")
    _write_progress(3, 4, pid)

    final_md = "\n".join(report_lines)
    out_path.write_text(final_md, encoding="utf-8")
    _write_progress(4, 4, pid)

    print(f"✅ Final report generated -> {out_path}")


if __name__ == "__main__":
    main()
