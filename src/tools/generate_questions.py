# -*- coding: utf-8 -*-
"""
Stage 3 · 维度定制化问题生成器（generate_questions.py · v3.1）

输入：
  - src/data/extracted/<proposal_id>/dimensions_v2.json

输出（权威路径 + 兼容副本）：
  1) 详细版（保留 qid/aspect/answer_type/links_to 等全量信息，方便调试与后续扩展）：
     - src/data/questions/<proposal_id>/generated_questions_detail.json

  2) 简化版（供 llm_answering / API 使用；与 server 传入的 --qs_file 一致）：
     - src/data/questions/<proposal_id>/generated_questions.json

  3) 同步写入同一简化 JSON 的副本（兼容 search_by_dimension、post_processing 等仍读全局路径的工具）：
     - src/data/config/question_sets/generated_questions.json
     结构示例：
     {
       "proposal_id": "XXX",
       "generated_at": "...",
       "model": "...",
       "provider": "...",
       "team": {
         "dimension": "team",
         "questions": ["问题1", "问题2", "..."],
         "search_hints": [],
         "source_proposal_id": "XXX"
       },
       ...
     }

核心特性（在 v3 基础上的改动）：
  - 问题必须显式“锚定”到该维度的 key_points / risks / mitigations（通过 links_to 索引）。
  - 每个维度鼓励至少 2 个 rating 问题 + 多个 analysis 问题，为后续打分和专家意见服务。
  - 问题设计显式基于 payload 内容（不允许脱离 dimensions_v2 瞎飞）。
  - 新增：“信息缺失处理规则”和“平台/中长期视角问题”的约束，减少后续 LLM 回答时的幻觉风险。
  - 新增：从 prepared/<proposal_id>/ 的 full_text 或 pages.json 按摘要要点检索 Top-K 原文摘录，与 dimensions 摘要分层注入出题 prompt。
  - 新增：写盘前「问题审核」——对 full_text 做 grounding 打分；默认再启用 LLM 分类/改述（QUESTION_AUDIT_LLM，可用环境变量或 --no_question_audit_llm 关闭）。
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ========== 路径配置 ==========

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROGRESS_FILE = DATA_DIR / "step_progress.json"


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass
QUESTIONS_DIR = DATA_DIR / "questions"
CONFIG_QS_DIR = DATA_DIR / "config" / "question_sets"

# ========== LLM 配置 ==========

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
PROVIDER = os.getenv("PROVIDER", "openai").lower()


def _get_domain_config():
    """Load config from centralized config system."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.config import get_config
        return get_config()
    except Exception:
        return None


def _get_dimension_names():
    cfg = _get_domain_config()
    if cfg:
        return cfg.dimension_names
    return ["team", "objectives", "strategy", "innovation", "feasibility"]


DIMENSION_NAMES = _get_dimension_names()

# 一些“平台/中长期视角”的 aspect id，用于简单 sanity check
PLATFORM_ASPECT_IDS = {
    "platform_and_extensibility",
    "scaling_and_globalization",
}

def _looks_like_team_bio_question(q_zh: str, q_en: str) -> bool:
    """简单用关键词判断一个问题是不是在问团队履历 / 经验（仅用于 team 维度兜底）。"""
    q_zh = q_zh or ""
    q_en = (q_en or "").lower()

    kw_zh = [
        "团队", "核心成员", "项目负责人", "负责人",
        "履历", "背景", "经历", "经验",
        "法规经验", "合规经验", "产业化", "转化", "商业化",
        "带队", "主导", "项目记录", "成功案例", "交付", "操盘", "任职",
        "行业经验", "大客户", "重大项目",
        # 医药场景常见词：仅作识别用，不强制所有材料都出现
        "临床", "pi",
    ]
    kw_en = [
        "team", "core team", "core member", "leader", "leadership",
        "principal investigator", "pi",
        "background", "track record", "experience", "experiences",
        "clinical", "regulatory", "commercialization", "industrial",
    ]

    return any(k in q_zh for k in kw_zh) or any(k in q_en for k in kw_en)


def _payload_suggests_healthcare_context(dim_payload: Dict[str, Any]) -> bool:
    """
    True if summary/key_points/risks text suggests medical, clinical, or pharma context.
    Used to gate clinical-specific question wording (team/objectives); non-healthcare
    materials should use general commercial delivery wording.
    """
    parts: List[str] = []
    s = dim_payload.get("summary")
    if isinstance(s, str):
        parts.append(s)
    for k in ("key_points", "risks", "mitigations"):
        v = dim_payload.get(k)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
    blob = " ".join(parts)
    if not blob.strip():
        return False
    blob_lower = blob.lower()
    zh_markers = (
        "临床", "患者", "适应症", "入组", "疗效", "试验", "三期", "二期", "一期",
        "药物", "新药", "仿制药", "制剂", "医疗器械", "诊断试剂", "医院", "诊疗",
        "医生", "处方", "药理", "毒理", "生物标志", "CRO", "GCP", "注册申报",
        "IND", "NDA", "BLA", "MAH", "真实世界",
    )
    en_markers = (
        "clinical trial", "phase 1", "phase 2", "phase 3", "patient", "patients",
        "endpoint", "pivotal", "indication", "pharma", "hospital", "fda", "ema",
    )
    if any(m in blob for m in zh_markers):
        return True
    if any(m in blob_lower for m in en_markers):
        return True
    return False


def _looks_like_market_question(q_zh: str, q_en: str) -> bool:
    """简单用关键词判断一个问题是不是在问市场 / 竞争 / 定价（用于 strategy/objectives 兜底）。"""
    q_zh = q_zh or ""
    q_en = (q_en or "").lower()

    kw_zh = [
        "市场", "市场规模", "市场分析", "市场机会",
        "竞争", "竞品", "竞争对手", "竞争格局",
        "客户", "患者群体", "目标人群",
        "cagr", "增长率", "销售", "营收", "定价", "报销",
    ]
    kw_en = [
        "market", "market size", "market analysis", "market opportunity",
        "competitive", "competition", "competitor", "competitors",
        "customer", "customers", "patient population", "target population",
        "cagr", "growth", "sales", "revenue", "pricing", "reimbursement",
    ]

    return any(k in q_zh for k in kw_zh) or any(k in q_en for k in kw_en)

# ========== 维度专属配置 ==========


def _build_dimension_config() -> Dict[str, Dict[str, Any]]:
    """
    Build DIMENSION_CONFIG from the centralized config system.
    Falls back to the same generic review skeleton as default.yaml if config is unavailable.
    """
    cfg = _get_domain_config()
    if cfg:
        result = {}
        for dim_name in cfg.dimension_names:
            d = cfg.get_dimension_config_dict(dim_name)
            if d:
                result[dim_name] = d
        if result:
            return result

    # Hardcoded fallback：与 default.yaml 对齐的「通用评审语义骨架」（config 不可用时）
    return {
        "team": {
            "min_q": 6, "max_q": 9,
            "focus_zh": "通用评审骨架：执行主体是否清晰；关键角色能力与分工；治理与外部协作是否支撑披露目标（以材料为准，不预设行业）。",
            "aspects": [
                {"id": "leadership_experience", "desc_zh": "牵头人/负责人对同类或同规模任务的执行记录与可核验证据（材料已披露部分）"},
                {"id": "domain_expertise", "desc_zh": "团队能力与任务范围的匹配度：知识深度、资质与关键技能覆盖（材料已披露部分）"},
                {"id": "collaboration_network", "desc_zh": "外部协作与关键依赖：合作方角色、互补性与集中度风险（材料已披露部分）"},
                {"id": "governance_and_decision_making", "desc_zh": "治理与决策：分工授权、决策链、质量与合规内控安排（材料已披露部分）"},
                {"id": "team_capacity_and_bandwidth", "desc_zh": "投入与承载：人力/时间/管理带宽与阶段目标的匹配（材料已披露部分）"},
            ],
        },
        "objectives": {
            "min_q": 6, "max_q": 9,
            "focus_zh": "通用评审骨架：目标是否可检验；阶段结果与成功判据；范围边界与优先级；与约束条件的自洽性（以材料为准）。",
            "aspects": [
                {"id": "overall_goal_clarity", "desc_zh": "总体目标陈述是否明确、可界定成功/失败（材料已披露部分）"},
                {"id": "milestones_and_timeline", "desc_zh": "阶段划分与时间节点是否具体、可执行与可检查（材料已披露部分）"},
                {"id": "outcome_and_success_metrics", "desc_zh": "成功判据与测度：指标、口径与证据来源是否交代清楚（材料已披露部分）"},
                {"id": "scope_and_prioritization", "desc_zh": "范围边界与不做什么；资源约束下的优先级取舍（材料已披露部分）"},
                {"id": "realism_and_ambition_balance", "desc_zh": "目标强度与可得资源、外部环境之间的一致性（材料已披露部分）"},
            ],
        },
        "strategy": {
            "min_q": 7, "max_q": 10,
            "focus_zh": "通用评审骨架：主路径与备选；对外价值实现与关键接口；合作与资源编排；证据与合规边界（以材料为准，不预设单一业态）。",
            "aspects": [
                {"id": "technical_strategy", "desc_zh": "主方案与备选方案：方法选择、关键假设与依赖（材料已披露部分）"},
                {"id": "commercialization_and_market_entry", "desc_zh": "价值到达路径：客户/用户/采购或付费关系、渠道与交付形态（材料已披露部分；无则评信息缺口）"},
                {"id": "partnership_and_business_model", "desc_zh": "合作与交易结构：权责、收益分配、排他与退出（材料已披露部分）"},
                {"id": "data_and_evidence_strategy", "desc_zh": "关键结论所依赖的数据、来源、留存与合规/保密边界（材料已披露部分）"},
                {"id": "scaling_and_globalization", "desc_zh": "扩张或跨区域/跨场景推广的前提、节奏与约束（材料已披露部分）"},
            ],
        },
        "innovation": {
            "min_q": 6, "max_q": 9,
            "focus_zh": "通用评审骨架：相对基准的差异；可辩护优势；保护与证据链；可扩展边界与外部替代风险（以材料为准）。",
            "aspects": [
                {"id": "novelty_vs_state_of_art", "desc_zh": "相对行业/惯例或对标基准的实质性差异点（材料已披露部分）"},
                {"id": "differentiation_and_competitive_edge", "desc_zh": "可辩护的差异化要点及与替代方案的关系（材料已披露部分）"},
                {"id": "ip_and_protection", "desc_zh": "可保护知识与合规资产：权利、合同安排、排他或关键 know-how（材料已披露部分）"},
                {"id": "evidence_strength_for_innovation", "desc_zh": "主张的支撑强度：验证设计、样本、独立性与可复核性（材料已披露部分）"},
                {"id": "platform_and_extensibility", "desc_zh": "方案的复用边界、接口与生态位：是否具备可扩展结构（材料已披露部分）"},
                {"id": "risk_of_obsolescence", "desc_zh": "外部变化下被替代、淘汰或规则变化的主要风险与窗口（材料已披露部分）"},
            ],
        },
        "feasibility": {
            "min_q": 7, "max_q": 10,
            "focus_zh": "通用评审骨架：资源与预算；执行编排；风险与缓解；关键障碍与时间—投入节拍（以材料为准）。",
            "aspects": [
                {"id": "resources_and_infrastructure", "desc_zh": "关键资源与基础设施的可得性、独占性与持续性（材料已披露部分）"},
                {"id": "funding_and_budget_planning", "desc_zh": "资金与预算结构：来源、用途、集中度与缓冲（材料已披露部分）"},
                {"id": "operational_execution_plan", "desc_zh": "从计划到落地的步骤、责任主体与关键路径（材料已披露部分）"},
                {"id": "risk_management", "desc_zh": "主要风险与监测指标；缓解与预案是否对应（材料已披露部分）"},
                {"id": "implementation_barriers", "desc_zh": "实施障碍：监管、供应链、组织、数据或外部依赖等（材料已披露部分）"},
                {"id": "timeline_and_resource_alignment", "desc_zh": "时间表与投入节拍、里程碑与资源曲线是否自洽（材料已披露部分）"},
            ],
        },
    }


DIMENSION_CONFIG: Dict[str, Dict[str, Any]] = _build_dimension_config()


# ========== 工具函数 ==========

def find_latest_extracted_proposal_id() -> str:
    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"未找到 extracted 目录: {EXTRACTED_DIR}")

    candidates = []
    for d in EXTRACTED_DIR.iterdir():
        if d.is_dir():
            candidates.append((d.stat().st_mtime, d.name))

    if not candidates:
        raise FileNotFoundError(f"extracted 目录下没有任何子目录: {EXTRACTED_DIR}")

    proposal_id = max(candidates, key=lambda x: x[0])[1]
    print(f"[INFO] [auto] 选中最新提案 ID: {proposal_id}")
    return proposal_id


def load_dimensions(proposal_id: str) -> Dict[str, Any]:
    path = EXTRACTED_DIR / proposal_id / "dimensions_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"dimensions_v2.json 不存在，请先运行 build_dimensions_from_facts.py: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"[INFO] 读取 dimensions_v2.json 成功: {path}")
    return data


def safe_truncate(text: str, max_len: int = 4000) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...（后续内容已截断，仅供生成问题时参考）"


def build_dimension_payload(dim_name: str, dim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 dimensions_v2.json 中构造给 LLM 的 payload，并附带 meta 统计信息。
    """
    summary = dim_data.get("summary", "") or ""
    key_points = dim_data.get("key_points", []) or []
    risks = dim_data.get("risks", []) or []
    mitigations = dim_data.get("mitigations", []) or []

    key_points_trunc = [safe_truncate(k, 400) for k in key_points[:12]]
    risks_trunc = [safe_truncate(r, 400) for r in risks[:8]]
    mitigations_trunc = [safe_truncate(m, 400) for m in mitigations[:8]]

    # ✅ 新增：把 risk_coverage 取出来，一起写进 meta
    risk_coverage = dim_data.get("risk_coverage", {})

    payload = {
        "dimension": dim_name,
        "summary": safe_truncate(summary, 1200),
        "key_points": key_points_trunc,
        "risks": risks_trunc,
        "mitigations": mitigations_trunc,
        "meta": {
            "key_points_count": len(key_points),
            "risks_count": len(risks),
            "mitigations_count": len(mitigations),
            "risk_coverage": risk_coverage,   # 👈 新增这一行
        },
    }
    return payload


def get_openai_client() -> OpenAI:
    if PROVIDER != "openai":
        print(f"[WARN] 当前仅实现 openai provider，PROVIDER={PROVIDER} 仍将使用 OpenAI。")
    return OpenAI()


# ========== Prompt 模板（强化防幻觉 + 平台视角） ==========

QUESTION_PROMPT_TEMPLATE = """
你现在扮演“项目评审专家 + 问卷设计顾问”的角色，负责为一个 AI 辅助评审系统设计【某一个维度】的问题集。

【系统背景（简要）】
- 系统会先从提案中抽取五个维度的摘要：team / objectives / strategy / innovation / feasibility。
- 你当前只负责其中一个维度：{dimension_name}。
- 我会给你这个维度的摘要 payload（summary + key_points + risks + mitigations），
  你必须【围绕这些内容出题】，而不是泛泛而谈。

【当前维度】
- 维度名称（英文 key）：{dimension_name}
- 该维度需要重点关注：{dimension_focus_zh}

【该维度的 aspect 配置说明】
- 我会给你一个 JSON 数组 aspects，每个元素形如：
  {{
    "id": "leadership_experience",
    "desc_zh": "项目负责人或核心骨干的领导经验与往期重大项目执行记录（行业不限）"
  }}
- 这些 aspects 是你可以用来“聚焦发问”的子方向。
- 你需要先阅读 payload，判断哪些 aspects 与当前提案此维度的信息最相关，然后在这些方面设计问题。
- 不要求每个 aspect 都出题，但至少应覆盖 3–5 个最关键的 aspects。
- 对于明显偏“平台/中长期扩展”的 aspects（例如 platform_and_extensibility、scaling_and_globalization），
  请至少设计 1 个从中长期或平台化视角出发的问题：
  - 如果 payload 中有相关信息，就围绕这些信息发问；
  - 如果 payload 完全没有相关信息，可以把问题设计为“评估该信息缺失带来的风险”。

【问题设计的内容绑定要求（极其重要）】
1. 你必须显式利用 payload 中的 key_points / risks / mitigations 来设计问题：
   - 至少一半的问题需要能指向一个或多个 key_points；
   - 如果 payload 中存在 risks 条目，至少设计 2 个问题专门围绕这些风险展开；
   - 如果 payload 中存在 mitigations 条目，至少设计 1 个问题评估这些应对措施的充分性。
2. 【实体名硬约束】问题文本中：
   - 严禁出现 payload 中完全没有出现过的具体公司、机构、工厂、渠道商、医院、平台、品牌、基金、国家或城市名称；
   - 若需泛指合作方但 payload 未给出具体名称，只能使用与行业相符的中性占位（如「某关键供应商」「某区域渠道伙伴」
     「某大型采购方」「某合作研发机构」；**仅当 payload 已体现医药/医疗场景时**，才可使用「某制药企业」「某医疗机构」等；
   - 如果 payload 中已经出现了某个实体名称（例如某家公司的正式名称），你可以在问题中以【完全相同的写法】引用它，
     但不得新增其它实体名称，也不得为人物杜撰新的外文姓名。
3. 如果需要讨论“目标客户类型”“市场规模”“毛利率/产能/交付周期”或（**仅当原文出现医药语境时**）“疗效或临床终点”等，
   但 payload 没给精确数字或具体对象：
   - 问题可以要求后续回答者“根据提案中已有信息进行定性分析或区间估计”，
   - 并在问题中明确加入类似措辞：
     “如提案未给出具体数值/名单，请在回答时先说明信息缺失，再分析其可能影响。”
4. 若用户消息中附有「原文摘录（检索）」：仅用于与维度摘要交叉核对措辞与数字；题目中出现的**具体机构名、数字、产品名**
   须能在「摘要 payload」与「原文摘录」的并集中找到依据；不得单独依据摘录编造 payload 未出现的断言。

【维度特定要求（team / objectives / strategy）——全行业商业评审；勿默认医药叙事】
- 如果当前维度是 team：
  - 至少设计 2 个高优先级（priority=1 或 2）的“履历与交付能力驱动”问题，围绕 payload 中对核心成员、负责人、
    关键岗位（如技术、运营、销售、供应链、财务合规等）的**原文已披露**履历、项目经验、治理与分工、资源投入等 key_points 发问。
  - 综合判断应表述为：团队在**本项目所属行业与交付形态**下推进里程碑、履约与客户/监管要求的能力（勿预设“临床阶段”或“医院场景”）。
  - **仅当** summary / key_points / risks 中实际出现临床、患者、适应症、试验、注册、诊疗、药物或医疗器械等医药语境时，
    才允许使用“临床推进”“患者人群”“监管路径”等医药专用措辞；否则一律用可验证的交付记录、重大合同与回款、产线/门店爬坡、
    招投标与集采、数字化上线、跨境合规等**与原文一致**的行业中性表述。
- 如果当前维度是 strategy：
  - 如果 payload 的 key_points 中出现“市场 / market / CAGR / 竞争 / 客户 / 销售 / 渠道 / 定价”等相关内容，
    至少设计 1 个高优先级的“市场驱动”问题，用于评估项目在目标细分市场中的定位、竞争压力与进入 / 商业化策略。
  - 该问题的 links_to.key_points 中，至少要包含一个与市场分析相关的条目。
- 如果当前维度是 objectives：
  - 若 key_points 中出现**一般商业**目标与市场机会（营收、份额、产能、门店/网点、GMV、出海、成本与毛利、ESG 等），
    至少设计 1 个问题，评估「披露目标 ↔ 所声称市场/资源窗口/约束条件」的匹配度与可验证性，并允许在信息不足时先指出缺口。
  - **仅当** key_points 等中实际出现疾病负担、患者人群、未满足医疗需求、临床终点或注册里程碑等表述时，
    才额外（或替代上述）设计 1 个问题，聚焦「项目目标与未满足临床需求 / 监管与市场准入路径」的匹配度；否则不要使用“临床需求”“患者”等未在原文出现的概念。

【信息缺失时的处理要求】
- 你设计的问题本身要允许“提案信息不足”的情况：
  - 对于依赖具体指标、具体国家/公司/里程碑细节的问题，
    请在中文问题中显式加入类似提示：
    “如果提案中未对 X 进行详细说明，请在回答时先指出这一信息缺失，再讨论其对评估的影响。”
  - 英文问题中给出等价表述，例如：
    "If the proposal does not provide sufficient details on X, please first state this information gap and then discuss its impact on the assessment."
- 严禁通过问题文本暗示“这些细节一定已经给出”，从而诱导后续回答者编造事实。

【links_to 字段的要求】
- 对于每个问题，请在 links_to 中标出它主要针对 payload 中哪些条目：
  - links_to.key_points: 使用 key_points 数组的下标列表（例如 [0,2]）；
  - links_to.risks: 使用 risks 数组的下标列表；
  - links_to.mitigations: 使用 mitigations 数组的下标列表。
- 如果某个问题主要是针对某个“缺失信息/盲区”，则可以让三个列表都为空。
- 下标从 0 开始，必须是整数。

【问题设计原则】
1. 针对性：
   - 问题必须紧扣当前维度的职责和 aspects，而不是泛泛而谈。
2. 可回答性：
   - 问题应该可以在“通用领域知识 + 维度摘要 payload”的基础上回答。
   - 避免依赖你看不到的隐性细节。
   - 不要在问题中要求回答者给出提案中完全未提到的“精确数值”或“完整公司名单”；
     如确有需要，必须附带前述“信息缺失时的处理提示”。
3. 结构化用途：
   - 问题类型（answer_type）从 ["analysis", "rating", "yes_no", "open"] 中选择：
     - "analysis": 要求给出分析性文字（例如“请分析……的主要优势和不足”）；
     - "rating": 可以在 1-5 分等尺度上打分（例如“在 1-5 分尺度上评价……的成熟度”）；
     - "yes_no": 判断类问题（建议后续附带理由说明）；
     - "open": 开放提问，不强制结构。
   - 同一维度的问题要覆盖不同 aspects，而不是十个问题都问同一个点。
4. 语言与形式：
   - 每个问题需要同时给出中文和英文版本：
     - question_zh：面向专家的正式中文问题；
     - question_en：自然、专业的英文问题，是 question_zh 的等价翻译。
   - 问题文本中可以适当提及“该项目”“该团队”“该技术方案”等泛称，不用重复 payload 里的长段文字。

【数量与优先级要求】
- 我会给出推荐问题数量区间（例如 [6, 9]），你要尽量控制在这个区间内：
  - 对信息较丰富的维度，靠近上限；
  - 对信息较少的维度，可以偏下限。
- priority 取值含义：
  - 1：高优先级（核心问题，系统一定会问）；
  - 2：中优先级（建议问）； 
  - 3：可选（在有余量时才问）。
- 每个维度至少要有 3 个 priority=1 的问题，其余可以为 2 或 3。

【类型配比的硬性期待（请尽量满足）】
- 每个维度至少设计 2 个 rating 问题，用于后续量化打分。
- 每个维度至少设计 3 个 analysis 问题，用于深入文字分析。
- 其余问题可以是 yes_no 或 open（但建议 yes_no 问题在说明中要求回答者给理由）。

【输出 JSON 结构要求】
- 你必须输出一个 JSON 对象，顶层只有一个键 "questions"。
- "questions" 对应一个数组，数组中的每个元素是一个问题对象，结构为：

  {{
    "aspect": "string，aspect 的 id，例如：leadership_experience / technical_strategy 等",
    "question_zh": "string，中文问题",
    "question_en": "string，英文问题",
    "answer_type": "analysis" | "rating" | "yes_no" | "open",
    "priority": 1 | 2 | 3,
    "links_to": {{
      "key_points": [0, 2],
      "risks": [],
      "mitigations": []
    }}
  }}

- 不要输出任何 JSON 以外的文字，不要解释，不要加注释。
"""

# ========== 调用 LLM 生成问题 ==========
def call_llm_for_dimension_questions(
    client: OpenAI,
    dimension_name: str,
    dim_payload: Dict[str, Any],
    min_q: int,
    max_q: int,
    dim_config: Dict[str, Any],
    proposal_id: str,
) -> List[Dict[str, Any]]:
    """
    调用 LLM，为某个维度生成【定制化】问题列表（带 links_to）。
    可选注入 prepared 原文检索摘录（与 dimensions 摘要分层）。
    """

    key_points = dim_payload.get("key_points", []) or []
    kp_cnt = len(key_points)

    # 根据信息量微调区间，但不突破维度 config 的 min/max
    if kp_cnt >= 8:
        target_min, target_max = max(min_q, dim_config["min_q"]), dim_config["max_q"]
    elif 4 <= kp_cnt <= 7:
        target_min, target_max = max(min_q, dim_config["min_q"] - 1), min(max_q, dim_config["max_q"])
    elif kp_cnt > 0:
        target_min, target_max = max(min_q, 4), min(max_q, dim_config["max_q"] - 1)
    else:
        target_min, target_max = min_q, min(max_q, 6)

    aspects = dim_config.get("aspects", [])
    aspects_str = json.dumps(aspects, ensure_ascii=False, indent=2)

    payload_str = json.dumps(dim_payload, ensure_ascii=False, indent=2)
    focus_zh = dim_config.get("focus_zh", "")

    # 当前维度的内容概览（给模型一个“量级感”）
    overview = dim_payload.get("meta", {})
    overview_str = json.dumps(overview, ensure_ascii=False, indent=2)

    prompt = (
        QUESTION_PROMPT_TEMPLATE
        .replace("{dimension_name}", dimension_name)
        .replace("{dimension_focus_zh}", focus_zh)
    )

    rq_parts: List[str] = [str(dim_payload.get("summary") or "")]
    for key in ("key_points", "risks", "mitigations"):
        seq = dim_payload.get(key) or []
        if isinstance(seq, list):
            rq_parts.extend(str(x) for x in seq if isinstance(x, str) and str(x).strip())
    retrieval_query = "\n".join(p for p in rq_parts if p)

    snippet_block = ""
    try:
        from src.tools.review_text_snippets import format_snippets_for_prompt

        snippet_block = format_snippets_for_prompt(
            BASE_DIR, proposal_id, retrieval_query, kp_cnt
        )
    except Exception as e:
        print(f"[WARN] 原文摘录检索失败（{dimension_name}）: {e}")

    snippet_section = ""
    if snippet_block.strip():
        snippet_section = "\n\n" + snippet_block.strip() + "\n"

    user_content = (
        prompt
        + "\n\n=== 该维度的 aspects 配置 ===\n"
        + aspects_str
        + "\n\n=== 当前维度内容概览（统计信息）===\n"
        + overview_str
        + "\n\n=== 当前维度的摘要 payload ===\n"
        + payload_str
        + snippet_section
        + "\n\n=== 生成数量提示 ===\n"
        + f"- 推荐问题数量区间：[{target_min}, {target_max}]，请尽量控制在这个范围内。\n"
        + "- 至少覆盖 3–5 个最关键的 aspects（包括至少 1 个平台/中长期视角的 aspect，如果存在）。\n"
        + "- 至少生成 2 个 rating 问题 和 3 个 analysis 问题（如果信息极少，可以适当降低，但请尽量满足）。\n"
        + "- 请严格按前述 JSON 结构输出。"
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的项目评审专家，负责为 AI 系统设计结构化问题。适用于任何领域的项目提案。",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=2200,
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] 维度 {dimension_name} 问题生成 JSON 解析失败，原始内容如下：")
        print(raw)
        raise e

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    cleaned: List[Dict[str, Any]] = []
    rating_count = 0
    analysis_count = 0
    linked_to_kp_count = 0
    platform_aspect_used = 0

    for q in questions:
        if not isinstance(q, dict):
            continue

        aspect = str(q.get("aspect", "")).strip()
        q_zh = str(q.get("question_zh", "")).strip()
        q_en = str(q.get("question_en", "")).strip()
        answer_type = str(q.get("answer_type", "analysis")).strip().lower()
        priority = q.get("priority", 2)
        links_to = q.get("links_to") or {}

        if not q_zh or not q_en:
            continue

        if answer_type not in ["analysis", "rating", "yes_no", "open"]:
            answer_type = "analysis"

        try:
            priority_int = int(priority)
        except Exception:
            priority_int = 2
        if priority_int < 1 or priority_int > 3:
            priority_int = 2

        if not aspect:
            aspect = "general"

        # 清洗 links_to 结构
        if not isinstance(links_to, dict):
            links_to = {}
        kp_idx = links_to.get("key_points", [])
        rk_idx = links_to.get("risks", [])
        mt_idx = links_to.get("mitigations", [])

        def _clean_index_list(v, max_len: int):
            if not isinstance(v, list):
                return []
            cleaned_idx = []
            for x in v:
                try:
                    ix = int(x)
                    if 0 <= ix < max_len:   # 既要非负，又不能越界
                        cleaned_idx.append(ix)
                except Exception:
                    continue
            return cleaned_idx

        kp_idx_clean = _clean_index_list(kp_idx, len(key_points))
        rk_idx_clean = _clean_index_list(
            rk_idx,
            len(dim_payload.get("risks", []) or [])
        )
        mt_idx_clean = _clean_index_list(
            mt_idx,
            len(dim_payload.get("mitigations", []) or [])
        )

        if kp_idx_clean:
            linked_to_kp_count += 1

        if answer_type == "rating":
            rating_count += 1
        if answer_type == "analysis":
            analysis_count += 1

        if aspect in PLATFORM_ASPECT_IDS:
            platform_aspect_used += 1

        cleaned.append(
            {
                "aspect": aspect,
                "question_zh": q_zh,
                "question_en": q_en,
                "answer_type": answer_type,
                "priority": priority_int,
                "links_to": {
                    "key_points": kp_idx_clean,
                    "risks": rk_idx_clean,
                    "mitigations": mt_idx_clean,
                },
            }
        )

    # ===== 维度特定兜底：确保至少有“简历驱动 / 市场驱动”问题 =====
    if cleaned and key_points:
        # --- team 维度：检查有没有“简历驱动”问题，没有就自动补一题 ---
        if dimension_name == "team":
            has_team_bio_q = any(
                _looks_like_team_bio_question(
                    q.get("question_zh", ""), q.get("question_en", "")
                )
                for q in cleaned
            )
            if not has_team_bio_q:
                print(f"[INFO] 维度 {dimension_name}: 未检测到明显的简历驱动问题，自动补充 1 题。")
                hc = _payload_suggests_healthcare_context(dim_payload)
                if hc:
                    extra_q_team = {
                        "aspect": "leadership_experience",
                        "question_zh": (
                            "基于提案中对核心成员和项目负责人的教育背景、临床与产业化相关经历及既往重大项目记录的描述，"
                            "您如何评价该团队在将本项目推进至监管要求下的验证、临床或商业化落地方面的整体能力？"
                            "如果提案对关键履历或可验证业绩描述不够具体，请在回答中先指出这一信息缺失，"
                            "并讨论其对评估结果的影响。"
                        ),
                        "question_en": (
                            "Based on the proposal's description of core team members and project leaders "
                            "(education, clinical and industrialization-related experience, and track record in "
                            "prior major programs), how would you assess the team's overall ability to advance "
                            "this project toward validation, clinical milestones or commercialization under "
                            "applicable regulatory expectations? If the proposal lacks specific or verifiable "
                            "track record, please first state this information gap and discuss its impact."
                        ),
                        "answer_type": "analysis",
                        "priority": 1,
                        "links_to": {
                            "key_points": list(range(len(key_points))),
                            "risks": [],
                            "mitigations": [],
                        },
                    }
                else:
                    extra_q_team = {
                        "aspect": "leadership_experience",
                        "question_zh": (
                            "基于提案中对核心成员与项目负责人（职责分工、相关行业或职能经历、可验证的重大项目或交付记录）"
                            "的描述，您如何评价该团队在履约本项目商业目标、里程碑以及质量与合规要求方面的综合能力？"
                            "若关键履历或可验证业绩描述不足，请在回答中先指出信息缺失，并讨论其对评估结果的影响。"
                        ),
                        "question_en": (
                            "Based on the proposal's description of core members and project leaders "
                            "(roles, relevant industry or functional experience, and verifiable track record on "
                            "major projects or deliveries), how would you assess the team's overall ability to "
                            "execute this commercial initiative against stated goals, milestones, and quality or "
                            "compliance expectations? If key credentials or verifiable performance are insufficient, "
                            "please first state the information gap and discuss its impact on your assessment."
                        ),
                        "answer_type": "analysis",
                        "priority": 1,
                        "links_to": {
                            "key_points": list(range(len(key_points))),
                            "risks": [],
                            "mitigations": [],
                        },
                    }
                cleaned.insert(0, extra_q_team)

        # --- strategy / objectives 维度：检查有没有“市场驱动”问题 ---
        if dimension_name in ("strategy", "objectives"):
            has_market_q = any(
                _looks_like_market_question(
                    q.get("question_zh", ""), q.get("question_en", "")
                )
                for q in cleaned
            )
            if not has_market_q:
                print(f"[INFO] 维度 {dimension_name}: 未检测到明显的市场驱动问题，自动补充 1 题。")
                if dimension_name == "strategy":
                    aspect_id = "commercialization_and_market_entry"
                else:
                    aspect_id = "overall_goal_clarity"

                extra_q_market = {
                    "aspect": aspect_id,
                    "question_zh": (
                        "结合提案中关于目标市场规模、增长率、主要竞争对手或目标客户群体的描述，"
                        "请评估本项目在所瞄准细分市场中的定位、竞争压力以及拟采用的市场进入/商业化策略是否合理。"
                        "如果提案中缺乏具体的市场规模或竞争格局数据，请在回答时先指出这一信息缺失，"
                        "并分析其对评估结论的影响。"
                    ),
                    "question_en": (
                        "Drawing on the proposal's description of the target market size, growth, main competitors "
                        "or target customer segments, how would you assess the project's positioning, competitive "
                        "pressure and the soundness of its market entry/commercialization strategy in the intended "
                        "niche? If the proposal does not provide concrete data on market size or the competitive "
                        "landscape, please first state this information gap and then discuss its impact on your "
                        "assessment."
                    ),
                    "answer_type": "analysis",
                    "priority": 1,
                    "links_to": {
                        "key_points": list(range(len(key_points))),
                        "risks": [],
                        "mitigations": [],
                    },
                }
                cleaned.insert(0, extra_q_market)

    # ✅ 先根据 target_min / target_max 控制问题数量（只对上限做硬控）
    if cleaned:
        if len(cleaned) > target_max:
            # 按 priority 排序，优先保留 1，再保留 2，最后 3
            cleaned.sort(key=lambda q: q.get("priority", 2))
            original_len = len(cleaned)
            cleaned = cleaned[:target_max]
            print(
                f"[INFO] 维度 {dimension_name}: LLM 生成 {original_len} 个问题，"
                f"已按 priority 截断为 {len(cleaned)} 个（上限 {target_max}）。"
            )
        elif len(cleaned) < target_min:
            print(
                f"[WARN] 维度 {dimension_name}: 实际只生成 {len(cleaned)} 个问题，"
                f"低于建议下限 {target_min}，如需补强建议后续人工加题。"
            )

    # 简单 sanity check：给你打日志，不强制重试
    if cleaned:
        if rating_count < 2:
            print(
                f"[WARN] 维度 {dimension_name}: rating 问题只有 {rating_count} 个，"
                f"可能不利于后续量化打分。"
            )
        if analysis_count < 3 and len(cleaned) >= 5:
            print(
                f"[WARN] 维度 {dimension_name}: analysis 问题只有 {analysis_count} 个，"
                f"可能不足以支撑深入文字分析。"
            )
        if linked_to_kp_count < len(cleaned) // 2:
            print(
                f"[WARN] 维度 {dimension_name}: 仅有 {linked_to_kp_count}/{len(cleaned)} "
                f"个问题显式链接到 key_points，建议人工抽检。"
            )
        if dimension_name == "innovation" and platform_aspect_used == 0:
            print(
                f"[WARN] 维度 {dimension_name}: 未检测到使用 platform_and_extensibility 等平台视角 aspect 的问题，"
                f"建议人工检查是否需要补充平台/中长期扩展相关问题。"
            )

    return cleaned

# ========== 主流程 ==========

def _env_flag(name: str, default: str = "1") -> bool:
    v = (os.getenv(name, default) or default).strip().lower()
    return v not in ("0", "false", "no", "off")


def run_generate_questions(
    proposal_id: str,
    min_q_per_dim: int = 5,
    max_q_per_dim: int = 10,
    *,
    enable_question_audit: bool = True,
    question_audit_use_llm: Optional[bool] = None,
):
    client = get_openai_client()
    dimensions = load_dimensions(proposal_id)

    all_dim_questions: Dict[str, Any] = {}

    _write_progress(0, len(DIMENSION_NAMES), proposal_id)
    for dim_idx, dim in enumerate(DIMENSION_NAMES):
        dim_data = dimensions.get(dim, {})
        dim_config = DIMENSION_CONFIG.get(dim)

        if not dim_config:
            print(f"[WARN] 维度 {dim} 没有配置 DIMENSION_CONFIG，将跳过。")
            _write_progress(dim_idx + 1, len(DIMENSION_NAMES), proposal_id)
            continue

        print(f"\n[INFO] 开始为维度 {dim} 生成问题 ...")

        dim_payload = build_dimension_payload(dim, dim_data)
        questions = call_llm_for_dimension_questions(
            client=client,
            dimension_name=dim,
            dim_payload=dim_payload,
            min_q=min_q_per_dim,
            max_q=max_q_per_dim,
            dim_config=dim_config,
            proposal_id=proposal_id,
        )

        dim_qs_with_id = []
        for idx, q in enumerate(questions, start=1):
            qid = f"{dim}_q{idx:02d}"
            item = dict(q)
            item["qid"] = qid
            item["dimension"] = dim
            dim_qs_with_id.append(item)

        print(
            f"[INFO] 维度 {dim} 生成问题数: {len(dim_qs_with_id)} "
            f"(推荐区间: [{dim_config['min_q']}, {dim_config['max_q']}])"
        )

        all_dim_questions[dim] = {
            "dimension": dim,
            "questions": dim_qs_with_id,
        }
        _write_progress(dim_idx + 1, len(DIMENSION_NAMES), proposal_id)

    audit_report: Dict[str, Any] = {}
    do_audit = bool(enable_question_audit) and _env_flag("QUESTION_AUDIT", "1")
    if question_audit_use_llm is True:
        use_llm_audit = True
    elif question_audit_use_llm is False:
        use_llm_audit = False
    else:
        use_llm_audit = _env_flag("QUESTION_AUDIT_LLM", "1")
    if do_audit:
        try:
            from src.tools.question_audit import audit_generated_questions_for_proposal

            all_dim_questions, audit_report = audit_generated_questions_for_proposal(
                project_root=BASE_DIR,
                proposal_id=proposal_id,
                dimension_names=list(DIMENSION_NAMES),
                dimensions=dimensions,
                all_dim_questions=all_dim_questions,
                openai_client=client if use_llm_audit else None,
                openai_model=OPENAI_MODEL,
                use_llm_audit=use_llm_audit,
            )
            print(
                f"[INFO] 问题审核完成（启发式 grounding"
                f"{' + LLM' if use_llm_audit else ''}）；"
                f"详见输出目录 question_audit_report.json"
            )
        except Exception as e:
            print(f"[WARN] 问题审核阶段失败（已跳过）: {e}")
            audit_report = {"error": str(e)}

    out_dir = QUESTIONS_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build simplified format (plain string question lists) — consumed by llm_answering.py
    qs_simple: Dict[str, Any] = {}
    for dim in DIMENSION_NAMES:
        dim_block = all_dim_questions.get(dim, {})
        q_items = dim_block.get("questions", []) if isinstance(dim_block, dict) else []
        q_texts = [
            str(q.get("question_zh", "")).strip()
            for q in q_items
            if isinstance(q, dict) and str(q.get("question_zh", "")).strip()
        ]
        qs_simple[dim] = {
            "dimension": dim,
            "questions": q_texts,
            "search_hints": [],
            "source_proposal_id": proposal_id,
        }

    simple_obj = {
        "proposal_id": proposal_id,
        "generated_at": generated_at_utc,
        "model": OPENAI_MODEL,
        "provider": PROVIDER,
        **qs_simple,
    }

    # ===== 1) 写简化版（供 llm_answering 使用）到 per-pid 目录 =====
    simple_out_path = out_dir / "generated_questions.json"
    simple_out_path.write_text(
        json.dumps(simple_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 问题集合已生成（供 llm_answering 使用）: {simple_out_path}")

    # 与旧工具链对齐：镜像一份到 config/question_sets（run_pipeline / search 等回退路径）
    try:
        CONFIG_QS_DIR.mkdir(parents=True, exist_ok=True)
        mirror_path = CONFIG_QS_DIR / "generated_questions.json"
        mirror_path.write_text(
            simple_out_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(f"[OK] 已同步问题集副本（兼容路径）: {mirror_path}")
    except Exception as e:
        print(f"[WARN] 未能写入 question_sets 副本: {e}")

    # ===== 2) 写详细版（含原始问题对象）到 per-pid 目录 =====
    detail_output_obj = {
        "proposal_id": proposal_id,
        "generated_at": generated_at_utc,
        "model": OPENAI_MODEL,
        "provider": PROVIDER,
        "dimensions": all_dim_questions,
        "question_audit": audit_report,
    }
    detail_out_path = out_dir / "generated_questions_detail.json"
    detail_out_path.write_text(
        json.dumps(detail_output_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 详细问题集合已生成: {detail_out_path}")

    if do_audit and audit_report and not audit_report.get("error"):
        ar_path = out_dir / "question_audit_report.json"
        try:
            ar_path.write_text(
                json.dumps(audit_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[OK] 问题审核报告: {ar_path}")
        except Exception as e:
            print(f"[WARN] 写入 question_audit_report.json 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: 基于 dimensions_v2.json 为五个维度生成差异化问题集（v3.1, 带 links_to + 防幻觉）"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="提案 ID（对应 src/data/extracted/<proposal_id>）",
    )
    parser.add_argument(
        "--min_q_per_dim",
        type=int,
        default=5,
        help="每个维度最少的问题数（推荐下限，默认 5）",
    )
    parser.add_argument(
        "--max_q_per_dim",
        type=int,
        default=10,
        help="每个维度最多的问题数（推荐上限，默认 10）",
    )
    parser.add_argument(
        "--llm_provider",
        required=False,
        help="LLM 提供商（当前仅支持 openai，默认为 .env 中的 PROVIDER）",
    )
    parser.add_argument(
        "--no_question_audit",
        action="store_true",
        help="跳过问题审核（不在 full_text 上做 grounding / 可选 LLM 分类）",
    )
    parser.add_argument(
        "--question_audit_llm",
        action="store_true",
        help="显式启用 LLM 审核（默认已为开；仅在与环境变量冲突时使用）",
    )
    parser.add_argument(
        "--no_question_audit_llm",
        action="store_true",
        help="关闭 LLM 审核（仅保留启发式 grounding；节省 API）",
    )

    args = parser.parse_args()

    global PROVIDER
    if args.llm_provider:
        PROVIDER = args.llm_provider.lower()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = find_latest_extracted_proposal_id()

    llm_override: Optional[bool] = None
    if args.no_question_audit_llm:
        llm_override = False
    elif args.question_audit_llm:
        llm_override = True

    run_generate_questions(
        proposal_id=pid,
        min_q_per_dim=args.min_q_per_dim,
        max_q_per_dim=args.max_q_per_dim,
        enable_question_audit=not args.no_question_audit,
        question_audit_use_llm=llm_override,
    )


if __name__ == "__main__":
    main()
