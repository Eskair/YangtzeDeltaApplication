# -*- coding: utf-8 -*-
"""
Stage 3 · 维度定制化问题生成器（generate_questions.py · v3.1）

输入：
  - src/data/extracted/<proposal_id>/dimensions_v2.json

输出（两个）：
  1) 详细版（保留 qid/aspect/answer_type/links_to 等全量信息，方便调试与后续扩展）：
     - src/data/questions/<proposal_id>/generated_questions.json

  2) 简化版（专门给 llm_answering 使用，只保留按维度的问题字符串列表）：
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
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
        "团队", "核心成员", "项目负责人", "负责人", "pi",
        "履历", "背景", "经历", "经验",
        "临床经验", "法规经验", "产业化", "转化", "商业化",
        "带队", "主导", "项目记录", "成功案例",
    ]
    kw_en = [
        "team", "core team", "core member", "leader", "leadership",
        "principal investigator", "pi",
        "background", "track record", "experience", "experiences",
        "clinical", "regulatory", "commercialization", "industrial",
    ]

    return any(k in q_zh for k in kw_zh) or any(k in q_en for k in kw_en)


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
    Falls back to hardcoded defaults if the config system is unavailable.
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

    # Hardcoded fallback (preserves original behavior)
    return {
        "team": {
            "min_q": 6, "max_q": 9,
            "focus_zh": "关注团队组成、核心负责人履历、跨机构合作网络、以及团队稳定性和时间投入。",
            "aspects": [
                {"id": "leadership_experience", "desc_zh": "项目负责人的领导经验与往期重大项目执行记录"},
                {"id": "domain_expertise", "desc_zh": "团队在目标领域的专业深度"},
                {"id": "collaboration_network", "desc_zh": "合作机构、产业伙伴网络及互补性"},
                {"id": "governance_and_decision_making", "desc_zh": "项目治理结构、决策机制、质量控制"},
                {"id": "team_capacity_and_bandwidth", "desc_zh": "团队当前人力负荷与资源投入能力"},
            ],
        },
        "objectives": {
            "min_q": 6, "max_q": 9,
            "focus_zh": "关注项目总体目标的清晰度、分阶段里程碑、可量化指标和可实现性。",
            "aspects": [
                {"id": "overall_goal_clarity", "desc_zh": "总体目标是否明确、聚焦"},
                {"id": "milestones_and_timeline", "desc_zh": "各阶段里程碑的设计是否合理可执行"},
                {"id": "outcome_and_success_metrics", "desc_zh": "是否有清晰可量化的成功指标"},
                {"id": "scope_and_prioritization", "desc_zh": "项目范围和优先级排序"},
                {"id": "realism_and_ambition_balance", "desc_zh": "目标在雄心和可行性之间的平衡"},
            ],
        },
        "strategy": {
            "min_q": 7, "max_q": 10,
            "focus_zh": "关注技术路线设计、市场与商业化路径、合作伙伴策略和资源利用方式。",
            "aspects": [
                {"id": "technical_strategy", "desc_zh": "技术路线的合理性与替代方案"},
                {"id": "commercialization_and_market_entry", "desc_zh": "商业化模式、定价与市场进入路径"},
                {"id": "partnership_and_business_model", "desc_zh": "合作模式（授权、共同开发、服务等）"},
                {"id": "data_and_evidence_strategy", "desc_zh": "数据利用策略及隐私合规安排"},
                {"id": "scaling_and_globalization", "desc_zh": "从验证到大规模推广的扩展策略"},
            ],
        },
        "innovation": {
            "min_q": 6, "max_q": 9,
            "focus_zh": "关注技术/产品相对现有方案的创新性、差异化优势、知识产权布局和证据支撑。",
            "aspects": [
                {"id": "novelty_vs_state_of_art", "desc_zh": "相对当前前沿方案的真正创新点"},
                {"id": "differentiation_and_competitive_edge", "desc_zh": "与替代方案相比的明确优势"},
                {"id": "ip_and_protection", "desc_zh": "专利/数据资产的保护布局"},
                {"id": "evidence_strength_for_innovation", "desc_zh": "创新点的实验/验证证据强度"},
                {"id": "platform_and_extensibility", "desc_zh": "是否构成可拓展的平台或仅单点创新"},
                {"id": "risk_of_obsolescence", "desc_zh": "技术在3-5年内被替代的风险评估"},
            ],
        },
        "feasibility": {
            "min_q": 7, "max_q": 10,
            "focus_zh": "关注资源与基础设施、资金与预算、实施路径、关键风险和应对措施、落地可行性。",
            "aspects": [
                {"id": "resources_and_infrastructure", "desc_zh": "资源平台是否充足且可长期稳定使用"},
                {"id": "funding_and_budget_planning", "desc_zh": "资金来源多样性、预算分配合理性"},
                {"id": "operational_execution_plan", "desc_zh": "实施路径是否具体清晰"},
                {"id": "risk_management", "desc_zh": "对各类风险的识别与缓解措施"},
                {"id": "implementation_barriers", "desc_zh": "在实际场景中落地的阻力"},
                {"id": "timeline_and_resource_alignment", "desc_zh": "时间表与资源投入是否匹配"},
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
    "desc_zh": "项目负责人 / 核心 PI 的领导经验与往期重大项目执行记录"
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
   - 严禁出现 payload 中完全没有出现过的具体公司、机构、大学、医院、平台、药物、基金、国家或城市名称；
   - 如果确实需要提到合作方或机构、医院等，但 payload 中没有给出具体名字，只能使用
     “某国际制药公司”“某合作医院”“某科研机构”“某平台型公司”等泛指表达，不能自己编造新的名称；
   - 如果 payload 中已经出现了某个实体名称（例如某家公司的正式名称），你可以在问题中以【完全相同的写法】引用它，
     但不得新增其它实体名称，也不得为人物杜撰新的外文姓名。
3. 如果需要讨论“目标客户类型”“市场规模”“疗效提升幅度”等，但 payload 没给精确数字或具体对象：
   - 问题可以要求后续回答者“根据提案中已有信息进行定性分析或区间估计”，
   - 并在问题中明确加入类似措辞：
     “如提案未给出具体数值/名单，请在回答时先说明信息缺失，再分析其可能影响。”

【维度特定要求（team / objectives / strategy）】
- 如果当前维度是 team：
  - 至少设计 2 个高优先级（priority=1 或 2）的“简历驱动”问题，显式围绕提案中对核心成员 / PI / 项目负责人
    的履历、既往项目经验、临床 / 产业化推进记录等 key_points 发问。
  - 这类问题应该要求后续回答者基于提案中已有的团队背景信息，综合判断团队推进本项目到临床应用和商业落地的能力。
- 如果当前维度是 strategy：
  - 如果 payload 的 key_points 中出现“市场 / market / CAGR / 竞争 / 客户 / 销售”等相关内容，
    至少设计 1 个高优先级的“市场驱动”问题，用于评估项目在目标市场中的定位、竞争压力和市场进入 / 商业化策略。
  - 该问题的 links_to.key_points 中，至少要包含一个与市场分析相关的条目。
- 如果当前维度是 objectives：
  - 如果 key_points 中提到了疾病负担、目标患者群体、目标市场机会等内容，
    至少设计 1 个问题，把“项目目标与未满足临床需求 / 市场机会的匹配度”作为核心评估点，
    并允许在信息不足时先指出提案中的信息缺口。

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
) -> List[Dict[str, Any]]:
    """
    调用 LLM，为某个维度生成【定制化】问题列表（带 links_to）。
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

    user_content = (
        prompt
        + "\n\n=== 该维度的 aspects 配置 ===\n"
        + aspects_str
        + "\n\n=== 当前维度内容概览（统计信息）===\n"
        + overview_str
        + "\n\n=== 当前维度的摘要 payload ===\n"
        + payload_str
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
                extra_q_team = {
                    "aspect": "leadership_experience",  # 在 DIMENSION_CONFIG['team']['aspects'] 里已经存在
                    "question_zh": (
                        "基于提案中对核心成员和项目负责人的教育背景、临床/产业化经验及既往重大项目记录的描述，"
                        "您如何评价该团队在将本项目推进至临床应用和商业落地方面的整体能力？"
                        "如果提案对关键履历或可验证业绩描述不够具体，请在回答中先指出这一信息缺失，"
                        "并讨论其对评估结果的影响。"
                    ),
                    "question_en": (
                        "Based on the proposal's description of the core team members and project leaders "
                        "(education, clinical/industrial experience, and track record in previous major projects), "
                        "how would you assess the team's overall ability to drive this project towards clinical "
                        "application and commercialization? If the proposal does not provide sufficiently specific "
                        "or verifiable track record information, please first state this information gap and then "
                        "discuss its impact on your assessment."
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
                    aspect_id = "unmet_need_alignment"

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

def run_generate_questions(
    proposal_id: str,
    min_q_per_dim: int = 5,
    max_q_per_dim: int = 10,
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

    # ===== 2) 写详细版（含原始问题对象）到 per-pid 目录 =====
    detail_output_obj = {
        "proposal_id": proposal_id,
        "generated_at": generated_at_utc,
        "model": OPENAI_MODEL,
        "provider": PROVIDER,
        "dimensions": all_dim_questions,
    }
    detail_out_path = out_dir / "generated_questions_detail.json"
    detail_out_path.write_text(
        json.dumps(detail_output_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 详细问题集合已生成: {detail_out_path}")


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

    args = parser.parse_args()

    global PROVIDER
    if args.llm_provider:
        PROVIDER = args.llm_provider.lower()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = find_latest_extracted_proposal_id()

    run_generate_questions(
        proposal_id=pid,
        min_q_per_dim=args.min_q_per_dim,
        max_q_per_dim=args.max_q_per_dim,
    )


if __name__ == "__main__":
    main()
