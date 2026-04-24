# -*- coding: utf-8 -*-
"""
Stage 2 · 维度构建器（build_dimensions_from_facts.py）
----------------------------------------------------
输入：
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  （Stage 1 输出）

输出：
  - src/data/extracted/<proposal_id>/dimensions_v2.json      （新的五维度文件）
  - src/data/extracted/<proposal_id>/dimension_facts.json    （按维度分组的 facts，方便后续调试与复用）
  - src/data/parsed/<proposal_id>/parsed_dimensions.clean.llm.json  （按提案隔离；并行多项目时勿混用）

核心职责：
  - 按 dimensions 标签把 facts 分桶到 team/objectives/strategy/innovation/feasibility
  - 对每个维度单独调用一次 LLM，只基于该维度的事实生成：
      summary / key_points / risks / mitigations
  - 严禁凭空造事实，所有内容必须能在事实列表中找到“影子”
  - 尽量覆盖该维度下出现过的不同 type（team_member/pipeline/market/risk/...）
  - 提示词面向各类商业项目评审：审批友好书面语；strategy/feasibility 强调收入模式、现金流、合规与数据安全等；
    risk 关键词含常见委婉表述（挑战、依赖、尚待验证等）。
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 假设本文件路径：<project_root>/src/tools/build_dimensions_from_facts.py
BASE_DIR = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"
PARSED_DIR = BASE_DIR / "src" / "data" / "parsed"   # 与 llm_answering 对齐
PROGRESS_FILE = BASE_DIR / "src" / "data" / "step_progress.json"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass


def _get_domain_config():
    """Load dimension config from centralized config system."""
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
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

client = None  # lazy-initialized


def _get_client():
    global client
    if client is None:
        client = OpenAI()
    return client

# 注意：这里用 {dimension_name} 标记占位，其它所有 { } 都是字面量 JSON 示例
# 后面用 .replace("{dimension_name}", xxx) 而不是 .format()
DIMENSION_PROMPT_TEMPLATE = """
你是一个严谨的项目评审专家，面向以下【材料语域】的审批场景，
现在要基于【已经抽取好的事实列表】为某一个维度生成结构化摘要。

【材料语域（与 src/config 共用；与 extract_facts_by_chunk 一致）】
__MATERIAL_DOMAIN_ZH_PLACEHOLDER__

⚠️ 最重要的硬性约束（请逐条遵守）：
1）你只能使用我给你的事实列表中的信息，不得凭空添加任何新的机构、人名、疾病、药物、技术、模型、
   管线名称、市场数字、患者数量、CAGR 或其他数字。
2）不允许使用“根据常识”“业内通常认为”“一般来说”“通常情况”“可能/likely/probably”这类
   推测性表达，除非这些话本身就是原始事实的一部分。
3）允许对事实做“合并和压缩”，但：
   - 每一个 key_point、每一条 risk、每一条 mitigation，都应该在某条 fact 文本中能找到对应影子；
   - 不要凭空发明新的风险或对策。
4）你只能总结【当前维度】的内容，可以顺带提及与该维度强相关的信息，
   但不要跑题到其他维度上去写泛泛而谈的评价。

【输出语体（审批友好）】
- summary 与 key_points 一律使用**客观、克制的中文书面语**，便于材料复核与审批阅读。
- **禁止**使用夸张或营销腔措辞（例如「必将」「颠覆」「唯一」「绝对领先」「史无前例」等），
  **除非**这些词在事实原文中已出现且你在忠实概括。
- 优先使用「材料显示」「事实表明」「提案披露」类中性表述，避免空泛吹捧或结论先行。

【当前维度】：{dimension_name}

【该维度的代表性内容示例】（只是提醒你关注点，不是让你照抄；面向各类商业与投资项目材料，而非学术论文或宣传通稿）：
- team:
  - 核心团队成员、机构、职称、研究方向、项目角色
  - 组织结构、国内外团队协同、分工比例
  - 与其他机构的合作团队构成
- objectives:
  - 总体目标（3–5 年想达到什么）
  - 分阶段里程碑（0–6/6–12/12–18/18–36 个月等）
  - 各条产品线/业务线或子项目的目标（若事实为医疗管线则据实概括）
- strategy:
  - 技术路线（含 AI/算法/数字化交付路径）与产品化节奏
  - **收入模式**（订阅/交易抽成/项目制等）、**定价与渠道**、**市场进入与 GTM**
  - **单店/单客户经济模型**、**获客成本与回款**、合作与供应链策略
  - **合规与数据安全**、行业监管与上市路径中与**经营策略**相关的事实
- innovation:
  - 技术/产品/模式的创新点
  - 相比现有方案的优势
  - 关键专利、独特数据/资源、验证证据
- feasibility:
  - 资源基础（实验室、平台、合作方、产能与交付）
  - **现金流**、**预算与融资用途**、费用与毛利结构（若事实中有）
  - **合规、数据安全与内控**、关键人依赖与供应链韧性
  - 风险矩阵及应对措施（技术、市场、资金、法规等）
  - 时间表、实施路径、资源约束等可行性因素

【输入给你的 JSON 结构】（payload）：

payload = {
  "all_facts": [
    {
      "text": "自然语言的一条事实",
      "dimensions": ["team", "strategy"],
      "type": "team_member" 或 "pipeline" 等,
      "meta": { ... }
    },
    ...
  ],
  "risk_facts": [
    // 仅包含 type = "risk" 或 文本中明显包含风险/挑战关键词 的事实
  ],
  "mitigation_facts": [
    // 仅包含 type = "mitigation" 或 文本中明显包含应对/缓解/解决方案关键词 的事实
  ]
}

你需要做两件事：

1) 在心里把 all_facts 归类、去重、合并，找出对【当前维度】最重要、最能代表该维度情况的点。
   - 尽量覆盖当前维度下出现过的不同 type，而不是只盯着一种类型。
2) 按下面格式输出一个 JSON 对象：

{
  "summary": "string，2-4 句中文，审批友好书面语，总结该维度整体情况（须基于 facts 压缩，不空评）",
  "key_points": [
    "string，要点 1（具体、克制，能在至少一条 fact 中找到影子）",
    "string，要点 2",
    "... 在事实足够多时，尽量输出 6-10 条互相不重复的要点。"
  ],
  "risks": [
    "string，基于 risk_facts 中的风险信息总结；",
    "如果 risk_facts 为空，可以写：提案中关于该维度的风险信息较少/未详细说明。",
    "不要凭空发明项目没有写明的风险。",
    "..."
  ],
  "mitigations": [
    "string，对应某个风险的应对措施，必须基于 mitigation_facts 或 all_facts 中出现过的应对策略/合作机制；",
    "如果 mitigation_facts 为空且 all_facts 中也没有明显的应对措施，可以写：提案未具体说明该维度风险的应对措施。",
    "同样不要凭空发明解决方案。",
    "..."
  ]
}

【关于 key_points 数量的硬性要求】
- 如果 all_facts 的数量 ≥ 20 条，请尽量输出 6–10 条互相不重复的 key_points；
- 如果 all_facts 的数量在 8–19 条之间，可以输出 5–8 条 key_points；
- 如果 all_facts 很少（≤ 7 条），则按实际情况输出 3–5 条即可。
- 无论哪种情况，key_points 之间尽量覆盖不同类型的信息，而不是重复同一类事实。

【覆盖不同信息类型的要求】
- 请尽量覆盖当前维度下出现过的不同 type 类型，而不是重复同一类信息。
  例如：
  - team: 如果既有团队成员信息（team_member），又有组织结构（org_structure）、协作模式（collaboration），请三类都各写至少 1 条；
  - objectives: 如果既有总体目标（用 pipeline/product/milestone 描述），又有阶段里程碑（milestone），请都覆盖；
  - strategy: 若事实中同时出现技术路线（tech_route）、市场/客户（market）、**收入与渠道/GTM**、
    合作与资金（collaboration/funding_source）、**合规与数据安全**（regulatory）等，请都尽量覆盖，避免写成纯论文式技术综述；
  - innovation: 请优先包括创新点（ip_asset/ai_model/tech_route）以及已有证据（evidence）；
  - feasibility: 请尽量包括资源/能力（resource）、**现金流与预算/融资**（budget_item/funding_source）、
    **合规与数据安全**、风险与应对（risk/mitigation），以及时间表/实施难度等（以事实为准）。

【关于风险与应对的约束】
- 在生成 "risks" 时，请优先基于 risk_facts 中的事实进行归纳；
- 在生成 "mitigations" 时，请优先基于 mitigation_facts，必要时可以参考 all_facts 中明确提到的应对策略；
- 不要凭空假设项目有某种风险或应对，只能总结文本中真实出现过的内容。

补充要求：
- 不要写“根据常识”“业内通常如何如何”这种话，只能说提案文本展示出来的内容。
- 全文语气保持**正式评审材料**语境：像给投资决策、立项复核、采购评标或监管报送用的摘要，而不是学术论文或宣传通稿。
- 不要输出任何 JSON 以外的文字（不要解释、不要加注释）。
"""


def load_raw_facts(proposal_id: str) -> List[Dict[str, Any]]:
    """
    Load facts, preferring verified_facts.jsonl (Stage 1.5 output) over raw_facts.jsonl.
    When verified facts are available, unverified facts are down-ranked by placing
    them after verified and partially_verified ones, giving downstream LLM calls
    higher-quality input within token limits.
    """
    verified_path = EXTRACTED_DIR / proposal_id / "verified_facts.jsonl"
    raw_path = EXTRACTED_DIR / proposal_id / "raw_facts.jsonl"

    if verified_path.exists():
        source_path = verified_path
        print(f"[INFO] Using verified facts from: {verified_path}")
    elif raw_path.exists():
        source_path = raw_path
        print(f"[INFO] Using raw facts from: {raw_path}")
    else:
        raise FileNotFoundError(
            f"Neither verified_facts.jsonl nor raw_facts.jsonl found for proposal: {proposal_id}"
        )

    facts: List[Dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                facts.append(obj)

    # Sort by verification score (highest first) when verification data is present
    has_verification = any(f.get("verification") for f in facts)
    if has_verification:
        status_rank = {"verified": 0, "partially_verified": 1, "unverified": 2}
        facts.sort(
            key=lambda f: (
                status_rank.get(f.get("verification", {}).get("status", "unverified"), 2),
                -(f.get("verification", {}).get("score", 0.0)),
            )
        )
        verified_count = sum(
            1 for f in facts
            if f.get("verification", {}).get("status") == "verified"
        )
        print(f"[INFO] Facts sorted by verification status: {verified_count} verified out of {len(facts)}")

    print(f"[INFO] Loaded {len(facts)} facts from {source_path}")
    return facts


def group_facts_by_dimension(facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按维度标签把 facts 分桶到五个维度。
    一条 fact 可能属于多个维度，会出现在多个桶里。
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {dim: [] for dim in DIMENSION_NAMES}

    for fact in facts:
        dims = fact.get("dimensions", [])
        if not isinstance(dims, list):
            continue
        for dim in dims:
            if dim in grouped:
                grouped[dim].append(fact)

    for dim in DIMENSION_NAMES:
        print(f"[INFO] 维度 {dim} 相关事实数: {len(grouped[dim])}")
    return grouped


def sort_facts_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按维度定义 type 优先级排序，保证更关键信息排在前面。
    这是纯通用规则，不依赖具体提案内容。
    """
    priority_map = {
        "team": [
            # 先看具体成员履历，再看组织结构和协作模式
            "team_member", "org_structure", "collaboration", "resource", "other"
        ],
        "objectives": [
            # 里程碑 + 管线目标仍然是核心，同时补充市场相关目标
            "milestone", "pipeline", "clinical_design", "product", "market", "other"
        ],
        "strategy": [
            # 把 market 放到最前，确保市场分析一定进入 LLM 视野
            "market", "tech_route", "product", "collaboration",
            "funding_source", "regulatory", "other"
        ],
        "innovation": [
            "ip_asset", "evidence", "ai_model", "tech_route", "product", "other"
        ],
        "feasibility": [
            "resource", "budget_item", "funding_source",
            "risk", "mitigation", "regulatory", "other"
        ],
    }
    order = priority_map.get(dimension_name, ["other"])

    def type_rank(t: str) -> int:
        return order.index(t) if t in order else len(order)

    def ver_score(f: Dict[str, Any]) -> float:
        v = f.get("verification") or {}
        if isinstance(v, dict):
            return float(v.get("score") or 0.0)
        return 0.0

    # type 优先；同 type 内用核验分作 tie-break，避免高价值句因启发式 unverified 被排到截断外
    return sorted(
        facts,
        key=lambda f: (type_rank(f.get("type", "other")), -ver_score(f)),
    )


def truncate_facts_for_prompt(facts: List[Dict[str, Any]], max_chars: int = 12000) -> List[Dict[str, Any]]:
    """
    为了防止单次 prompt 爆 context，对 facts 做一个简单的字符长度截断。
    按顺序累加 text，超过 max_chars 就停（meta 仍然保留）。
    """
    kept: List[Dict[str, Any]] = []
    total = 0
    for fact in facts:
        t = fact.get("text", "") or ""
        t_len = len(t)
        if total + t_len > max_chars and kept:
            break
        kept.append(fact)
        total += t_len
    return kept


# ===== 辅助：基于文本再兜底识别 risk / mitigation =====

_RISK_CN = [
    "风险", "挑战", "瓶颈", "不确定性", "不足", "局限", "缺陷", "障碍", "难点",
    "依赖", "依赖度", "集中度", "单一客户", "第一大客户", "客户集中",
    "承压", "波动", "下滑", "放缓", "收紧", "恶化", "回落",
    "短板", "薄弱", "薄弱环节", "隐忧", "顾虑", "担忧",
    "尚待", "有待", "尚不明确", "存在不确定性", "尚需验证", "待验证",
    "同质化", "价格战", "红海", "竞争加剧",
    "敏感", "脆弱", "波动较大", "不及预期",
    "合规风险", "数据安全", "泄露", "处罚", "诉讼",
    "现金流紧张", "垫资", "回款周期", "账期延长", "应收账款",
]
_RISK_EN = ["risk", "risks", "challenge", "challenges", "bottleneck", "bottlenecks",
            "uncertainty", "limitation", "limitations", "weakness", "weaknesses",
            "barrier", "barriers", "issue", "issues", "difficulty", "difficulties"]

_MITIG_CN = ["应对", "缓解", "降低", "减少", "解决", "克服", "应对措施", "改进", "优化", "管控"]
_MITIG_EN = ["mitigation", "mitigate", "mitigating", "address", "addresses", "addressing",
             "solve", "solves", "solving", "overcome", "overcoming",
             "reduce", "reduces", "reducing", "decrease", "decreases", "decreasing",
             "improve", "improves", "improving", "optimize", "optimizing", "optimization"]


def _looks_like_risk(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in _RISK_CN):
        return True
    if any(k in t for k in _RISK_EN):
        return True
    return False

def reclassify_risk_mitigation_global(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    在进入维度构建之前，对所有 facts 做一次全局的 risk/mitigation 纠偏：
    - 如果 type 不是 risk/mitigation，但文本明显是风险/挑战，就改成 risk
    - 如果 type 不是 risk/mitigation，但文本明显是应对/缓解方案，就改成 mitigation
    - 如果本身标成 risk/mitigation 但文本看起来不像风险/应对，则降级为 other
      （必要时在 risk 和 mitigation 之间互换）
    """
    new_facts: List[Dict[str, Any]] = []

    for f in facts:
        t = f.get("type", "other") or "other"
        txt = f.get("text", "") or ""

        # 优先纠错：如果已经标成 risk/mitigation，但文本不符合，就降级/互换
        if t == "risk":
            if not _looks_like_risk(txt):
                # 文本更像应对措施，就改成 mitigation；否则降级成 other
                if _looks_like_mitigation(txt):
                    t = "mitigation"
                else:
                    t = "other"
        elif t == "mitigation":
            if not _looks_like_mitigation(txt):
                # 文本更像风险描述，就改成 risk；否则降级成 other
                if _looks_like_risk(txt):
                    t = "risk"
                else:
                    t = "other"
        else:
            # 如果原始类型既不是 risk 也不是 mitigation，就尝试“升格”
            if _looks_like_risk(txt):
                t = "risk"
            elif _looks_like_mitigation(txt):
                t = "mitigation"

        f["type"] = t
        new_facts.append(f)

    return new_facts

def _looks_like_mitigation(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in _MITIG_CN):
        return True
    if any(k in t for k in _MITIG_EN):
        return True
    return False


def call_llm_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对单一维度调用一次 LLM，生成 summary/key_points/risks/mitigations。
    强约束：只能基于 facts，不得脑补。
    """

    # 1) 先按 type 排序，再截断，确保更重要的信息优先被看到
    sorted_facts = sort_facts_for_dimension(dimension_name, facts)
    all_facts_for_prompt = truncate_facts_for_prompt(sorted_facts, max_chars=12000)

    # 2) 识别风险 / 对策 facts：不仅看 type，还看文本关键词
    risk_facts: List[Dict[str, Any]] = []
    mitigation_facts: List[Dict[str, Any]] = []

    for f in all_facts_for_prompt:
        t = f.get("type", "")
        txt = f.get("text", "") or ""

        # 明确标成 risk 的，或者文本看起来是在描述风险/挑战，都归入
        if t == "risk" or _looks_like_risk(txt):
            risk_facts.append(f)

        # 明确标成 mitigation 的，或者文本看起来是在描述应对/解决方案，也归入
        if t == "mitigation" or _looks_like_mitigation(txt):
            mitigation_facts.append(f)

    payload = {
        "all_facts": all_facts_for_prompt,
        "risk_facts": risk_facts,
        "mitigation_facts": mitigation_facts,
    }
    facts_json_str = json.dumps(payload, ensure_ascii=False, indent=2)

    # 用 replace，而不是 format，避免 JSON 里的 { } 被当成占位符
    try:
        from src.config import material_domain_zh_for_prompts

        md = material_domain_zh_for_prompts()
    except Exception:
        md = ""
    prompt = (
        DIMENSION_PROMPT_TEMPLATE.replace("{dimension_name}", dimension_name).replace(
            "__MATERIAL_DOMAIN_ZH_PLACEHOLDER__", md.strip() or "（材料语域未配置，仅以事实列表为准。）"
        )
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的项目评审助手，只能基于给定的事实列表进行总结，不得编造。"
            "典型材料为各类商业与投融资书面材料（不限行业）：输出须客观、克制、审批友好；除非事实原文已有，避免夸张营销措辞。",
        },
        {
            "role": "user",
            "content": prompt + "\n\n=== payload 开始 ===\n" + facts_json_str,
        },
    ]

    resp = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=2600,
    )
    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] 维度 {dimension_name} JSON 解析失败，原始内容如下：")
        print(raw)
        raise e

    # 兜底：保证四个字段存在
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    key_points = data.get("key_points", [])
    if not isinstance(key_points, list):
        key_points = []
    risks = data.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    mitigations = data.get("mitigations", [])
    if not isinstance(mitigations, list):
        mitigations = []

    # 轻量兜底：facts 足够多但 key_points 太少，打日志提醒（先不强制重试）
    fact_count = len(all_facts_for_prompt)
    if fact_count >= 20 and len(key_points) < 6:
        print(
            f"[WARN] 维度 {dimension_name}: all_facts={fact_count} 但 key_points 只有 {len(key_points)} 条，"
            f"如有需要可以在此处加重试/补点逻辑。"
        )

    return {
        "summary": summary.strip(),
        "key_points": [str(x).strip() for x in key_points if str(x).strip()],
        "risks": [str(x).strip() for x in risks if str(x).strip()],
        "mitigations": [str(x).strip() for x in mitigations if str(x).strip()],
    }


def run_build(proposal_id: str):
    pid = proposal_id
    facts = load_raw_facts(proposal_id)
    # 全局先做一次 risk/mitigation 纠偏
    facts = reclassify_risk_mitigation_global(facts)
    grouped = group_facts_by_dimension(facts)

    # 额外输出一份 dimension_facts.json，方便后续人工检查和调试
    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_facts_path = out_dir / "dimension_facts.json"
    dim_facts_path.write_text(
        json.dumps(grouped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 已写出按维度分组的 facts: {dim_facts_path}")

    dimensions_result: Dict[str, Dict[str, Any]] = {}

    _write_progress(0, len(DIMENSION_NAMES), pid)
    for dim_idx, dim in enumerate(DIMENSION_NAMES):
        dim_facts = grouped.get(dim, [])
        print(f"\n[INFO] 开始构建维度 {dim} ...")

        if not dim_facts:
            # 没有任何事实，给一个空壳（显式指出信息缺失）
            dimensions_result[dim] = {
                "summary": f"提案文本中关于 {dim} 维度的明确信息较少，无法做出详细总结。",
                "key_points": [],
                "risks": [f"提案中关于 {dim} 维度的细节信息较少，可能影响评估。"],
                "mitigations": ["提案未具体说明如何补充或缓解该维度信息不足的问题。"],
            }
            print(f"[INFO] 维度 {dim} 无事实，写入占位结果。")
            _write_progress(dim_idx + 1, len(DIMENSION_NAMES), pid)
            continue

        data = call_llm_for_dimension(dim, dim_facts)

        # ==== 风险覆盖度标记 ====
        risk_count = len(data.get("risks", []) or [])
        if risk_count == 0:
            level = "low"
            reason = "提案文本中几乎没有显式描述该维度相关的风险，系统无法进行充分的风险细化。"
        elif risk_count <= 2:
            level = "medium"
            reason = "该维度仅有少量风险相关描述，风险分析的粒度有限。"
        else:
            level = "high"
            reason = "该维度在提案中有较为丰富的风险相关描述，可以进行较细致的风险分析。"

        data["risk_coverage"] = {
            "level": level,
            "reason": reason,
            "risk_count": risk_count,
        }

        dimensions_result[dim] = data
        _write_progress(dim_idx + 1, len(DIMENSION_NAMES), pid)

        print(
            f"[INFO] 维度 {dim} 完成：summary_len={len(data['summary'])}, "
            f"key_points={len(data['key_points'])}, risks={len(data['risks'])}, "
            f"mitigations={len(data['mitigations'])}, "
            f"risk_coverage={level}"
        )

    # 1) 写入 per-proposal 维度文件
    out_path = out_dir / "dimensions_v2.json"
    out_path.write_text(
        json.dumps(dimensions_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 新版五维度文件已生成: {out_path}")

    # 2) 按 proposal_id 写入 parsed，避免并行评审多个项目时互相覆盖（旧版单文件仍兼容读取，见 search_by_dimension）
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_sub = PARSED_DIR / pid
    parsed_sub.mkdir(parents=True, exist_ok=True)
    parsed_path = parsed_sub / "parsed_dimensions.clean.llm.json"

    parsed_obj = {
        dim: {
            "summary": data.get("summary", ""),
            "key_points": data.get("key_points", []),
            "risks": data.get("risks", []),
            "mitigations": data.get("mitigations", []),
        }
        for dim, data in dimensions_result.items()
    }

    parsed_path.write_text(
        json.dumps(parsed_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 已写出 parsed 维度文件供 llm_answering 使用: {parsed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: 基于 raw_facts.jsonl 构建五个维度的 dimensions_v2.json"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="提案 ID（对应 src/data/extracted/<proposal_id>）",
    )
    args = parser.parse_args()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        # 默认用 extracted 里最新的一个子目录
        if not EXTRACTED_DIR.exists():
            raise FileNotFoundError(f"未找到 extracted 目录: {EXTRACTED_DIR}")
        candidates = [
            (d.stat().st_mtime, d.name)
            for d in EXTRACTED_DIR.iterdir()
            if d.is_dir()
        ]
        if not candidates:
            raise FileNotFoundError(f"extracted 目录下没有任何子目录: {EXTRACTED_DIR}")
        pid = max(candidates, key=lambda x: x[0])[1]
        print(f"[INFO] [auto] 选中最新提案 ID: {pid}")

    run_build(pid)


if __name__ == "__main__":
    main()
