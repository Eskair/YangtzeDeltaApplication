# -*- coding: utf-8 -*-
"""
Stage 1 · 块级事实抽取器（extract_facts_by_chunk.py）
----------------------------------------------------
输入：
  - src/data/prepared/<proposal_id>/full_text.txt  （由 prepare_proposal_text.py 生成）

输出：
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  每行一个 JSON fact

职责：
  - 将长文按字符切块（带 overlap）
  - 对每个块，用 LLM 抽取“原子事实列表”
  - 每条事实附带：dimensions[], type, meta(chunk_index, char_range 由代码填充)
  - 后续 Stage 2 再用这些 facts 去构建五个维度的最终 dimensions_v2.json

本版优化要点：
  1）减小 chunk 大小、加大 overlap，提高单块信息覆盖率，避免一个块塞太多信息导致抽不全。
  2）加强 Prompt 中对“五维覆盖”的要求，显式提醒模型不要只抽单一维度的信息。
  3）更激进的 type→dimensions 映射，让跨维度事实被多个维度同时看到，避免某维度信息过于稀薄。
  4）扩充关键词推断逻辑 _infer_dims_from_text，补充 objectives / innovation / feasibility 等隐性表述。
  5）在运行结束时输出五个维度的事实数量分布，并对明显偏少的维度给出警告，便于调参与排错。
  6）新增：对“文本很长但抽取事实过少”的 chunk 自动再跑一轮 dense 模式，强化召回。
  7）按 dense / JSON 重试轮次动态提高输出 token 上限，降低截断导致 JSON 非法的概率。
  8）跨 chunk 按规范化文本去重，减轻 overlap 带来的重复事实写入 raw_facts.jsonl。
  9）主提示词由 build_fact_prompt() 按当前领域配置的 dimensions/type 动态拼装，减少枚举漂移。
 10）主提示词通过 material_domain_zh 提供可选「材料形态」说明（投融资、市场、合规等常见片段），
     帮助模型识别各类商业书面材料；具体行业以原文为准，YAML 可覆盖。
 11）材料语域长段说明来自 src.config.material_domain_zh（default.yaml），与 build_dimensions 共用。
"""

import os
import json
import argparse
import inspect
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Mapping

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ========= 路径配置 =========

BASE_DIR = Path(__file__).resolve().parents[2]
PREPARED_DIR = BASE_DIR / "src" / "data" / "prepared"
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"
PROGRESS_FILE = BASE_DIR / "src" / "data" / "step_progress.json"


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

client = None  # lazy-initialized to avoid import-time side effects


def _get_client():
    global client
    if client is None:
        client = OpenAI()
    return client


def _get_domain_config():
    """Load dimension/type config from the centralized config system."""
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from src.config import get_config
        return get_config()
    except Exception:
        return None


def _get_valid_dimensions():
    cfg = _get_domain_config()
    if cfg:
        return cfg.valid_dimensions
    return ["team", "objectives", "strategy", "innovation", "feasibility"]


def _get_valid_types():
    cfg = _get_domain_config()
    if cfg:
        return cfg.valid_types
    return [
        "team_member", "org_structure", "collaboration", "resource",
        "pipeline", "milestone", "market", "tech_route", "product",
        "ip_asset", "evidence", "budget_item", "funding_source",
        "risk", "mitigation", "ai_model", "clinical_design",
        "regulatory", "other",
    ]


def _get_type_to_dims_map():
    cfg = _get_domain_config()
    if cfg:
        return cfg.get_type_to_dims_map()
    return {}


# Keep module-level references for backward compatibility
VALID_DIMENSIONS = _get_valid_dimensions()
VALID_TYPES = _get_valid_types()

# type 字段说明：与 _get_valid_types() 默认集对齐；未知 type 由 build_fact_prompt 泛化一行
_TYPE_FIELD_HELP: Mapping[str, str] = {
    "team_member": "成员身份、任职机构（如××有限公司/股份有限公司/（有限合伙））、职责、行业经验与项目角色",
    "org_structure": "团队/公司股权与治理结构、部门分工、核心岗位占比",
    "collaboration": "战略客户、渠道伙伴、供应商/代工厂、校企或产业联盟、联合推广等协同模式",
    "resource": "产线、仓网、数据平台、牌照资质、关键设备或合作资源等",
    "pipeline": "产品线/业务线、SKU 或服务模块的目标、交付节奏；若原文为医疗语境可含适应症/管线（以原文为准）",
    "milestone": "版本发布、门店/产线爬坡、融资交割、上市节点、关键合同签署等含时间的里程碑",
    "market": "赛道规模、增速/CAGR、市占率、竞品对比、客单价、区域/渠道结构、估值与收入预测等商业量化信息",
    "tech_route": "产品研发路线、架构选型、供应链数字化、SaaS 多租户方案等技术与交付路径",
    "product": "产品形态（软硬件/SaaS/平台）、SKU、交付方式、订阅与定价、客户场景",
    "ip_asset": "专利/软著/商标、核心算法与数据资产、排他协议或关键 know-how",
    "evidence": "试点/POC、客户案例与标杆订单、复购与留存数据；或研发侧测试与验证结论（以原文为准）",
    "budget_item": "研发/市场/人力/资本开支等分阶段预算、毛利/费用结构、单店或单客户经济模型中的数字",
    "funding_source": "自有资金、天使/A/B 轮、战略投资、领投/跟投、政府或产业基金、可转债等融资安排",
    "risk": "市场、竞争、供应链、现金流、合规与数据安全、关键人依赖等风险或不确定性",
    "mitigation": "对上述风险的具体应对、预案、合规与内控措施",
    "ai_model": "所采用的模型/算法/架构或智能化能力（若原文提及）",
    "clinical_design": "若原文为医疗/临床：试验阶段、入组与终点；若为商业项目：灰度发布、抽样审计、效果评估方案等验证设计——仅当原文出现该结构时抽取",
    "regulatory": "上市合规、行业准入、数据安全与隐私、广告与营销合规、跨境经营许可等（区别于纯市场规模数字）",
    "other": "无法归类但又与项目有关的事实",
}


def _format_type_catalogue(types: List[str]) -> str:
    lines = []
    for t in types:
        desc = _TYPE_FIELD_HELP.get(t)
        if desc:
            lines.append(f'- "{t}": {desc}')
        else:
            lines.append(
                f'- "{t}": 与项目内容相关（此 type 来自当前领域配置）；结合文本从允许列表中选择。'
            )
    return "\n".join(lines)


def _format_dimension_help(dimensions: List[str]) -> str:
    """Short human-readable line per default five-dim set; unknown dims get generic line."""
    known = {
        "team": "创始人/高管/核心骨干、组织与股权、激励与关键人、客户成功或交付团队等",
        "objectives": "营收与毛利目标、市场份额/门店数/GMV、融资里程碑、产品路线图上的阶段性目标等",
        "strategy": "GTM 与渠道、定价与回款、供应链与成本、竞争策略、合作伙伴与生态、监管与上市路径中的经营策略等",
        "innovation": "技术/产品/商业模式差异化、专利与数据壁垒、与竞品对比的明确优势等",
        "feasibility": "现金流与融资用途、产能与交付、合规与数据安全、实施计划与风险应对等",
    }
    lines = []
    for d in dimensions:
        if d in known:
            lines.append(f'- "{d}": {known[d]}')
        else:
            lines.append(f'- "{d}": 与评估维度「{d}」相关的事实（来自当前领域配置）。')
    return "\n".join(lines)


def build_fact_prompt() -> str:
    """
    组装 Stage1 主提示词：dimensions/type 与 src.config 当前返回值一致，避免枚举漂移。
    """
    dimensions = _get_valid_dimensions()
    types = _get_valid_types()
    dims_join = ", ".join(f'"{d}"' for d in dimensions)
    type_catalogue = _format_type_catalogue(types)
    dim_help = _format_dimension_help(dimensions)

    body = f"""
你是一个“事实抽取器”，负责从一小段提案文本中【逐条抽取原子事实】。

【材料语域（与 src/config 共用；修改请编辑 src/config/*.yaml 的 material_domain_zh）】
__MATERIAL_DOMAIN_ZH_PLACEHOLDER__

⚠️ 非常重要的硬性约束（请逐条遵守）：
1）你只处理当前这一个文本块，不要猜测其他页面或上下文的内容。
2）不允许编造任何文本中没有出现的：机构、人物、公司、客户、供应商、投资人、国家、城市、
   技术/产品/模型名称、商标、项目名称、市场规模、估值、融资额、收入/毛利、市占率、CAGR 或其他具体数字。
   金额、百分比、量纲尽量沿用原文写法（阿拉伯数字、中文「万/亿/千万」、千分位、全角/半角等），便于后续自动核验。
   若原文为医疗/医药语境且出现疾病名、药物名、适应症等，仅可忠实摘录，不得补充未出现的医学结论。
3）禁止使用“可能 / likely / 一般认为 / 通常 / 预计 / 大多 / 被认为 / typically / usually / generally /
   probably / potentially / it is believed that”等带有推测性的词语，
   除非这些词本身已经出现在原文中并且你是在忠实复述原文。
4）每条 fact 必须能在原文中找到对应内容，可以轻微改写，但必须保留原文中的关键短语
   （例如公司全称带「有限公司」「股份有限公司」「（有限合伙）」、领投/跟投与轮次、产品名、竞品名、关键数字等）。
5）每条 fact 尽量控制在 1–2 句内，不要写成长段落；尽量保持具体、可验证。

【目标领域】
- 上文「材料语域」仅帮助识别常见商业信息形态；**若与当前文本类型不符，忽略其暗示，只按原文抽取**，不要强行套用商业叙事。
- 你不能凭空假设未出现的行业或数据，只能按文本本身来。

【常见商业项目表述形态（仅原文出现时才抽取，勿编造）】
- **融资与资本**：轮次（天使/A/B…）、领投/跟投、投前投后估值、募资用途、可转债/SAFE 等 → 多用 type "funding_source" / "budget_item"，并与 dimensions 中 objectives/feasibility/strategy 等联动。
- **经营与财务**：收入、毛利/毛利率、费用、现金流、单店模型、订单与 backlog、复购等 → "market" / "budget_item" / "evidence" 等按语义选择。
- **市场与竞争**：市占、竞品对比、价格带、渠道结构、区域扩张、客户案例与标杆客户 → "market" / "collaboration" / "evidence"。
- **产品与交付**：产品线、SKU、SaaS 版本迭代、供应链与履约、交付周期 → "product" / "pipeline" / "tech_route"。
- **合规与资产**：上市与监管路径、数据安全与隐私、知识产权与软著商标 → "regulatory" / "ip_asset"。

【什么是“原子事实”？】
- 一条 fact 应该尽量只表达“一个相对独立、可以单独复述的事实”；
- 不要把很多主题揉成一条长句；
- 若同句同时出现「团队配置」与「下一轮融资里程碑」，或「供应链方案」与「目标市占」，请拆成多条 facts。
- 若一句里包含多个独立要点（例如「团队背景 + SaaS 多租户路线 + 华东渠道目标」），也应拆成多条 facts。

【高优先级内容（必须尽量完整抽取）】
- 若文本出现**核心团队 / 创始人 / 高管 / 业务负责人**等：为每位关键人拆分多条事实，包括：
  - 职务与任职主体（尽量保留「××有限公司」「股份有限公司」「（有限合伙）」等法定名称片段）；
  - 与当前项目相关的经历（如曾负责某品类从 0 到 1、主导关键客户签约、搭建供应链体系等，须原文有据）。
- 若文本出现**融资与市场**内容：对轮次、领投/跟投、估值、募资用途、市占与竞品、收入与毛利预测、渠道与客户案例等，
  **每条重要数字或结论单独一条 fact**；市场与竞争量化多用 type="market"，资金安排多用 "funding_source" / "budget_item"。
- 若文本出现**市场分析**段落（规模、增速、区域、竞品、定价与支付方等）：拆条抽取并优先 type="market"，勿用一句笼统概括整段。

【抽取数量与粒度要求】（全篇统一：任何模式、任何重试下 facts 总数硬上限为 25 条）
- 请尽量把长句拆成多条事实，保证每条 fact 聚焦一个主题。
- 当本块**明显包含多段、多主题**信息时，在**不超过 25 条**的前提下尽量多抽（例如约 15–25 条）；**不要为凑条数编造**。
- 当本块**只围绕一个主题**或信息较少时，**少而准优于硬凑**；可以少于 10 条甚至返回 {{"facts": []}}，但不要漏掉清晰事实。
- **任何情况下不得超过 25 条 facts**。

【维度标签（dimensions 的含义）】
{dim_help}

⚠️ 维度标注的硬性要求：
- 只要某条 fact 明显与上述任一维度有关，就必须把对应维度写进 dimensions。
- 一条 fact 可以同时属于多个维度（例如 ["team","strategy"]）。
- 每条 fact 的 dimensions 至少要包含 1 个标签；如果你真的无法判断，请使用 ["feasibility"] 兜底。
- 「宁可多标维度」是指：针对不同**信息点**分别标注；**不要在同一件事的表述上重复拆成多条几乎相同的 facts**。

【type 可选值】（以下为当前任务允许的 type 字符串，**必须与之一完全一致**）
{type_catalogue}

【标注提示（通用规则）】
- 市场规模、增速/CAGR、市占、竞品对比、销售额/订单、渠道结构、客户画像、支付与账期（非单纯合规流程）→ "market"
- 上市与行业准入、数据安全与隐私、广告与营销合规、跨境许可等 → "regulatory"
- 融资轮次、领投/跟投、战略投资、政府/产业基金、募资用途与资金到账安排 → 优先 "funding_source"；具体金额拆分也可 "budget_item"
- 费用预算、CAPEX/OPEX、分阶段投入与毛利结构中的数字 → "budget_item"
- 风险、不确定性、瓶颈、挑战 → "risk"；明确应对与预案 → "mitigation"
- 客户试点、标杆案例、POC/试点数据、订单与复购描述 → 常用 "evidence" 或 "market"（择更贴近语义者）
- 产品形态、SKU、订阅与交付 → "product"；研发与架构路线 → "tech_route"
- 具体 AI/算法/模型（若原文点名）→ "ai_model"
- 专利/软著/商标、核心数据资产 → "ip_asset"
- 若不确定 type，可用 "other"，但请优先选最接近的类型。

【维度覆盖要求（非常重要）】
- 如果该文本块中同时出现了团队信息、项目目标、技术/商业策略、创新亮点、可行性/风险等多种内容，
  请尽量保证抽取出来的 facts 在这些维度上都有覆盖。
- 不要把所有 fact 都集中在单一维度（例如只抽团队信息），而忽略同一文本块中出现的目标、
  策略、创新或可行性的信息。
- 当你需要“取舍”时，优先保留能代表不同维度、不同主题的事实，而不是在同一个小点上反复细化。

【输出格式要求】
- 只输出**一个** JSON 对象；**不要**使用 markdown 代码围栏（不要使用 ``` 包裹）。
- 顶层**只能**包含键 "facts"，**不要**添加其他任何顶层键。
- "facts" 为数组；数组元素仅含字段 "text"（字符串）、"dimensions"（字符串数组）、"type"（字符串）。
- dimensions 中每一项必须从以下集合选取（可多个）：{dims_join}
- type 必须从本节【type 可选值】列表中的字符串**原样**选取其一。
- JSON 必须可被标准解析器解析：字符串内的引号、换行等必须正确转义；**禁止**在 JSON 中使用 // 或 /* */ 注释。
- 不要输出 meta 字段（由系统自动补充）。
- 若无有用事实，返回 {{"facts": []}}。
- 除该 JSON 外不要输出任何解释性自然语言。

【与分块重叠的说明】
- 相邻文本块可能因重叠而包含相同原文片段；对同一条信息请尽量保持与原文**关键短语一致**的写法，避免同义反复拆成多条几乎相同的事实（允许忠实前提下轻微压缩表述）。

现在开始处理我给你的文本块。
"""
    try:
        from src.config import material_domain_zh_for_prompts

        md = material_domain_zh_for_prompts()
    except Exception:
        md = ""
    if not md.strip():
        md = (
            "各类商业项目评审材料（投融资计划书、可行性研究、招标与响应文件、公司介绍、备忘录等），行业不限。\n"
            "中文或以中文为主、中英混排较为常见；若为纯外文、临床方案、科研课题等，仅以原文为准，勿套用未出现的商业要素。"
        )
    return body.replace("__MATERIAL_DOMAIN_ZH_PLACEHOLDER__", md.strip())


def find_latest_prepared_proposal() -> str:
    if not PREPARED_DIR.exists():
        raise FileNotFoundError(f"未找到 prepared 目录: {PREPARED_DIR}")

    candidates = []
    for d in PREPARED_DIR.iterdir():
        if d.is_dir():
            candidates.append((d.stat().st_mtime, d.name))

    if not candidates:
        raise FileNotFoundError(f"prepared 目录下没有任何提案子目录: {PREPARED_DIR}")

    proposal_id = max(candidates, key=lambda x: x[0])[1]
    print(f"[INFO] [auto] 选中最新提案 ID: {proposal_id}")
    return proposal_id


def load_full_text(proposal_id: str) -> str:
    path = PREPARED_DIR / proposal_id / "full_text.txt"
    if not path.exists():
        raise FileNotFoundError(f"full_text.txt 不存在: {path}")
    text = path.read_text(encoding="utf-8")
    print(f"[INFO] 读取 full_text: {path} (长度 {len(text)} 字符)")
    return text


def make_chunks(text: str, max_chars: int = 1800, overlap: int = 400) -> List[Dict[str, Any]]:
    """
    简单按字符切块，带 overlap；不做句子级别切分。
    默认 max_chars=1800, overlap=400，比之前更细、更密，有利于提高抽取覆盖率。
    返回列表，每个元素包含: chunk_text, start, end, index
    """
    chunks = []
    n = len(text)
    if n == 0:
        return chunks

    idx = 0
    chunk_idx = 0
    while idx < n:
        end = min(n, idx + max_chars)
        chunk_text = text[idx:end]
        chunks.append(
            {
                "index": chunk_idx,
                "start": idx,
                "end": end,
                "text": chunk_text,
            }
        )
        chunk_idx += 1
        if end == n:
            break
        idx = max(0, end - overlap)

    print(f"[INFO] 已切分为 {len(chunks)} 个 chunk (max_chars={max_chars}, overlap={overlap})")
    return chunks


_max_output_param_name: Optional[str] = None


def _max_output_kw_name(client: OpenAI) -> str:
    """Prefer max_completion_tokens when the SDK exposes it; else max_tokens."""
    global _max_output_param_name
    if _max_output_param_name is not None:
        return _max_output_param_name
    try:
        sig = inspect.signature(client.chat.completions.create)
        params = sig.parameters
        if "max_completion_tokens" in params:
            _max_output_param_name = "max_completion_tokens"
        elif "max_tokens" in params:
            _max_output_param_name = "max_tokens"
        else:
            _max_output_param_name = "max_tokens"
    except (TypeError, ValueError):
        _max_output_param_name = "max_tokens"
    return _max_output_param_name


def _max_completion_budget(dense: bool, attempt: int) -> int:
    """
    Output token budget for chat.completions.
    dense 模式要求更多 facts，JSON 体积大；JSON 解析失败后的重试再略放宽。
    """
    if dense:
        return 8192
    if attempt > 1:
        return 6144
    return 4096


def _fact_dedup_key(text: str) -> str:
    """Normalize fact text for cross-chunk deduplication (overlap regions)."""
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t.casefold()


def call_llm_for_chunk(chunk_text: str, attempt: int = 1, dense: bool = False) -> Dict[str, Any]:
    """
    调用 OpenAI，对单个 chunk 抽取 facts。
    - attempt > 1：用于 JSON 解析失败后的重试（提示“上一次 JSON 不合法”）。
    - dense = True：用于“当前 chunk 文本很长但事实过少”的第二轮密集抽取，会额外要求多抽一些 facts。
    """
    extra_hint_parts = []

    if attempt > 1:
        extra_hint_parts.append(
            "⚠️ 注意：上一次返回的 JSON 解析失败（可能因截断或转义错误）。"
            "请在 facts 总数不超过 25 条的前提下，优先输出一份合法、完整的 JSON；"
            "若必要可适当减少条数，但仍应覆盖本块中最重要的、可分属不同维度的信息。"
        )

    if dense:
        extra_hint_parts.append(
            "⚠️ 本块较长但上次抽取偏少。请在 facts 总数不超过 25 条的前提下，"
            "尽量补全本块中与团队、目标、策略、创新、可行性相关的关键事实，并覆盖不同主题；"
            "不要只聚焦单一侧面；不要为凑条数编造。"
        )

    extra_hint = ""
    if extra_hint_parts:
        extra_hint = "\n\n" + "\n".join(extra_hint_parts)

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的事实抽取器，只能基于给定文本块抽取原子事实，不得编造。"
            "典型输入为各类商业与投融资书面材料（不限行业）；若文本类型不同，仍以原文为准，勿套用未出现的要素。",
        },
        {
            "role": "user",
            "content": build_fact_prompt()
            + extra_hint
            + "\n\n=== 文本块开始 ===\n"
            + chunk_text.strip(),
        },
    ]

    cli = _get_client()
    out_budget = _max_completion_budget(dense, attempt)
    out_kw = {_max_output_kw_name(cli): out_budget}
    resp = cli.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        **out_kw,
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("[WARN] JSON 解析失败，返回原始内容，便于你排查：")
        print(raw)
        if attempt == 1:
            print("[INFO] 尝试使用收缩版 prompt 重试该 chunk ...")
            return call_llm_for_chunk(chunk_text, attempt=2, dense=dense)
        # 第二次还失败就直接抛出
        raise e

    if not isinstance(data, dict):
        data = {"facts": []}
    if "facts" not in data or not isinstance(data["facts"], list):
        data["facts"] = []

    return data

# ===== 市场类关键词 & 识别函数 =====

_MARKET_CN = [
    "市场", "市场规模", "市场容量", "市场需求", "市场前景", "市场潜力",
    "目标市场", "细分市场", "市场份额", "市占率", "市占", "渗透率",
    "客户", "客户群体", "目标客户", "目标人群",
    "患者群体", "目标患者",
    "销售", "销售额", "销量", "营收", "收入", "收益", "毛利", "毛利率",
    "定价", "价格", "报销", "支付方", "医保", "保险",
    "商业化", "商业模式", "商业机会",
    "竞争", "竞品", "竞争对手", "竞争格局",
    "融资", "估值", "领投", "跟投",
    "CAGR", "增长率"
]

_MARKET_EN = [
    "market", "market size", "market volume", "market demand", "market potential",
    "target market", "segment", "niche",
    "market share", "penetration",
    "customer", "customers", "client", "clients",
    "patient population", "target patients",
    "sales", "revenue", "turnover", "income",
    "pricing", "price", "reimbursement", "payer", "insurance",
    "commercialization", "commercialisation", "business model",
    "competition", "competitive", "competitor", "competitors",
    "cagr", "growth rate"
]


def _looks_like_market_fact(text: str) -> bool:
    """
    判断一条 fact 的文本是否“明显是市场/商业相关”的内容。
    注意：这里只用来纠偏 type，对 risk/mitigation 不覆盖。
    """
    if not text:
        return False
    t = text.lower()

    # 中文关键字
    if any(k in text for k in _MARKET_CN):
        return True

    # 英文关键字
    if any(k in t for k in _MARKET_EN):
        return True

    return False

def _infer_dims_from_text(text: str) -> List[str]:
    """
    当 LLM 没有给出 dimensions 且 type 也无法可靠映射时，
    用中英文关键词做一轮粗略的维度推断，尽量不要丢掉有用信息。
    这里只做“补充”，不会覆盖已有维度。
    """
    t_lower = (text or "").lower()
    t = text or ""

    candidates = set()

    # ---- team ----
    team_keywords = [
        "团队", "小组", "联合体", "合作单位", "合作方", "协作单位",
        "研究者", "研究团队", "临床团队", "项目组",
        "负责人", "项目负责人", "带头人",
        "教授", "副教授", "主任", "专家", "研究员", "博士", "博士后",
        "医院", "大学", "研究所", "中心", "实验室",
        "ceo", "coo", "cto", "cso", "vp", "vice president",
        "founder", "co-founder", "chief executive officer",
    ]
    if any(k in t for k in team_keywords) or any(k in t_lower for k in team_keywords):
        candidates.add("team")

    # ---- objectives ----
    obj_keywords_cn = [
        "目标", "总体目标", "阶段性目标", "里程碑", "阶段性里程碑",
        "计划", "任务", "工作包", "kpi", "终点", "主要终点", "次要终点",
        "本项目旨在", "本项目将", "本项目计划", "预期达到", "希望实现",
        "营收目标", "收入目标", "GMV", "估值目标", "融资目标", "市占目标",
    ]
    obj_keywords_en = [
        "aim", "aims to", "aimed to",
        "objective", "objectives", "goal", "goals",
        "milestone", "milestones", "endpoint", "endpoints",
        "is designed to", "seeks to", "intends to", "in order to",
    ]
    if any(k in t for k in obj_keywords_cn) or any(k in t_lower for k in obj_keywords_en):
        candidates.add("objectives")

    # ---- strategy ----
    strat_keywords_cn = [
        "策略", "路径", "路线", "方案", "技术路线", "实施方案",
        "商业模式", "市场进入", "商业化", "推广策略",
        "合作模式", "运营模式", "联合开发", "授权引进",
        "供应链", "渠道", "经销商", "分销", "SaaS", "订阅",
        "市场", "市场规模", "市场需求", "市场前景", "市场潜力",
        "市场份额", "竞争格局", "竞品", "竞争对手"
    ]
    strat_keywords_en = [
        "strategy", "strategies", "pathway", "roadmap",
        "commercial", "commercialization", "business model",
        "market entry", "go-to-market", "go to market",
        "market", "market size", "market demand", "market potential",
        "market share", "competitive landscape", "competition", "competitor", "competitors",
        "partnership", "licensing", "co-development",
        "development plan", "regulatory strategy",
    ]

    if any(k in t for k in strat_keywords_cn) or any(k in t_lower for k in strat_keywords_en):
        candidates.add("strategy")

    # ---- innovation ----
    inno_keywords_cn = [
        "创新", "创新性", "差异化", "独特", "首创", "领先", "颠覆",
        "新一代", "新型", "原创", "填补空白", "突破性", "首个", "第一例",
    ]
    inno_keywords_en = [
        "novel", "novelty", "innovative", "innovation",
        "differentiated", "differentiation", "unique",
        "first-in-class", "best-in-class", "state-of-the-art",
        "cutting-edge", "breakthrough", "original", "disruptive",
        "fills the gap", "fill the gap",
    ]
    if any(k in t for k in inno_keywords_cn) or any(k in t_lower for k in inno_keywords_en):
        candidates.add("innovation")

    # ---- feasibility ----
    feas_keywords_cn = [
        "可行性", "可行", "可实施", "资源", "平台", "基础设施",
        "预算", "经费", "资金", "成本", "成本负担",
        "风险", "挑战", "瓶颈", "不确定性",
        "时间表", "进度", "周期", "排期",
        "入组难度", "依从性", "工作量", "实施复杂度",
    ]
    feas_keywords_en = [
        "feasibility", "feasible", "resource", "resources", "infrastructure",
        "budget", "funding", "cost", "costs", "cost-effectiveness",
        "risk", "risks", "challenge", "challenges", "bottleneck", "uncertainty",
        "timeline", "schedule", "timeframe",
        "enrollment", "recruitment", "compliance", "adherence", "burden",
    ]
    if any(k in t for k in feas_keywords_cn) or any(k in t_lower for k in feas_keywords_en):
        candidates.add("feasibility")

    # ---- 专门为“市场分析段落”兜底：强行打上 strategy / objectives ----
    market_keywords_cn = [
        "市场", "市场规模", "市场分析", "市场份额", "市场占有率",
        "cagr", "复合年增长率", "销售额", "营收", "收入", "销售收入",
        "增长率", "增长幅度", "客户", "用户", "消费群体",
    ]
    market_keywords_en = [
        "market size", "market", "cagr", "market share", "share of",
        "sales", "revenue", "revenues", "turnover",
        "growth rate", "compound annual growth", "customer", "customers",
        "payer", "payers",
    ]
    if any(k in t for k in market_keywords_cn) or any(k in t_lower for k in market_keywords_en):
        candidates.add("strategy")
        candidates.add("objectives")

    return [d for d in candidates if d in VALID_DIMENSIONS]

def mark_numeric_suspect(fact: Dict[str, Any], chunk_text: str) -> Dict[str, Any]:
    """
    对包含数字的 fact 做简单校验：
    - 把 fact.text 里的数字片段（连续数字，不管是年份/金额）提取出来
    - 如果某个数字完全不出现在 chunk_text 中，则认为这条 fact 存在数字幻觉风险
    - 加 meta.suspect_numeric = True/False
    """
    text = fact.get("text", "") or ""
    nums = re.findall(r"\d+", text)
    if not nums:
        return fact  # 没数字，不管

    chunk_flat = (chunk_text or "").replace(" ", "")
    suspect = False
    for n in nums:
        if n not in chunk_flat:
            suspect = True
            break

    meta = fact.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["suspect_numeric"] = suspect
    fact["meta"] = meta
    return fact

def normalize_fact(
    fact: Dict[str, Any],
    proposal_id: str,
    chunk_index: int,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    给每条 fact 填上 meta 信息；清洗 dimensions / type。
    同时对维度标签做 type→dimensions 的通用“补标签”映射（支持多维度），
    尽量保证五个维度的信息都不会被漏掉。
    """
    text = fact.get("text", "")
    if not isinstance(text, str):
        text = str(text)

    # 原始维度标签清洗
    dims = fact.get("dimensions", [])
    if not isinstance(dims, list):
        dims = []
    dims_clean = [d for d in dims if isinstance(d, str) and d in VALID_DIMENSIONS]

    type_val = fact.get("type", "other")
    if type_val not in VALID_TYPES:
        type_val = "other"

    # ===== 市场类事实的自动纠偏（在 type→dimensions 映射之前）=====
    # 如果 LLM 没有标成 market，但文本里明显是市场/商业内容，则强制改为 "market"
    # （避免所有市场信息都被丢在 "other" 或 "product" 里）
    if type_val not in ["market", "risk", "mitigation"]:
        if _looks_like_market_fact(text):
            type_val = "market"

    # 先把已有维度放进一个 set，后面按 type / 文本内容补充
    dim_set = set(dims_clean)

    # ===== 1. Config-driven type→dimensions mapping =====
    # Try the centralized config first; fall back to hardcoded rules
    type_dims_map = _get_type_to_dims_map()
    if type_dims_map and type_val in type_dims_map:
        mapped = type_dims_map[type_val]
        if mapped:
            dim_set.update(mapped)
    else:
        # Hardcoded fallback (preserves original behavior)
        _FALLBACK_TYPE_DIMS = {
            "team_member": ["team"],
            "org_structure": ["team"],
            "collaboration": ["team", "strategy"],
            "pipeline": ["objectives", "strategy"],
            "milestone": ["objectives", "feasibility"],
            "clinical_design": ["objectives", "strategy", "feasibility"],
            "market": ["objectives", "strategy", "feasibility"],
            "product": ["objectives", "strategy", "feasibility"],
            "tech_route": ["strategy", "feasibility"],
            "regulatory": ["strategy", "feasibility"],
            "funding_source": ["strategy", "feasibility"],
            "ip_asset": ["innovation", "feasibility"],
            "evidence": ["innovation", "feasibility"],
            "ai_model": ["innovation", "strategy"],
            "resource": ["feasibility"],
            "budget_item": ["feasibility"],
            "risk": ["feasibility"],
            "mitigation": ["feasibility"],
        }
        if type_val in _FALLBACK_TYPE_DIMS:
            dim_set.update(_FALLBACK_TYPE_DIMS[type_val])

    # ===== 2. 如果还是没维度，用文本关键词再判断一轮 =====
    if not dim_set:
        inferred = _infer_dims_from_text(text)
        for d in inferred:
            dim_set.add(d)

    # ===== 3. 最终兜底：还没有，就放到 feasibility，避免彻底丢失 =====
    if not dim_set:
        dim_set.add("feasibility")

    dims_final = [d for d in dim_set if d in VALID_DIMENSIONS]

    meta = fact.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta.update(
        {
            "proposal_id": proposal_id,
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
        }
    )

    # ==== 计算 primary_dimension ====
    primary_dim = None
    if dims_final:
        for d in VALID_DIMENSIONS:
            if d in dims_final:
                primary_dim = d
                break
        if primary_dim is None:
            primary_dim = dims_final[0]
    else:
        cfg = _get_domain_config()
        primary_dim = cfg.fallback_dimension if cfg else "feasibility"

    return {
        "text": text.strip(),
        "dimensions": dims_final,
        "type": type_val,
        "primary_dimension": primary_dim,
        "meta": meta,
    }

def run_extract(proposal_id: str, max_chars: int = 1800, overlap: int = 400):
    full_text = load_full_text(proposal_id)
    chunks = make_chunks(full_text, max_chars=max_chars, overlap=overlap)

    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_facts.jsonl"

    total_facts = 0
    dup_skipped = 0
    seen_fact_keys: set = set()
    # 统计每个维度的 fact 数量，便于 sanity check
    dim_counts = {dim: 0 for dim in VALID_DIMENSIONS}

    _write_progress(0, len(chunks), proposal_id)
    with out_path.open("w", encoding="utf-8") as f_out:
        for ch in chunks:
            idx = ch["index"]
            chunk_len = ch["end"] - ch["start"]
            print(f"\n[INFO] 处理 chunk {idx+1}/{len(chunks)} (chars={chunk_len})...")

            # 第一次正常抽取
            data = call_llm_for_chunk(ch["text"])
            facts = data.get("facts", [])
            if not isinstance(facts, list):
                facts = []

            # 如果 chunk 很长，但抽取的 facts 过少，则尝试 dense 模式重跑一次
            if chunk_len >= 1200 and len(facts) < 5:
                print(
                    f"[INFO] 当前 chunk 文本较长(chars={chunk_len})，但只抽取到 {len(facts)} 条事实，"
                    f"尝试使用 dense 模式重试以提高召回..."
                )
                dense_data = call_llm_for_chunk(ch["text"], dense=True)
                dense_facts = dense_data.get("facts", [])
                if isinstance(dense_facts, list) and len(dense_facts) > len(facts):
                    print(
                        f"[INFO] dense 模式抽取到 {len(dense_facts)} 条事实（优于原先的 {len(facts)} 条），"
                        f"采用 dense 结果。"
                    )
                    facts = dense_facts
                else:
                    print(
                        f"[INFO] dense 模式未显著提升（原 {len(facts)} 条，dense={len(dense_facts)} 条），"
                        f"保留原始抽取结果。"
                    )

            normalized_list = []
            for fact in facts:
                if not isinstance(fact, dict):
                    continue

                # 先基于 chunk_text 标记 suspect_numeric
                fact = mark_numeric_suspect(fact, ch["text"])

                norm = normalize_fact(
                    fact,
                    proposal_id=proposal_id,
                    chunk_index=idx,
                    start=ch["start"],
                    end=ch["end"],
                )

                # 过滤掉空文本
                if norm["text"]:
                    normalized_list.append(norm)

            for fact in normalized_list:
                dkey = _fact_dedup_key(fact.get("text", ""))
                if not dkey:
                    continue
                if dkey in seen_fact_keys:
                    dup_skipped += 1
                    continue
                seen_fact_keys.add(dkey)
                f_out.write(json.dumps(fact, ensure_ascii=False) + "\n")
                total_facts += 1
                # 更新维度计数
                for d in fact.get("dimensions", []):
                    if d in dim_counts:
                        dim_counts[d] += 1

            print(
                f"[INFO] 该 chunk 最终写入 {len(normalized_list)} 条事实，"
                f"目前累计 {total_facts} 条。"
            )
            _write_progress(idx + 1, len(chunks), proposal_id)

    print(f"\n[OK] 已写出事实文件: {out_path} (总事实数={total_facts})")
    if dup_skipped:
        print(f"[INFO] 因与先前 chunk 文本重复而跳过的事实条数: {dup_skipped}")

    # ===== 全局维度分布检查 =====
    print("\n[SUMMARY] 维度分布统计（基于 raw_facts.jsonl）：")
    for dim in VALID_DIMENSIONS:
        print(f"  - {dim}: {dim_counts[dim]} facts")

    # 简单 sanity check：如果某个维度明显偏少，打个警告（这里只做提示，不终止）
    if total_facts > 0:
        avg = total_facts / len(VALID_DIMENSIONS)
        for dim in VALID_DIMENSIONS:
            if dim_counts[dim] < max(8, 0.25 * avg):
                print(
                    f"[WARN] 维度 {dim} 的事实数仅 {dim_counts[dim]}，"
                    f"显著低于平均值 {avg:.1f}，可能存在抽取不足或映射偏差，"
                    f"建议检查 raw_facts.jsonl 或适当调整 Prompt/映射。"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: 按 chunk 抽取原子事实（raw_facts.jsonl）"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="提案 ID（对应 src/data/prepared/<proposal_id>）",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1800,
        help="每个 chunk 最大字符数（默认 1800）",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=400,
        help="chunk 之间的字符重叠数（默认 400）",
    )
    args = parser.parse_args()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = find_latest_prepared_proposal()

    run_extract(pid, max_chars=args.max_chars, overlap=args.overlap)


if __name__ == "__main__":
    main()
