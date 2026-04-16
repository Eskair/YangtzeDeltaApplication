# -*- coding: utf-8 -*-
"""
llm_answering.py · Proposal-aware LLM Answering (ChatGPT + DeepSeek, no web search)

新版职责：
- 基于「维度抽取管线」生成的提案事实（dimensions_from_facts）+ 生成的问题集，
  为每个维度的问题生成“强关联该提案”的结构化回答。
- 不再依赖外部 Web 检索；只使用：
    • 提案维度事实（summary/key_points/risks/mitigations/numbers 等）
    • LLM 自身通识做解释，但禁止脑补新实验/新数字/新机构
- 输出结构保持兼容：
    data/refined_answers/{pid}/all_refined_items.json
    data/refined_answers/{pid}/chatgpt_raw.json
    data/refined_answers/{pid}/deepseek_raw.json

本版新增：
- 明确要求回答中包含“行业基准 / baseline 对比”和“常见坑 & 证据要求”两类内容；
- 这些通识部分必须写成“行业普遍情况/一般建议”，不能写成项目已经达成的事实；
- 新增字段 general_insights：专门装“行业基准 / 常见坑 / 证据要求”这类专家经验层；
- 修复 answer 分点格式问题，避免出现 “1. 1. xxx” 这种双重编号。
"""

import os
import sys
import json
import time
import re
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv

# ========== 环境 & 路径 ==========
load_dotenv()
ROOT = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT / "data"
PROGRESS_FILE = DATA_DIR / "step_progress.json"


def _write_llm_progress(done: int, total: int, pid: str = "") -> None:
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass
EXTRACTED_DIR = DATA_DIR / "extracted"
PARSED_DIR = DATA_DIR / "parsed"
CONFIG_QS_DEFAULT = DATA_DIR / "config" / "question_sets" / "generated_questions.json"
OUT_REFINED = DATA_DIR / "refined_answers"


def _init_answering_dim_order():
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.config import get_config
        return get_config().dimension_names
    except Exception:
        return ["team", "objectives", "strategy", "innovation", "feasibility"]


DIM_ORDER = _init_answering_dim_order()

# ========== SDK ==========
try:
    from openai import OpenAI as OpenAIClient
except Exception:  # pragma: no cover
    OpenAIClient = None

# ========== 变体 & provider 能力 ==========
VARIANTS = ["default", "risk", "implementation"]
TEMP_BY_VARIANT = {"default": 0.25, "risk": 0.35, "implementation": 0.30}

PROVIDER_CAPS = {
    "openai": {"json_mode": True, "batch_ok": True},
    "deepseek": {"json_mode": False, "batch_ok": False},
}


def provider_caps(provider: str) -> Dict[str, Any]:
    return PROVIDER_CAPS.get(provider, {"json_mode": True, "batch_ok": True})


CONF_MIN, CONF_MAX = 0.50, 0.92
MAX_LIST_LEN = 10

# ========== 工具函数 ==========
def read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def detect_latest_pid() -> str:
    if not EXTRACTED_DIR.exists():
        return "unknown"
    cands = [d for d in EXTRACTED_DIR.iterdir() if d.is_dir()]
    if not cands:
        return "unknown"
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0].name

def _flatten_list_field(block: Dict[str, Any], keys: List[str], limit: int = 12) -> List[str]:
    items: List[str] = []
    for k in keys:
        v = block.get(k)
        if not v:
            continue
        if isinstance(v, str):
            items.extend([x.strip() for x in re.split(r"[;\n]", v) if x.strip()])
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        items.append(s)
    uniq, seen = [], set()
    for x in items:
        key = x.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
        if len(uniq) >= limit:
            break
    return uniq


def _build_dim_context_text(dim: str, block: Dict[str, Any]) -> str:
    """
    把从 facts 管线来的一个维度 block 转成适合放进 prompt 的 context 文本。
    控制长度，优先 summary / key_points / risks / mitigations / numbers。
    """
    if not isinstance(block, dict):
        block = {}

    summary = str(block.get("summary") or "").strip()
    key_points = _flatten_list_field(block, ["key_points", "keypoints", "key_facts", "bullets"], limit=10)
    risks = _flatten_list_field(block, ["risks", "risk_points"], limit=8)
    mitigations = _flatten_list_field(block, ["mitigations", "mitigation_points"], limit=8)
    numbers = _flatten_list_field(block, ["numbers", "key_numbers"], limit=8)

    parts: List[str] = []
    if summary:
        parts.append(f"【{dim} 概览】{summary}")
    if key_points:
        parts.append("【关键要点】" + "；".join(key_points))
    if risks:
        parts.append("【主要风险/不确定性】" + "；".join(risks))
    if mitigations:
        parts.append("【已有缓解措施】" + "；".join(mitigations))
    if numbers:
        parts.append("【关键量化信息】" + "；".join(numbers))

    text = "\n".join(parts)
    if len(text) > 2000:
        text = text[:2000]
    return text


def load_dimension_context(pid: str, dim_file: Optional[Path]) -> Dict[str, str]:
    """
    返回：{dim: context_text}
    若找不到文件或结构异常，所有维度返回空字符串（模型会退化成通识回答）
    """
    ctx: Dict[str, str] = {d: "" for d in DIM_ORDER}
    if dim_file is None or not dim_file.exists():
        return ctx

    try:
        raw = read_json(dim_file)
    except Exception:
        return ctx

    if isinstance(raw, dict) and "dimensions" in raw and isinstance(raw["dimensions"], dict):
        root = raw["dimensions"]
    else:
        root = raw if isinstance(raw, dict) else {}

    for dim in DIM_ORDER:
        block = root.get(dim) or {}
        ctx[dim] = _build_dim_context_text(dim, block)
    return ctx


# ========== 问题集 & 监管提示 ==========
def get_q_list(block: Any) -> List[str]:
    if isinstance(block, dict) and isinstance(block.get("questions"), list):
        return [q for q in block["questions"] if isinstance(q, str)]
    if isinstance(block, list):
        return [q for q in block if isinstance(q, str)]
    return []


def _load_reg_hints(qs_cfg: Dict[str, Any], dim: str, limit: int = 8) -> List[str]:
    """
    从问题集里抓一点监管/术语 hints，但只作为“可选方向提示”，不强依赖。
    """
    block = qs_cfg.get(dim, {}) or {}
    hints = block.get("search_hints") or block.get("reg_hints") or []
    out: List[str] = []
    if isinstance(hints, list):
        for h in hints:
            if not isinstance(h, str):
                continue
            s = h.strip()
            if not s:
                continue
            out.append(s)
    uniq, seen = [], set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
        if len(uniq) >= limit:
            break
    return uniq

# ========== Prompt 相关 ==========
SYSTEM_CN = (
    "你是一名严格的项目评审专家。"
    "你的任务是：在完整阅读给定的【提案事实】之后，围绕问题进行“与该提案强相关”的专业分析。"
    "原则："
    "1）必须优先基于【提案事实】给出结论；"
    "2）不得发明提案中未出现的新试验、新数据、新机构或具体数字；"
    "3）当你在【提案事实】中找不到某一类信息时，只能说“当前材料中未看到关于 X 的具体说明”，"
    "   并优先写成“已有 A/B，但在 C/D 方面细节不足”；严禁使用“完全没有分析”“未进行任何评估”"
    "   “未提供任何信息”等绝对化表述；"
    "4）可以使用行业通识解释这些事实的意义，但不得虚构具体注册号/临床编号/专利号/精确样本量等细节；"
    "5）回答必须结构化、条理清晰，适合作为专家评审报告的组成部分；"
    "6）你需要在回答中补充“行业基准/baseline 对比”和“类似项目常见的坑与证据要求”等通识内容，"
    "   但这些通识部分必须明确表述为“行业普遍情况/一般建议”，不得写成项目已经达成的事实；"
)

def _schema_structured() -> str:
    return """
请严格返回 JSON 对象，键名固定：
{
  "answer": "分点形式的主回答（中文，条理清晰，含项目现状+问题+改进方向等）",
  "claims": ["关键可验证结论1","关键可验证结论2","..."],
  "evidence_hints": ["哪条提案事实/段落支撑对应结论，或需进一步核查的线索"],
  "general_insights": ["行业基准/类似项目常见做法/常见坑与证据要求（通识，不代表本项目已完成；建议最后一条给出统一免责声明）"],
  "topic_tags": ["维度内的小主题/标签"],
  "confidence": 0.0,
  "caveats": "限制/注意事项（若无可空）"
}
""".strip()


def _variant_instructions(variant_id: str) -> str:
    if variant_id == "risk":
        return (
            "变体：risk（风险视角）——重点：\n"
            "- 优先识别不确定性、缺失信息、潜在合规或技术风险；\n"
            "- 对每个主要风险给出“风险来源 + 可能影响 + 建议补充材料”；\n"
            "- 若提案事实中未覆盖关键环节，要显式指出“信息缺口”；\n"
            "- 在 general_insights 中，总结同类项目在该维度常见的风险模式、监管关注点和证据要求，"
            "  明确说明这些是行业通识，不代表本项目已经满足。\n"
        )
    if variant_id == "implementation":
        return (
            "变体：implementation（落地视角）——重点：\n"
            "- 列出 4–8 条“接下来 3–12 个月可执行的具体动作”，每条包含：动作主体/对象 + 具体步骤 + 预期产出；\n"
            "- 动作必须与提案当前状态匹配，不得假设已完成的工作；\n"
            "- 可以提及需要收集/补充的证据或文档类型（而不是凭空给出结论）；\n"
            "- 在 general_insights 中，总结同类项目在该维度的主流落地路径、里程碑拆解和常见踩坑点。\n"
        )
    return (
        "变体：default（综合视角）——重点：\n"
        "- 结合该维度的“当前方案/优势/问题/建议”，给出整体评价；\n"
        "- 既要指出做得好的地方，也要指出存在的不足或不确定性；\n"
        "- 至少包含：现状概括、优势点、主要问题、改进方向四类内容；\n"
        "- 在 answer 中，你需要显式区分：\n"
        "  a) 基于【提案事实】得出的本项目现状与问题；\n"
        "  b) 行业基准 / baseline 对比（同类项目通常达到什么水平、需要哪些能力/数据/里程碑）；\n"
        "  c) 同类项目常见的坑和证据要求/监管关注点；\n"
        "- 在 claims 中，优先写“提案已经说明了什么/在哪些方面信息仍然有限”，"
        "  避免使用“项目未提供/没有进行任何……”这类绝对否定，如果只能判断信息有限，"
        "  应写成“现有材料未看到关于 X 的更详细说明”。\n"
        "- 在 general_insights 字段中，对 b) 和 c) 做更加抽象的行业通识总结，"
        "  明确使用“通常/一般而言/在同类项目中”等措辞，避免暗示本项目已经达成这些条件。\n"
    )

def build_single_prompt(
    dimension: str,
    question: str,
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
) -> str:
    reg_txt = "；".join(reg_hints) if reg_hints else "无特别提示"
    return f"""
维度：{dimension}

[提案事实]（只能在这里面引用具体细节；如无相关信息请据实指出）：
{proposal_context or "（该维度的提案事实为空，仅可做通识性分析）"}

[可选监管/术语方向提示]（非必须引用，如与提案无关可忽略）：
{reg_txt}

题目：
{question.strip()}

{_variant_instructions(variant_id)}

回答要求：
- 语言：中文（专有名词可保留英文）；
- 结构：answer 必须以 3–8 条分点给出，每条前加“1. 2. 3.” 等编号，每条尽量控制在一到两句话；
- 内容结构建议（非强制格式，但需覆盖）：\n
  1）先基于[提案事实]总结本项目在该维度的“现状 + 优势 + 主要问题”；\n
  2）再用 1–3 条要点，对标“行业基准 / baseline”，说明同类项目通常在该维度需要达到什么水平（注意：这是行业通识，不代表本项目已达到）；\n
  3）再用 1–3 条要点，总结“同类项目常见的坑、证据要求、监管或临床关注点”；\n
- 关联度：优先基于[提案事实]分析该项目的真实情况；对于[提案事实]中没有的信息，只能用“缺失/需补充”的方式描述，不得假设已经存在；\n
- 若[提案事实]中已经给出某一方面的部分信息（例如有市场规模但缺少细分、有竞争方列表但缺少对位分析），"
"  必须写成“已有……但在……方面仍缺乏具体细节”，不得笼统说“未提供市场分析/未进行竞争对手评估”；\n
- 通识与事实的区分：凡是基于行业经验的内容，需要在句中用“通常/一般而言/在同类项目中”等词标识清楚，避免写成好像本项目已经完成这些工作；\n
- general_insights 字段：请单独列出 3–8 条不依赖本项目具体事实的“行业基准/常见坑/证据要求”要点，这些内容应可用于评估任何类似项目，且必须表述为通识建议；最后一条建议写成类似 “以上为行业通识建议，不代表本项目已经达成相关要求。” 这样的统一免责声明；\n
- 审慎性：避免下结论式的绝对语气，多用“可能/建议/需要确认”等方式，并尽量指出需要的佐证材料类型；\n
- 仅输出一个 JSON 对象，不得有任何额外说明文字或 Markdown 围栏。

{_schema_structured()}
""".strip()


def build_batch_prompt(
    dimension: str,
    questions: List[str],
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
) -> str:
    reg_txt = "；".join(reg_hints) if reg_hints else "无特别提示"
    q_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""
你将针对同一维度下的多道问题，基于同一份【提案事实】给出与该提案高度相关的回答。
请严格以对象数组 JSON 返回结果，格式为：{{"answers":[<对象1>,<对象2>,...]}}，数组长度必须与题目数一致。

维度：{dimension}

[提案事实]（只能在这里面引用具体细节；如无相关信息请据实指出）：
{proposal_context or "（该维度的提案事实为空，仅可做通识性分析）"}

[可选监管/术语方向提示]（非必须引用，如与提案无关可忽略）：
{reg_txt}

问题列表：
{q_block}

{_variant_instructions(variant_id)}

统一回答要求：
- 每道题的 answer：3–8 条分点，每条前加数字编号；条数不足时宁可只写 3–4 条扎实要点，也不要凑模板口号；
- 建议在 answer 中显式覆盖三类信息：\n
  a) 仅基于[提案事实]得出的本项目现状/优势/问题；\n
  b) 行业基准 / baseline 对比（同类项目通常需要的团队能力、数据规模、里程碑等——需标明是行业通识）；\n
  c) 同类项目在该维度常见的坑、证据要求、监管或临床关注点；\n
- 每道题的 claims：2–6 条可以被事后核对的结论，优先基于[提案事实]；若信息不足，请将“信息缺口”本身写入 claims；
- 每道题的 evidence_hints：指明“哪类提案事实/哪一段内容/哪类文档可以支撑这些结论”，避免写空泛模板；
- 每道题的 general_insights：3–8 条不依赖本项目具体事实的“行业基准/常见坑/证据要求”要点，只能写成行业普遍情况/一般建议，不得暗示本项目已经达成；建议其中最后一条写成统一的免责声明，例如 “以上为行业通识建议，并不代表本项目已经满足相关条件。”；\n
- 若[提案事实]中已经给出某一方面的部分信息（例如有市场规模、竞品列举、风险表等），"
"  只能评价为“现有描述在……方面仍不够细化/缺乏定量对比”，不得一概写成“项目未提供市场需求分析/竞争对手评估”等绝对否定；\n
- 不得发明提案中不存在的新实验/新数据/新机构/具体注册号；对未给出的信息，只能以“需补充/需确认”的方式表达；\n
- 仅输出一个 JSON 对象，不得有额外文字或 Markdown 围栏。

{_schema_structured()}
""".strip()


def build_refine_prompt(candidate_obj: Dict[str, Any], proposal_context: str, dimension: str) -> str:
    original = json.dumps(candidate_obj, ensure_ascii=False, indent=2)
    return f"""
请在不引入任何超出【提案事实】的新信息的前提下，对下面的结构化回答做一次快速自我复核：
- 删改过于武断或缺乏依据的强结论；
- 若某条结论在【提案事实】中找不到依据，请改写为“需要补充的材料/信息”；
- 优化 answer 的分点表达，让每条更具体、更可执行，但不要发明新试验/新数据；
- 检查 general_insights：确保里面只包含“行业基准/类似项目常见做法/常见坑与证据要求”等通识内容，不得把本项目的具体事实写进 general_insights；\n
- 保持 JSON 结构和字段名完全不变（包括 general_insights）。

维度：{dimension}

[提案事实]：
{proposal_context or "（该维度的提案事实为空，仅可做轻度通识性修正）"}

原候选：
{original}
""".strip()


# ========== LLM 调用基础 ==========
def _with_retry(fn, max_tries: int = 4, base: float = 0.8):
    for i in range(max_tries):
        try:
            return fn()
        except Exception:
            if i == max_tries - 1:
                raise
            time.sleep(base * (2 ** i) + random.random() * 0.2)


ERR_PATTERNS = [
    r"\[?ERROR[:\]]",
    r"\bHTTP\s*4\d{2}\b",
    r"\bHTTP\s*5\d{2}\b",
    r"insufficient[_\s-]?quota",
    r"invalid[_\s-]?api[_\s-]?key",
    r"request\s+timed\s*out",
    r"rate\s*limit",
    r"payment\s*required",
    r"bad gateway",
    r"service unavailable",
    r"connection (?:reset|refused)",
]
_err_re = re.compile("|".join(ERR_PATTERNS), re.I)


def is_error_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return bool(_err_re.search(t))


def _chat_completion_json(
    client,
    model: str,
    system_text: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    force_json: bool,
):
    def call(json_mode: bool):
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    try:
        if force_json:
            resp = _with_retry(lambda: call(json_mode=True))
            out = (resp.choices[0].message.content or "").strip()
            if is_error_text(out):
                raise RuntimeError("provider_error_json_mode")
            return out
        else:
            resp = _with_retry(lambda: call(json_mode=False))
            out = (resp.choices[0].message.content or "").strip()
            if is_error_text(out):
                raise RuntimeError("provider_error_text_mode")
            return out
    except Exception:
        if force_json:
            resp = _with_retry(lambda: call(json_mode=False))
            out = (resp.choices[0].message.content or "").strip()
            if is_error_text(out):
                raise RuntimeError("provider_error_text_mode")
            return out
        raise


def _safe_parse_json_plus(txt: str) -> Optional[Any]:
    if is_error_text(txt):
        return None

    t = (txt or "").replace("\ufeff", "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*\n?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\n?\s*```\s*$", "", t, flags=re.IGNORECASE)
    t = t.replace("\xa0", " ")

    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return {"answers": obj}
        return obj
    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
    if m:
        frag = m.group(1)
        try:
            tmp = json.loads(frag)
            if isinstance(tmp, list):
                return {"answers": tmp}
            return tmp
        except Exception:
            pass

    # 尝试给常见字段名补双引号，包括 general_insights
    cand = re.sub(
        r"(\banswer|claims|evidence_hints|general_insights|topic_tags|confidence|caveats\b)\s*:",
        r'"\1":',
        t,
    )
    try:
        return json.loads(cand)
    except Exception:
        return None


# ========== 模型初始化 ==========
def init_openai():
    if OpenAIClient is None:
        return None, None
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None, None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    client = OpenAIClient(api_key=key)
    print(f"✅ 已加载 OpenAI 模型：{model}")
    return client, model


def init_deepseek():
    if OpenAIClient is None:
        return None, None
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None, None
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    client = OpenAIClient(api_key=key, base_url=base)
    print(f"✅ 已加载 DeepSeek 模型：{model}")
    return client, model


# ========== 答案规范化 & 后处理 ==========
def _norm_str(s: Any) -> str:
    return " ".join(str(s or "").strip().split())


def _uniq_cut(lst: List[Any], k: int = MAX_LIST_LEN) -> List[str]:
    seen, out = set(), []
    for x in lst:
        t = _norm_str(x)
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= k:
            break
    return out


RE_DATE = re.compile(r"\b(20\d{2}|19\d{2})([-/.]|年)\d{1,2}([-/\.日]|月)\d{1,2}\b|\b(Q[1-4]\s*-\s*20\d{2})\b", re.I)
RE_MONEY = re.compile(
    r"\b(\$|USD|EUR|CNY|RMB|CAD)\s*\d{2,}(,\d{3})*(\.\d+)?\b|\b\d+(\.\d+)?\s*(million|billion|万|亿)\b",
    re.I,
)
RE_TRIAL = re.compile(r"\bNCT\d{8}\b|\bEUCTR-\d{4}-\d{6}-\d{2}\b", re.I)
RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
RE_PATENT = re.compile(r"\b(US|EP|CN)\d{5,}\b|\bWO\d{7,}\b", re.I)
RE_ISO = re.compile(r"\bISO\s?\d{4,5}(-\d+)?\b", re.I)
RE_STDNUM = re.compile(r"\bEN\s?\d{3,5}\b|\bASTM\s?[A-Z]?\d{2,5}\b", re.I)
RE_ID_ANY = re.compile(r"(注册号|登记号|批准文号|备案号)", re.I)


def _is_redline(text: str) -> bool:
    t = text or ""
    if RE_DATE.search(t):
        return True
    if RE_MONEY.search(t):
        return True
    if RE_TRIAL.search(t):
        return True
    if RE_DOI.search(t):
        return True
    if RE_PATENT.search(t):
        return True
    if RE_ISO.search(t):
        return True
    if RE_STDNUM.search(t):
        return True
    if RE_ID_ANY.search(t):
        return True
    return False


def _scrub_claims_to_hints(claims: List[Any], hints: List[Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    把明显带编号/金额的句子从 claims 挪到 evidence_hints，并记录到 moved_facts。
    不再额外注水模板；只做清洗和重分类。
    """
    new_claims: List[str] = []
    new_hints: List[str] = [str(x).strip() for x in (hints or []) if str(x).strip()]
    moved_facts: List[str] = []

    for c in claims or []:
        s = _norm_str(c)
        if not s:
            continue
        if _is_redline(s):
            new_hints.append(s)
            moved_facts.append(s)
        else:
            new_claims.append(s)

    return _uniq_cut(new_claims, MAX_LIST_LEN), _uniq_cut(new_hints, MAX_LIST_LEN), _uniq_cut(moved_facts, MAX_LIST_LEN)


def _to_bullets(answer: str) -> Tuple[str, int]:
    """
    把任意文本整理成 1. 2. 3. 形式的分点；不过度强制数量，只要 >=1 即可。
    修复双重编号问题：无论是按行还是按句拆分，都会先去掉已有编号/符号再统一加“1. 2. 3.”。
    """
    raw = (answer or "").strip()
    if not raw:
        return "", 0

    text = raw.replace("\r\n", "\n").strip()

    def _normalize_item(line: str) -> str:
        s = line.strip()
        # 去掉 Markdown 项目符号
        s = re.sub(r"^\s*[-*•●▪]+\s*", "", s)
        # 去掉前导数字编号（1. / 1) / 1、 等）
        s = re.sub(r"^\s*\d+[\.\)．、]\s*", "", s)
        # 压缩空白
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        if not s:
            return ""

        # 丢掉只剩数字或“数字+空格”的内容（例如 "1"、"2"、"1 2" 等）
        s_nospace = s.replace(" ", "")
        if s_nospace.isdigit() and len(s_nospace) <= 4:
            return ""

        return s

    # 先按行拆分（利用 LLM 原始的换行结构）
    lines = [ln for ln in text.split("\n") if ln.strip()]
    items: List[str] = []
    for ln in lines:
        norm = _normalize_item(ln)
        if norm:
            items.append(norm)

    # 如果按行只有 0–1 条，说明可能是整段一坨 -> 再按句号/分号拆一轮
    if len(items) <= 1:
        chunks = re.split(r"[。；;.!?？]\s*", text)
        items = []
        for ch in chunks:
            norm = _normalize_item(ch)
            if norm:
                items.append(norm)

    if not items:
        # 实在拆不出有效内容，就保留原文
        return raw, 0

    # 控制条数上限，避免 answer 过长
    items = items[:8]

    numbered = [f"{i+1}. {seg}" for i, seg in enumerate(items)]
    return "\n".join(numbered), len(items)

def _calibrate_conf(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.65
    if v < CONF_MIN:
        v = CONF_MIN
    if v > CONF_MAX:
        v = CONF_MAX
    return round(v, 2)


def _normalize_candidate_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    out: Dict[str, Any] = {}

    # ❗ 保留原始换行，不再用 _norm_str 压缩，避免破坏 LLM 已经分好的行
    raw_answer = obj.get("answer", "")

    # ====== 新增：专门处理 list / dict 形式的 answer，避免 json.dumps 成一坨 ======
    if isinstance(raw_answer, list):
        # LLM 有时会给 ["1\n2. ...", "3\n4. ..."] 这种
        pieces: List[str] = []
        for elem in raw_answer:
            if elem is None:
                continue
            # 再兜一层 list（极端情况）
            if isinstance(elem, list):
                for sub in elem:
                    s = str(sub or "").strip()
                    if s:
                        pieces.append(s)
            else:
                s = str(elem or "").strip()
                if s:
                    pieces.append(s)
        raw_answer = "\n".join(pieces)  # 变成多行文本，交给 _to_bullets 按行拆

    elif isinstance(raw_answer, dict):
        # 如果以后 LLM 返回 {"bullets":[...]} 之类，优先拉出里面的 list
        for key in ("bullets", "points", "items"):
            val = raw_answer.get(key)
            if isinstance(val, list):
                pieces = [str(x or "").strip() for x in val if str(x or "").strip()]
                raw_answer = "\n".join(pieces)
                break
        else:
            # 实在没有结构化 list，再兜底 dump 成字符串
            raw_answer = json.dumps(raw_answer, ensure_ascii=False)

    # 其他情况：本来就是字符串/数字，直接转成字符串
    out["answer"] = str(raw_answer or "")

    raw_claims = obj.get("claims", [])
    raw_hints = obj.get("evidence_hints", [])
    raw_tags = obj.get("topic_tags", [])
    raw_gi = obj.get("general_insights", [])

    if isinstance(raw_claims, (str, int, float)):
        raw_claims = [raw_claims]
    if isinstance(raw_hints, (str, int, float)):
        raw_hints = [raw_hints]
    if isinstance(raw_tags, (str, int, float)):
        raw_tags = [raw_tags]
    if isinstance(raw_gi, (str, int, float)):
        raw_gi = [raw_gi]

    out["claims"] = [str(x).strip() for x in (raw_claims or []) if str(x).strip()]
    out["evidence_hints"] = [str(x).strip() for x in (raw_hints or []) if str(x).strip()]
    out["topic_tags"] = [str(x).strip() for x in (raw_tags or []) if str(x).strip()]
    out["general_insights"] = [str(x).strip() for x in (raw_gi or []) if str(x).strip()]
    try:
        out["confidence"] = float(obj.get("confidence", 0.65))
    except Exception:
        out["confidence"] = 0.65
    out["caveats"] = _norm_str(obj.get("caveats", ""))
    return out


def _validate_candidate_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("answer"), str) or not obj.get("answer").strip():
        return False
    if not isinstance(obj.get("claims"), list):
        return False
    if not isinstance(obj.get("evidence_hints"), list):
        return False
    if not isinstance(obj.get("topic_tags"), list):
        return False
    if not isinstance(obj.get("general_insights"), list):
        return False
    return True


def _build_topic_tags(tags: List[Any], dimension: str, answer: str) -> List[str]:
    base = [str(t).lower().strip() for t in (tags or []) if str(t).strip()]
    extra: List[str] = []

    dim_token = dimension.lower().strip()
    if dim_token:
        extra.append(dim_token)

    words = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,8}", answer)
    freq: Dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if len(wl) < 2:
            continue
        freq[wl] = freq.get(wl, 0) + 1
    common = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]
    extra.extend([w for w, _ in common])

    return _uniq_cut(base + extra, MAX_LIST_LEN)


def _quick_score(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    给后续选优一个简单分数：
    - 分点条数
    - claims 数量
    - confidence
    （general_insights 目前只在下游使用，这里不额外打分，避免逻辑过重）
    """
    bullets = len([ln for ln in str(cand.get("answer", "")).splitlines() if ln.strip()])
    claims_n = len(cand.get("claims") or [])
    conf = float(cand.get("confidence", 0.65))

    score = 0.0
    if bullets >= 3:
        score += 0.25
    if 3 <= bullets <= 8:
        score += 0.20
    if claims_n >= 2:
        score += 0.25
    if claims_n >= 4:
        score += 0.10
    score += max(0.0, min(0.20, (conf - CONF_MIN) / (CONF_MAX - CONF_MIN + 1e-6) * 0.20))

    cand["quick_score"] = round(score, 3)
    return cand


def _finalize_candidate(
    obj: Dict[str, Any],
    provider: str,
    model: str,
    variant_id: str,
    sample_id: int,
    dimension: str,
) -> Dict[str, Any]:
    base = _normalize_candidate_obj(obj)
    answer_bullets, n_bullets = _to_bullets(base["answer"])
    base["answer"] = answer_bullets

    new_claims, new_hints, moved_facts = _scrub_claims_to_hints(base.get("claims", []), base.get("evidence_hints", []))
    base["claims"] = new_claims
    base["evidence_hints"] = new_hints
    base["facts_redlined"] = moved_facts

    # general_insights 只做去重+截断，不做红线搬移（允许包含“通常需要 NCT 编号”这类行业通识）
    base["general_insights"] = _uniq_cut(base.get("general_insights", []), MAX_LIST_LEN)

    base["topic_tags"] = _build_topic_tags(base.get("topic_tags", []), dimension, base["answer"])
    base["confidence"] = _calibrate_conf(base.get("confidence", 0.65))

    if not base.get("caveats"):
        base["caveats"] = "结论需结合原始提案全文与支撑材料进一步核查。"

    base["provider"] = provider
    base["model"] = model
    base["variant_id"] = variant_id
    base["sample_id"] = int(sample_id)
    base["generated_at"] = now_str()

    base["diag"] = {
        "bullet_count": n_bullets,
        "claims_count": len(base["claims"]),
        "hints_count": len(base["evidence_hints"]),
        "general_insights_count": len(base["general_insights"]),
    }

    base = _quick_score(base)
    return base


def _simhash_key(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s[:256]


def dedup_nearby(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        key = _simhash_key(c.get("answer", ""))
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept


# ========== 问答主逻辑（batch / single） ==========
def ask_model_batch(
    provider: str,
    client,
    model: str,
    dimension: str,
    q_list: List[str],
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    对支持 batch 的模型：同一维度下多题一起问；
    若解析失败或数组长度不对，由上层改走逐题兜底。
    返回：长度 == len(q_list) 的 candidate 列表。
    """
    prompt = build_batch_prompt(dimension, q_list, proposal_context, reg_hints, variant_id)
    try:
        txt = _chat_completion_json(
            client,
            model,
            SYSTEM_CN,
            prompt,
            max_tokens=max_tokens,
            temperature=TEMP_BY_VARIANT.get(variant_id, 0.3),
            force_json=provider_caps(provider)["json_mode"],
        )
    except Exception:
        return []

    low = (txt or "").strip().lower()
    if any(k in low for k in ("incorrect api key", "invalid api key", "rate limit", "quota", "access denied")):
        return []

    obj = _safe_parse_json_plus(txt)
    if not isinstance(obj, dict):
        return []
    arr = obj.get("answers")
    if not isinstance(arr, list) or len(arr) != len(q_list):
        return []

    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(arr, 1):
        cand_norm = _normalize_candidate_obj(it)
        if not _validate_candidate_dict(cand_norm):
            out.append({})
            continue
        finalized = _finalize_candidate(
            cand_norm,
            provider=provider,
            model=model,
            variant_id=variant_id,
            sample_id=idx,
            dimension=dimension,
        )
        out.append(finalized)
    return out


def ask_model_single(
    provider: str,
    client,
    model: str,
    dimension: str,
    question: str,
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = build_single_prompt(dimension, question, proposal_context, reg_hints, variant_id)
    try:
        txt = _chat_completion_json(
            client,
            model,
            SYSTEM_CN,
            prompt,
            max_tokens=max_tokens,
            temperature=TEMP_BY_VARIANT.get(variant_id, 0.3),
            force_json=provider_caps(provider)["json_mode"],
        )
    except Exception:
        return {"error": True, "answer": ""}

    obj = _safe_parse_json_plus(txt)
    if not isinstance(obj, dict):
        return {"error": True, "answer": ""}

    cand_norm = _normalize_candidate_obj(obj)
    if not _validate_candidate_dict(cand_norm):
        return {"error": True, "answer": ""}

    return _finalize_candidate(
        cand_norm,
        provider=provider,
        model=model,
        variant_id=variant_id,
        sample_id=1,
        dimension=dimension,
    )


def refine_candidate(
    candidate: Dict[str, Any],
    client,
    model: str,
    dimension: str,
    proposal_context: str,
    provider: str,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    try:
        rp = build_refine_prompt(candidate, proposal_context, dimension)
        txt = _chat_completion_json(
            client,
            model,
            SYSTEM_CN,
            rp,
            max_tokens=max_tokens,
            temperature=0.2,
            force_json=provider_caps(provider)["json_mode"],
        )
        obj = _safe_parse_json_plus(txt)
        if isinstance(obj, dict):
            cand_norm = _normalize_candidate_obj(obj)
            if not _validate_candidate_dict(cand_norm):
                return candidate
            return _finalize_candidate(
                cand_norm,
                provider=candidate.get("provider", provider),
                model=candidate.get("model", model),
                variant_id=candidate.get("variant_id", "default"),
                sample_id=candidate.get("sample_id", 1),
                dimension=dimension,
            )
        return candidate
    except Exception:
        return candidate


# ========== 维度级问答 ==========
def print_dim_banner(provider_name: str, dim: str, total: int, mode: str):
    bar = "=" * 12
    print(f"\n{bar} [{provider_name}] 维度：{dim} | 题目数：{total} | 模式：{mode} {bar}", flush=True)


def print_q_progress(provider_name: str, dim: str, idx: int, total: int, qtext: str):
    preview = qtext.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    print(f"[{provider_name}] ({dim}) Q{idx}/{total} ▶ {preview}", flush=True)


def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield i, lst[i : i + n]


def answer_dimension(
    provider: str,
    client,
    model_name: str,
    dim: str,
    q_list: List[str],
    proposal_context: str,
    reg_hints: List[str],
    refine: bool,
    group_size: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """
    返回：list[
      {
        "dimension": dim,
        "q_index": idx,
        "question": q,
        "candidates": [candidate_obj, ...]
      }, ...
    ]
    """
    out_items: List[Dict[str, Any]] = []
    provider_name = "ChatGPT" if provider == "openai" else "DeepSeek"

    if not q_list:
        return out_items

    caps = provider_caps(provider)
    supports_batch = bool(caps.get("batch_ok", True))

    mode = "批量+变体" if supports_batch else "逐题+变体"
    print_dim_banner(provider_name, dim, len(q_list), mode)

    # 不支持 batch：逐题 × 多变体
    if not supports_batch:
        for idx, q in enumerate(q_list, 1):
            print_q_progress(provider_name, dim, idx, len(q_list), q)
            raw_cands: List[Dict[str, Any]] = []
            for v in VARIANTS:
                cand = ask_model_single(
                    provider=provider,
                    client=client,
                    model=model_name,
                    dimension=dim,
                    question=q,
                    proposal_context=proposal_context,
                    reg_hints=reg_hints,
                    variant_id=v,
                    max_tokens=min(900, max_tokens),
                )
                if isinstance(cand, dict) and not cand.get("error") and cand.get("answer", "").strip():
                    raw_cands.append(cand)
            cands = dedup_nearby(raw_cands)
            if refine and cands:
                cands = [
                    refine_candidate(
                        c,
                        client=client,
                        model=model_name,
                        dimension=dim,
                        proposal_context=proposal_context,
                        provider=provider,
                        max_tokens=min(700, max_tokens),
                    )
                    for c in cands
                ]
            for c in cands:
                c["dimension"] = dim
                c["q_index"] = idx

            out_items.append(
                {
                    "dimension": dim,
                    "q_index": idx,
                    "question": q,
                    "candidates": cands,
                }
            )
        return out_items

    # 支持 batch：按 group_size 分批 + 变体
    group_size = max(1, min(int(group_size), 4))
    for start_idx, sub_qs in chunked(q_list, group_size):
        batch_tag = f"{start_idx+1}-{start_idx+len(sub_qs)}"
        per_variant_results: Dict[str, List[Dict[str, Any]]] = {}
        batch_failed = False

        for v in VARIANTS:
            t0 = time.time()
            arr = ask_model_batch(
                provider=provider,
                client=client,
                model=model_name,
                dimension=dim,
                q_list=sub_qs,
                proposal_context=proposal_context,
                reg_hints=reg_hints,
                variant_id=v,
                max_tokens=max_tokens,
            )
            ok = bool(arr) and len(arr) == len(sub_qs)
            print(
                f"[{provider_name}] ({dim}) 小批 {batch_tag} · {v:<13} 返回 {len(arr)}/{len(sub_qs)} 条，用时 {time.time()-t0:.1f}s（{'OK' if ok else 'FAIL→逐题'}）"
            )
            if not ok:
                batch_failed = True
                break
            per_variant_results[v] = arr

        if not batch_failed:
            # 合成每题的 candidates
            for j, q in enumerate(sub_qs, 1):
                global_idx = start_idx + j
                raw_cands = [per_variant_results[v][j - 1] for v in VARIANTS]
                cands = [c for c in raw_cands if isinstance(c, dict) and c.get("answer", "").strip()]
                cands = dedup_nearby(cands)
                if refine and cands:
                    cands = [
                        refine_candidate(
                            c,
                            client=client,
                            model=model_name,
                            dimension=dim,
                            proposal_context=proposal_context,
                            provider=provider,
                            max_tokens=min(700, max_tokens),
                        )
                        for c in cands
                    ]
                for c in cands:
                    c["dimension"] = dim
                    c["q_index"] = global_idx
                out_items.append(
                    {
                        "dimension": dim,
                        "q_index": global_idx,
                        "question": q,
                        "candidates": cands,
                    }
                )
            continue

        # 批量失败：这一小批改逐题
        for j, q in enumerate(sub_qs, 1):
            global_idx = start_idx + j
            print_q_progress(provider_name, dim, global_idx, len(q_list), q)
            raw_cands: List[Dict[str, Any]] = []
            for v in VARIANTS:
                cand = ask_model_single(
                    provider=provider,
                    client=client,
                    model=model_name,
                    dimension=dim,
                    question=q,
                    proposal_context=proposal_context,
                    reg_hints=reg_hints,
                    variant_id=v,
                    max_tokens=min(900, max_tokens),
                )
                if isinstance(cand, dict) and not cand.get("error") and cand.get("answer", "").strip():
                    raw_cands.append(cand)
            cands = dedup_nearby(raw_cands)
            if refine and cands:
                cands = [
                    refine_candidate(
                        c,
                        client=client,
                        model=model_name,
                        dimension=dim,
                        proposal_context=proposal_context,
                        provider=provider,
                        max_tokens=min(700, max_tokens),
                    )
                    for c in cands
                ]
            for c in cands:
                c["dimension"] = dim
                c["q_index"] = global_idx
            out_items.append(
                {
                    "dimension": dim,
                    "q_index": global_idx,
                    "question": q,
                    "candidates": cands,
                }
            )

    return out_items


# ========== 多模型合并 ==========
def merge_two_models(chatgpt_items: List[Dict[str, Any]], deepseek_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_of(x: Dict[str, Any]):
        return (x["dimension"], x["q_index"], x["question"])

    pool: Dict[Any, Dict[str, Any]] = {}
    for it in chatgpt_items:
        pool[key_of(it)] = {
            "dimension": it["dimension"],
            "q_index": it["q_index"],
            "question": it["question"],
            "candidates": list(it.get("candidates", [])),
        }
    for it in deepseek_items:
        k = key_of(it)
        if k in pool:
            pool[k]["candidates"].extend(it.get("candidates", []))
        else:
            pool[k] = {
                "dimension": it["dimension"],
                "q_index": it["q_index"],
                "question": it["question"],
                "candidates": list(it.get("candidates", [])),
            }

    items = list(pool.values())
    items.sort(
        key=lambda x: (
            DIM_ORDER.index(x["dimension"]) if x["dimension"] in DIM_ORDER else 99,
            x["q_index"],
        )
    )
    return items


# ========== CLI ==========
def parse_args():
    ap = argparse.ArgumentParser(
        description="Proposal-aware Dual-LLM Answering (ChatGPT + DeepSeek, no web search, dimension-facts aware)"
    )
    ap.add_argument(
        "--proposal_id",
        type=str,
        default="",
        help="指定提案 ID；不填则自动检测 data/extracted 最近目录名",
    )
    ap.add_argument(
        "--qs_file",
        type=str,
        default=str(CONFIG_QS_DEFAULT),
        help="问题集 JSON 路径（通常由 generate_questions.py 生成）",
    )
    ap.add_argument(
        "--dim-file",
        type=str,
        default="",
        help="维度 JSON 路径；默认使用 data/extracted/{pid}/dimensions_v2.json",
    )
    ap.add_argument(
        "--refine",
        type=int,
        default=1,
        help="是否进行轻量自我复核（0/1）",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=2200,
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=3,
        help="每批问题数（支持 batch 的模型才生效，建议 2–4）",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        try:
            random.seed(int(args.seed))
        except Exception:
            pass

    pid = args.proposal_id.strip() or detect_latest_pid()
    if pid == "unknown":
        print("⚠️ 未检测到 data/extracted 下的提案目录，将使用占位 pid=unknown。")
    print(f"🧩 proposal_id = {pid}")

    qs_path = Path(args.qs_file)
    if not qs_path.exists():
        raise FileNotFoundError(f"未找到问题集：{qs_path}")
    qs_cfg = read_json(qs_path)

    missing = [d for d in DIM_ORDER if not get_q_list(qs_cfg.get(d, []))]
    if missing:
        raise RuntimeError(
            f"问题集缺少维度或该维度题目为空：{missing}（请检查 {qs_path} 或 generate_questions.py 输出）"
        )

    # ========= 维度上下文：固定读取 extracted/{pid}/dimensions_v2.json =========
    if args.dim_file.strip():
        dim_file = Path(args.dim_file.strip())
    else:
        dim_file = EXTRACTED_DIR / pid / "dimensions_v2.json"

    if not dim_file.exists():
        raise FileNotFoundError(
            f"未找到维度文件：{dim_file} ；请先运行 build_dimensions_from_facts.py 生成 dimensions_v2.json"
        )

    print(f"📄 使用维度上下文文件：{dim_file}")
    dim_context_map = load_dimension_context(pid, dim_file)

    reg_hints_map: Dict[str, List[str]] = {}
    for dim in DIM_ORDER:
        reg_hints_map[dim] = _load_reg_hints(qs_cfg, dim, limit=8)

    oa_client, oa_model = init_openai()
    ds_client, ds_model = init_deepseek()
    if (oa_client is None or oa_model is None) and (ds_client is None or ds_model is None):
        raise RuntimeError("未检测到可用模型：请在 .env 配置 OPENAI_API_KEY 与/或 DEEPSEEK_API_KEY。")

    dims = DIM_ORDER[:]
    chatgpt_items_all: List[Dict[str, Any]] = []
    deepseek_items_all: List[Dict[str, Any]] = []

    # ChatGPT
    if oa_client and oa_model:
        print("🧠 ChatGPT 答题中 ...")
        _write_llm_progress(0, len(dims), pid)
        for i, dim in enumerate(dims):
            q_list = get_q_list(qs_cfg.get(dim, []))
            ctx = dim_context_map.get(dim, "")
            reg_hints = reg_hints_map.get(dim, [])
            items = answer_dimension(
                provider="openai",
                client=oa_client,
                model_name=oa_model,
                dim=dim,
                q_list=q_list,
                proposal_context=ctx,
                reg_hints=reg_hints,
                refine=bool(args.refine),
                group_size=int(args.group_size),
                max_tokens=int(args.max_tokens),
            )
            chatgpt_items_all.extend(items)
            _write_llm_progress(i + 1, len(dims), pid)

        out_path = OUT_REFINED / pid / "chatgpt_raw.json"
        write_json(
            out_path,
            {
                "meta": {
                    "model": oa_model,
                    "provider": "openai",
                    "generated_at": now_str(),
                    "pid": pid,
                },
                "items": chatgpt_items_all,
            },
        )
        print(f"✅ ChatGPT 结果 -> {out_path}")
    else:
        print("⚠️ 跳过 ChatGPT：未配置 OPENAI_API_KEY。")

    # DeepSeek
    if ds_client and ds_model:
        print("🧠 DeepSeek 答题中 ...")
        for dim in dims:
            q_list = get_q_list(qs_cfg.get(dim, []))
            ctx = dim_context_map.get(dim, "")
            reg_hints = reg_hints_map.get(dim, [])
            items = answer_dimension(
                provider="deepseek",
                client=ds_client,
                model_name=ds_model,
                dim=dim,
                q_list=q_list,
                proposal_context=ctx,
                reg_hints=reg_hints,
                refine=bool(args.refine),
                group_size=int(args.group_size),
                max_tokens=int(args.max_tokens),
            )
            deepseek_items_all.extend(items)

        out_path = OUT_REFINED / pid / "deepseek_raw.json"
        write_json(
            out_path,
            {
                "meta": {
                    "model": ds_model,
                    "provider": "deepseek",
                    "generated_at": now_str(),
                    "pid": pid,
                },
                "items": deepseek_items_all,
            },
        )
        print(f"✅ DeepSeek 结果 -> {out_path}")
    else:
        print("⚠️ 跳过 DeepSeek：未配置 DEEPSEEK_API_KEY。")

    merged_items = merge_two_models(chatgpt_items_all, deepseek_items_all)
    merged = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "schema": "refined_items.v2.proposal_aware_with_general_insights",
            "args": {
                "refine": bool(args.refine),
                "max_tokens": int(args.max_tokens),
                "group_size": int(args.group_size),
            },
            "models": {
                "chatgpt": {"model": oa_model, "provider": "openai"} if chatgpt_items_all else None,
                "deepseek": {"model": ds_model, "provider": "deepseek"} if deepseek_items_all else None,
            },
        },
        "items": merged_items,
    }
    out_path = OUT_REFINED / pid / "all_refined_items.json"
    write_json(out_path, merged)
    print(f"📦 合并结果 -> {out_path}")
    print("🎯 完成。")


if __name__ == "__main__":  # pragma: no cover
    main()
