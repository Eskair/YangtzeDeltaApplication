# -*- coding: utf-8 -*-
"""
Domain configuration module.

Provides a centralized, configurable source for:
  - Evaluation dimensions (names, descriptions, aspect configs)
  - Fact types (names, descriptions, dimension mappings)
  - Domain-specific labels and prompts
  - Scoring weights and parameters

All pipeline stages should import from here instead of hardcoding
dimension names, fact types, or domain-specific terminology.

Usage:
    from src.config import get_config
    cfg = get_config()           # uses REVIEW_DOMAIN env var, defaults to "default"
    cfg = get_config("biomedical")  # explicit domain

    cfg.dimension_names          # ["team", "objectives", ...]
    cfg.valid_types              # ["team_member", "org_structure", ...]
    cfg.get_dimension("team")    # DimensionConfig object
    cfg.get_type_to_dims_map()   # {"team_member": ["team"], ...}
    cfg.material_domain_zh       # 与 extract / build_dimensions 共用的中文材料语域说明（YAML 可覆盖）
    material_domain_zh_for_prompts()  # 供流水线脚本注入提示词，避免多处硬编码漂移
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import yaml


@dataclass
class AspectConfig:
    """A sub-aspect within a dimension (used for question generation)."""
    id: str
    desc_zh: str
    desc_en: str = ""


@dataclass
class DimensionConfig:
    """Configuration for a single evaluation dimension."""
    name: str
    label_zh: str
    label_en: str
    description: str
    focus_zh: str
    focus_en: str
    aspects: List[AspectConfig] = field(default_factory=list)
    min_questions: int = 6
    max_questions: int = 9
    weight: float = 1.0


@dataclass
class FactTypeConfig:
    """Configuration for a single fact type."""
    name: str
    description: str
    mapped_dimensions: List[str] = field(default_factory=list)


@dataclass
class DomainConfig:
    """Complete domain configuration."""
    domain_name: str
    dimensions: List[DimensionConfig]
    fact_types: List[FactTypeConfig]
    fallback_dimension: str = "feasibility"
    # 与 extract_facts / build_dimensions 等共用的「材料语域」中文说明（default.yaml 的 material_domain_zh）
    material_domain_zh: str = ""

    @property
    def dimension_names(self) -> List[str]:
        return [d.name for d in self.dimensions]

    @property
    def valid_dimensions(self) -> List[str]:
        return self.dimension_names

    @property
    def valid_types(self) -> List[str]:
        return [t.name for t in self.fact_types]

    @property
    def dimension_labels_zh(self) -> Dict[str, str]:
        return {d.name: d.label_zh for d in self.dimensions}

    @property
    def dimension_weights(self) -> Dict[str, float]:
        return {d.name: d.weight for d in self.dimensions}

    def get_dimension(self, name: str) -> Optional[DimensionConfig]:
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    def get_type_to_dims_map(self) -> Dict[str, List[str]]:
        """Map each fact type to its default dimensions."""
        return {t.name: t.mapped_dimensions for t in self.fact_types}

    def get_aspects_for_dimension(self, dim_name: str) -> List[Dict[str, str]]:
        d = self.get_dimension(dim_name)
        if not d:
            return []
        return [{"id": a.id, "desc_zh": a.desc_zh} for a in d.aspects]

    def get_dimension_config_dict(self, dim_name: str) -> Dict[str, Any]:
        """Return a dict compatible with the existing DIMENSION_CONFIG structure."""
        d = self.get_dimension(dim_name)
        if not d:
            return {}
        return {
            "min_q": d.min_questions,
            "max_q": d.max_questions,
            "focus_zh": d.focus_zh,
            "aspects": [{"id": a.id, "desc_zh": a.desc_zh} for a in d.aspects],
        }


_CONFIG_DIR = Path(__file__).resolve().parent
_config_cache: Dict[str, DomainConfig] = {}

# 当 YAML 未配置 material_domain_zh 时的兜底（与 default.yaml 建议内容保持一致）
DEFAULT_MATERIAL_DOMAIN_ZH = (
    "各类商业项目评审材料（投融资计划书、可行性研究、招标与响应文件、公司介绍、备忘录等），行业不限。\n"
    "中文或以中文为主、中英混排较为常见；若为纯外文、临床方案、科研课题等，仅以原文为准，勿套用未出现的商业要素。"
)


def _load_domain_yaml(domain: str) -> Dict[str, Any]:
    """Load a domain YAML config file."""
    path = _CONFIG_DIR / f"{domain}.yaml"
    if not path.exists():
        path = _CONFIG_DIR / "default.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_config(domain: str, data: Dict[str, Any]) -> DomainConfig:
    """Parse raw YAML data into a DomainConfig."""
    dimensions = []
    for d in data.get("dimensions", []):
        aspects = []
        for a in d.get("aspects", []):
            aspects.append(AspectConfig(
                id=a.get("id", ""),
                desc_zh=a.get("desc_zh", ""),
                desc_en=a.get("desc_en", ""),
            ))
        dimensions.append(DimensionConfig(
            name=d["name"],
            label_zh=d.get("label_zh", d["name"]),
            label_en=d.get("label_en", d["name"]),
            description=d.get("description", ""),
            focus_zh=d.get("focus_zh", ""),
            focus_en=d.get("focus_en", ""),
            aspects=aspects,
            min_questions=d.get("min_questions", 6),
            max_questions=d.get("max_questions", 9),
            weight=d.get("weight", 1.0),
        ))

    fact_types = []
    for t in data.get("fact_types", []):
        fact_types.append(FactTypeConfig(
            name=t["name"],
            description=t.get("description", ""),
            mapped_dimensions=t.get("mapped_dimensions", []),
        ))

    raw_md = (data.get("material_domain_zh") or "").strip()
    if not raw_md:
        raw_md = DEFAULT_MATERIAL_DOMAIN_ZH.strip()

    return DomainConfig(
        domain_name=domain,
        dimensions=dimensions,
        fact_types=fact_types,
        fallback_dimension=data.get("fallback_dimension", "feasibility"),
        material_domain_zh=raw_md,
    )


def get_config(domain: Optional[str] = None) -> DomainConfig:
    """
    Get the domain configuration. Uses REVIEW_DOMAIN env var if domain not specified.
    Falls back to "default" if the env var is not set.
    """
    if domain is None:
        domain = os.getenv("REVIEW_DOMAIN", "default").strip().lower()

    if domain in _config_cache:
        return _config_cache[domain]

    data = _load_domain_yaml(domain)
    if not data:
        # Fall back to default
        data = _load_domain_yaml("default")

    if not data:
        # Ultimate fallback: build from hardcoded defaults
        config = _build_hardcoded_default()
    else:
        config = _parse_config(domain, data)

    _config_cache[domain] = config
    return config


def _build_hardcoded_default() -> DomainConfig:
    """YAML 缺失时的兜底：与 default.yaml 一致的通用评审语义骨架。"""
    dimensions = [
        DimensionConfig(
            name="team", label_zh="团队与治理", label_en="Team & Governance",
            description="Team composition, individual backgrounds, institutional affiliations, roles, collaboration",
            focus_zh="通用评审骨架：执行主体是否清晰；关键角色能力与分工；治理与外部协作是否支撑披露目标（以材料为准，不预设行业）。",
            focus_en="Team composition, leadership experience, collaboration network, stability",
            min_questions=6, max_questions=9, weight=1.0,
            aspects=[
                AspectConfig("leadership_experience", "牵头人/负责人对同类或同规模任务的执行记录与可核验证据（材料已披露部分）"),
                AspectConfig("domain_expertise", "团队能力与任务范围的匹配度：知识深度、资质与关键技能覆盖（材料已披露部分）"),
                AspectConfig("collaboration_network", "外部协作与关键依赖：合作方角色、互补性与集中度风险（材料已披露部分）"),
                AspectConfig("governance_and_decision_making", "治理与决策：分工授权、决策链、质量与合规内控安排（材料已披露部分）"),
                AspectConfig("team_capacity_and_bandwidth", "投入与承载：人力/时间/管理带宽与阶段目标的匹配（材料已披露部分）"),
            ],
        ),
        DimensionConfig(
            name="objectives", label_zh="项目目标", label_en="Objectives",
            description="Overall goals, milestones, KPIs, sub-project targets",
            focus_zh="通用评审骨架：目标是否可检验；阶段结果与成功判据；范围边界与优先级；与约束条件的自洽性（以材料为准）。",
            focus_en="Goal clarity, milestones, measurable metrics, achievability",
            min_questions=6, max_questions=9, weight=1.0,
            aspects=[
                AspectConfig("overall_goal_clarity", "总体目标陈述是否明确、可界定成功/失败（材料已披露部分）"),
                AspectConfig("milestones_and_timeline", "阶段划分与时间节点是否具体、可执行与可检查（材料已披露部分）"),
                AspectConfig("outcome_and_success_metrics", "成功判据与测度：指标、口径与证据来源是否交代清楚（材料已披露部分）"),
                AspectConfig("scope_and_prioritization", "范围边界与不做什么；资源约束下的优先级取舍（材料已披露部分）"),
                AspectConfig("realism_and_ambition_balance", "目标强度与可得资源、外部环境之间的一致性（材料已披露部分）"),
            ],
        ),
        DimensionConfig(
            name="strategy", label_zh="实施路径与战略", label_en="Strategy",
            description="Technical approach, development path, market strategy, partnerships, operations",
            focus_zh="通用评审骨架：主路径与备选；对外价值实现与关键接口；合作与资源编排；证据与合规边界（以材料为准，不预设单一业态）。",
            focus_en="Technical approach, market entry, partnership strategy, resource utilization",
            min_questions=7, max_questions=10, weight=1.0,
            aspects=[
                AspectConfig("technical_strategy", "主方案与备选方案：方法选择、关键假设与依赖（材料已披露部分）"),
                AspectConfig("commercialization_and_market_entry", "价值到达路径：客户/用户/采购或付费关系、渠道与交付形态（材料已披露部分；无则评信息缺口）"),
                AspectConfig("partnership_and_business_model", "合作与交易结构：权责、收益分配、排他与退出（材料已披露部分）"),
                AspectConfig("data_and_evidence_strategy", "关键结论所依赖的数据、来源、留存与合规/保密边界（材料已披露部分）"),
                AspectConfig("scaling_and_globalization", "扩张或跨区域/跨场景推广的前提、节奏与约束（材料已披露部分）"),
            ],
        ),
        DimensionConfig(
            name="innovation", label_zh="技术与产品创新", label_en="Innovation",
            description="Novelty, differentiation, IP, evidence strength, extensibility",
            focus_zh="通用评审骨架：相对基准的差异；可辩护优势；保护与证据链；可扩展边界与外部替代风险（以材料为准）。",
            focus_en="Novelty vs state of art, differentiation, IP protection, evidence strength",
            min_questions=6, max_questions=9, weight=1.10,
            aspects=[
                AspectConfig("novelty_vs_state_of_art", "相对行业/惯例或对标基准的实质性差异点（材料已披露部分）"),
                AspectConfig("differentiation_and_competitive_edge", "可辩护的差异化要点及与替代方案的关系（材料已披露部分）"),
                AspectConfig("ip_and_protection", "可保护知识与合规资产：权利、合同安排、排他或关键 know-how（材料已披露部分）"),
                AspectConfig("evidence_strength_for_innovation", "主张的支撑强度：验证设计、样本、独立性与可复核性（材料已披露部分）"),
                AspectConfig("platform_and_extensibility", "方案的复用边界、接口与生态位：是否具备可扩展结构（材料已披露部分）"),
                AspectConfig("risk_of_obsolescence", "外部变化下被替代、淘汰或规则变化的主要风险与窗口（材料已披露部分）"),
            ],
        ),
        DimensionConfig(
            name="feasibility", label_zh="资源与可行性", label_en="Feasibility",
            description="Resources, budget, implementation path, risks, timeline",
            focus_zh="通用评审骨架：资源与预算；执行编排；风险与缓解；关键障碍与时间—投入节拍（以材料为准）。",
            focus_en="Resources, funding, implementation path, risk management, timeline feasibility",
            min_questions=7, max_questions=10, weight=1.20,
            aspects=[
                AspectConfig("resources_and_infrastructure", "关键资源与基础设施的可得性、独占性与持续性（材料已披露部分）"),
                AspectConfig("funding_and_budget_planning", "资金与预算结构：来源、用途、集中度与缓冲（材料已披露部分）"),
                AspectConfig("operational_execution_plan", "从计划到落地的步骤、责任主体与关键路径（材料已披露部分）"),
                AspectConfig("risk_management", "主要风险与监测指标；缓解与预案是否对应（材料已披露部分）"),
                AspectConfig("implementation_barriers", "实施障碍：监管、供应链、组织、数据或外部依赖等（材料已披露部分）"),
                AspectConfig("timeline_and_resource_alignment", "时间表与投入节拍、里程碑与资源曲线是否自洽（材料已披露部分）"),
            ],
        ),
    ]

    fact_types = [
        FactTypeConfig("team_member", "Member identity, institution, role", ["team"]),
        FactTypeConfig("org_structure", "Team/org structure, division of labor", ["team"]),
        FactTypeConfig("collaboration", "Partnerships, cross-institutional cooperation", ["team", "strategy"]),
        FactTypeConfig("resource", "Lab, platform, data, partnership resources", ["feasibility"]),
        FactTypeConfig("pipeline", "Project/sub-project goals, technical approach", ["objectives", "strategy"]),
        FactTypeConfig("milestone", "Phase-specific tasks, timelines", ["objectives", "feasibility"]),
        FactTypeConfig("market", "Market size, CAGR, competition, customers", ["objectives", "strategy", "feasibility"]),
        FactTypeConfig("tech_route", "Technical/algorithmic/experimental approach", ["strategy", "feasibility"]),
        FactTypeConfig("product", "Product/platform form, business model", ["objectives", "strategy", "feasibility"]),
        FactTypeConfig("ip_asset", "Patents, proprietary technology, unique data", ["innovation", "feasibility"]),
        FactTypeConfig("evidence", "Experimental/validation data and conclusions", ["innovation", "feasibility"]),
        FactTypeConfig("budget_item", "Budget amounts or cost breakdown", ["feasibility"]),
        FactTypeConfig("funding_source", "Funding sources (VC, grants, self-funded)", ["strategy", "feasibility"]),
        FactTypeConfig("risk", "Technical/market/regulatory risks or uncertainties", ["feasibility"]),
        FactTypeConfig("mitigation", "Risk countermeasures or mitigation strategies", ["feasibility"]),
        FactTypeConfig("ai_model", "Specific AI model/algorithm/architecture", ["innovation", "strategy"]),
        FactTypeConfig("regulatory", "Regulatory pathway, approval strategy", ["strategy", "feasibility"]),
        FactTypeConfig("other", "Other relevant facts", []),
    ]

    return DomainConfig(
        domain_name="default",
        dimensions=dimensions,
        fact_types=fact_types,
        fallback_dimension="feasibility",
        material_domain_zh=DEFAULT_MATERIAL_DOMAIN_ZH.strip(),
    )


def material_domain_zh_for_prompts(domain: Optional[str] = None) -> str:
    """Shared Chinese material-domain blurb for LLM prompts (extract, build_dimensions, …)."""
    try:
        return get_config(domain).material_domain_zh.strip() or DEFAULT_MATERIAL_DOMAIN_ZH.strip()
    except Exception:
        return DEFAULT_MATERIAL_DOMAIN_ZH.strip()


def clear_cache() -> None:
    """Clear the configuration cache."""
    _config_cache.clear()
