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

    return DomainConfig(
        domain_name=domain,
        dimensions=dimensions,
        fact_types=fact_types,
        fallback_dimension=data.get("fallback_dimension", "feasibility"),
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
    """Build a default config matching the original codebase's hardcoded values."""
    dimensions = [
        DimensionConfig(
            name="team", label_zh="团队与治理", label_en="Team & Governance",
            description="Team composition, individual backgrounds, institutional affiliations, roles, collaboration",
            focus_zh="关注团队组成、核心负责人履历、跨机构合作网络、以及团队稳定性和时间投入。",
            focus_en="Team composition, leadership experience, collaboration network, stability",
            min_questions=6, max_questions=9, weight=1.0,
            aspects=[
                AspectConfig("leadership_experience", "项目负责人的领导经验与往期重大项目执行记录"),
                AspectConfig("domain_expertise", "团队在目标领域的专业深度"),
                AspectConfig("collaboration_network", "合作机构、产业伙伴网络及互补性"),
                AspectConfig("governance_and_decision_making", "项目治理结构、决策机制、质量控制"),
                AspectConfig("team_capacity_and_bandwidth", "团队当前人力负荷与资源投入能力"),
            ],
        ),
        DimensionConfig(
            name="objectives", label_zh="项目目标", label_en="Objectives",
            description="Overall goals, milestones, KPIs, sub-project targets",
            focus_zh="关注项目总体目标的清晰度、分阶段里程碑、可量化指标和可实现性。",
            focus_en="Goal clarity, milestones, measurable metrics, achievability",
            min_questions=6, max_questions=9, weight=1.0,
            aspects=[
                AspectConfig("overall_goal_clarity", "总体目标是否明确、聚焦"),
                AspectConfig("milestones_and_timeline", "各阶段里程碑的设计是否合理可执行"),
                AspectConfig("outcome_and_success_metrics", "是否有清晰可量化的成功指标"),
                AspectConfig("scope_and_prioritization", "项目范围和优先级排序"),
                AspectConfig("realism_and_ambition_balance", "目标在雄心和可行性之间的平衡"),
            ],
        ),
        DimensionConfig(
            name="strategy", label_zh="实施路径与战略", label_en="Strategy",
            description="Technical approach, development path, market strategy, partnerships, operations",
            focus_zh="关注技术路线设计、市场与商业化路径、合作伙伴策略和资源利用方式。",
            focus_en="Technical approach, market entry, partnership strategy, resource utilization",
            min_questions=7, max_questions=10, weight=1.0,
            aspects=[
                AspectConfig("technical_strategy", "技术路线的合理性与替代方案"),
                AspectConfig("commercialization_and_market_entry", "商业化模式、定价与市场进入路径"),
                AspectConfig("partnership_and_business_model", "合作模式（授权、共同开发、服务等）"),
                AspectConfig("data_and_evidence_strategy", "数据利用策略及隐私合规安排"),
                AspectConfig("scaling_and_globalization", "从验证到大规模推广的扩展策略"),
            ],
        ),
        DimensionConfig(
            name="innovation", label_zh="技术与产品创新", label_en="Innovation",
            description="Novelty, differentiation, IP, evidence strength, extensibility",
            focus_zh="关注技术/产品相对现有方案的创新性、差异化优势、知识产权布局和证据支撑。",
            focus_en="Novelty vs state of art, differentiation, IP protection, evidence strength",
            min_questions=6, max_questions=9, weight=1.10,
            aspects=[
                AspectConfig("novelty_vs_state_of_art", "相对当前前沿方案的真正创新点"),
                AspectConfig("differentiation_and_competitive_edge", "与替代方案相比的明确优势"),
                AspectConfig("ip_and_protection", "专利/数据资产的保护布局"),
                AspectConfig("evidence_strength_for_innovation", "创新点的实验/验证证据强度"),
                AspectConfig("platform_and_extensibility", "是否构成可拓展的平台或仅单点创新"),
                AspectConfig("risk_of_obsolescence", "技术在3-5年内被替代的风险评估"),
            ],
        ),
        DimensionConfig(
            name="feasibility", label_zh="资源与可行性", label_en="Feasibility",
            description="Resources, budget, implementation path, risks, timeline",
            focus_zh="关注资源与基础设施、资金与预算、实施路径、关键风险和应对措施、落地可行性。",
            focus_en="Resources, funding, implementation path, risk management, timeline feasibility",
            min_questions=7, max_questions=10, weight=1.20,
            aspects=[
                AspectConfig("resources_and_infrastructure", "资源平台是否充足且可长期稳定使用"),
                AspectConfig("funding_and_budget_planning", "资金来源多样性、预算分配合理性"),
                AspectConfig("operational_execution_plan", "实施路径是否具体清晰"),
                AspectConfig("risk_management", "对各类风险的识别与缓解措施"),
                AspectConfig("implementation_barriers", "在实际场景中落地的阻力"),
                AspectConfig("timeline_and_resource_alignment", "时间表与资源投入是否匹配"),
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
    )


def clear_cache() -> None:
    """Clear the configuration cache."""
    _config_cache.clear()
