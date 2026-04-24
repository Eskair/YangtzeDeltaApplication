# -*- coding: utf-8 -*-
"""
Post-processing authority / coverage lexicons by YAML profile.

Resolution order for profile name:
  1. cfg[\"authority_profile\"] (merged from src/data/config/postproc/config.json)
  2. env POSTPROC_AUTHORITY_PROFILE
  3. env REVIEW_DOMAIN: biomedical-like -> regulated_products
  4. default

Files: src/config/postproc_authority/<profile>.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

LEXICON_DIR = Path(__file__).resolve().parents[1] / "config" / "postproc_authority"

REGULATED_FROM_REVIEW_DOMAIN = frozenset(
    {
        "biomedical",
        "biomed",
        "clinical",
        "pharma",
        "regulated",
        "regulated_products",
        "medtech",
        "ivd",
    }
)

# 行政审批 / 政务中文材料（与 biomedical 互斥使用 REVIEW_DOMAIN 即可）
APPROVAL_FROM_REVIEW_DOMAIN = frozenset(
    {
        "approval",
        "administrative",
        "government",
        "gov_approval",
        "tender_government",
        "investment_approval",
        "infra_approval",
    }
)


def resolve_authority_profile_name(cfg: Dict[str, Any]) -> str:
    explicit = (cfg.get("authority_profile") or "").strip()
    if explicit:
        return explicit.lower()
    env_p = (os.getenv("POSTPROC_AUTHORITY_PROFILE") or "").strip()
    if env_p:
        return env_p.lower()
    dom = (os.getenv("REVIEW_DOMAIN") or "default").strip().lower()
    if dom in REGULATED_FROM_REVIEW_DOMAIN:
        return "regulated_products"
    return "default"


def _builtin_fallback_lexicon() -> Dict[str, Any]:
    """If YAML files are missing (packaging / tests)."""
    return {
        "profile_id": "default_fallback",
        "authority_tokens": [
            "iso",
            "iec",
            "nist",
            "gdpr",
            "合同",
            "招标",
            "认证",
            "审计",
        ],
        "authority_keywords": {
            "iso": 2.0,
            "iec": 1.8,
            "nist": 1.6,
            "gdpr": 1.8,
            "doi": 1.4,
        },
        "authority_keywords_zh": {
            "认证": 1.4,
            "合规": 1.5,
            "合同": 1.3,
            "招标": 1.2,
            "标准": 1.3,
        },
        "coverage_bank": {
            "standards": ["iso", "iec", "gbt", "国家标准", "行业标准", "astm", "en "],
            "security_compliance": ["gdpr", "iso 27001", "nist", "等保", "soc"],
            "commerce": ["合同", "招标", "投标", "采购", "invoice", "sla"],
            "publication": ["doi", "论文", "期刊", "arxiv", "预印本"],
            "ip": ["专利", "patent", "uspto", "epo", "商标"],
        },
        "alias_map": {
            "iso13485": "iso 13485",
            "iso27001": "iso 27001",
        },
        "strong_alignment_bonus": False,
    }


def _normalize_lexicon(raw: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    tokens = [str(x).strip().lower() for x in (raw.get("authority_tokens") or []) if str(x).strip()]
    kw_en: Dict[str, float] = {}
    for k, v in (raw.get("authority_keywords") or {}).items():
        kk = str(k).strip().lower()
        if not kk:
            continue
        try:
            kw_en[kk] = float(v)
        except (TypeError, ValueError):
            kw_en[kk] = 1.0
    kw_zh: Dict[str, float] = {}
    for k, v in (raw.get("authority_keywords_zh") or {}).items():
        kk = str(k).strip()
        if not kk:
            continue
        try:
            kw_zh[kk] = float(v)
        except (TypeError, ValueError):
            kw_zh[kk] = 1.0
    bank: Dict[str, list] = {}
    for cat, arr in (raw.get("coverage_bank") or {}).items():
        if isinstance(arr, list):
            bank[str(cat)] = [str(x).strip().lower() for x in arr if str(x).strip()]
        else:
            bank[str(cat)] = []
    aliases: Dict[str, str] = {}
    for k, v in (raw.get("alias_map") or {}).items():
        ks = str(k).strip().lower()
        vs = str(v).strip().lower()
        if ks and vs:
            aliases[ks] = vs
    bonus = bool(raw.get("strong_alignment_bonus", False))
    return {
        "profile_id": str(raw.get("id") or profile_name),
        "authority_tokens": tokens,
        "authority_keywords": kw_en,
        "authority_keywords_zh": kw_zh,
        "coverage_bank": bank,
        "alias_map": aliases,
        "strong_alignment_bonus": bonus,
    }


def load_postproc_lexicon(cfg: Dict[str, Any]) -> Dict[str, Any]:
    name = resolve_authority_profile_name(cfg)
    path = LEXICON_DIR / f"{name}.yaml"
    if not path.is_file():
        name = "default"
        path = LEXICON_DIR / "default.yaml"
    if not path.is_file():
        return _builtin_fallback_lexicon()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return _builtin_fallback_lexicon()
    return _normalize_lexicon(raw, name)
