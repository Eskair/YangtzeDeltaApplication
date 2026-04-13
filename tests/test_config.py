# -*- coding: utf-8 -*-
"""Tests for the domain configuration module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config, clear_cache, DomainConfig


def test_default_config_loads():
    clear_cache()
    cfg = get_config("default")
    assert isinstance(cfg, DomainConfig)
    assert cfg.domain_name == "default"


def test_dimension_names():
    cfg = get_config("default")
    dims = cfg.dimension_names
    assert "team" in dims
    assert "objectives" in dims
    assert "strategy" in dims
    assert "innovation" in dims
    assert "feasibility" in dims
    assert len(dims) == 5


def test_valid_types():
    cfg = get_config("default")
    types = cfg.valid_types
    assert "team_member" in types
    assert "market" in types
    assert "risk" in types
    assert "other" in types
    assert len(types) >= 15


def test_type_to_dims_map():
    cfg = get_config("default")
    mapping = cfg.get_type_to_dims_map()
    assert "team_member" in mapping
    assert "team" in mapping["team_member"]
    assert "collaboration" in mapping
    assert "team" in mapping["collaboration"]
    assert "strategy" in mapping["collaboration"]


def test_dimension_labels():
    cfg = get_config("default")
    labels = cfg.dimension_labels_zh
    assert labels["team"] == "团队与治理"
    assert labels["feasibility"] == "资源与可行性"


def test_dimension_weights():
    cfg = get_config("default")
    weights = cfg.dimension_weights
    assert weights["team"] == 1.0
    assert weights["innovation"] == 1.10
    assert weights["feasibility"] == 1.20


def test_get_dimension():
    cfg = get_config("default")
    team = cfg.get_dimension("team")
    assert team is not None
    assert team.name == "team"
    assert len(team.aspects) > 0
    assert team.min_questions > 0


def test_get_dimension_config_dict():
    cfg = get_config("default")
    d = cfg.get_dimension_config_dict("strategy")
    assert "min_q" in d
    assert "max_q" in d
    assert "focus_zh" in d
    assert "aspects" in d
    assert isinstance(d["aspects"], list)


def test_fallback_dimension():
    cfg = get_config("default")
    assert cfg.fallback_dimension == "feasibility"


def test_nonexistent_domain_falls_back():
    clear_cache()
    cfg = get_config("nonexistent_domain_xyz")
    assert isinstance(cfg, DomainConfig)
    assert len(cfg.dimension_names) > 0


if __name__ == "__main__":
    tests = [
        test_default_config_loads,
        test_dimension_names,
        test_valid_types,
        test_type_to_dims_map,
        test_dimension_labels,
        test_dimension_weights,
        test_get_dimension,
        test_get_dimension_config_dict,
        test_fallback_dimension,
        test_nonexistent_domain_falls_back,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
