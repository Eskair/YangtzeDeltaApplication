# -*- coding: utf-8 -*-
"""
Prompt management: load YAML prompt templates with Jinja2 rendering.

Architecture:
  Each YAML file in src/prompts/ has a three-layer structure:
    - instruction:  general task description (domain-agnostic)
    - domain:       domain-specific context and terminology (swappable)
    - template:     the actual prompt template with {{ variable }} placeholders

  The `render()` function loads the YAML, merges the three layers,
  and renders the template with the given runtime variables.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from jinja2 import Environment, BaseLoader

_PROMPTS_DIR = Path(__file__).resolve().parent
_cache: Dict[str, Dict[str, Any]] = {}

_jinja_env = Environment(loader=BaseLoader())
_jinja_env.keep_trailing_newline = True


def _load_yaml(name: str) -> Dict[str, Any]:
    """Load and cache a YAML prompt file by stem name (without .yaml)."""
    if name in _cache:
        return _cache[name]

    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _cache[name] = data
    return data


def render(name: str, domain: str = "default", **kwargs: Any) -> str:
    """
    Load prompt YAML `name`, select the `domain` section,
    and render the template with the given keyword arguments.

    The final prompt is assembled as:
      [instruction layer]
      [domain layer]
      [template rendered with kwargs]
    """
    data = _load_yaml(name)

    instruction = data.get("instruction", "")
    domains = data.get("domains", {})
    domain_text = domains.get(domain, domains.get("default", ""))
    template_str = data.get("template", "")

    if not template_str:
        raise ValueError(f"Prompt '{name}' has no 'template' field")

    # Render with Jinja2
    try:
        tmpl = _jinja_env.from_string(template_str)
        rendered = tmpl.render(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to render prompt '{name}': {e}") from e

    parts = []
    if instruction:
        parts.append(instruction.strip())
    if domain_text:
        parts.append(domain_text.strip())
    parts.append(rendered.strip())

    return "\n\n".join(parts)


def get_system_prompt(name: str, domain: str = "default") -> str:
    """Load just the system prompt (instruction + domain) without the template."""
    data = _load_yaml(name)
    instruction = data.get("instruction", "")
    domains = data.get("domains", {})
    domain_text = domains.get(domain, domains.get("default", ""))

    parts = []
    if instruction:
        parts.append(instruction.strip())
    if domain_text:
        parts.append(domain_text.strip())

    system = data.get("system", "")
    if system:
        parts.append(system.strip())

    return "\n\n".join(parts) if parts else ""


def get_raw(name: str) -> Dict[str, Any]:
    """Return the raw YAML data for a prompt file."""
    return _load_yaml(name)


def clear_cache() -> None:
    """Clear the prompt cache."""
    _cache.clear()
