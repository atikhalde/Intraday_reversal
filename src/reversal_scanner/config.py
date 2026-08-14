from __future__ import annotations

import os
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default = resources.files("reversal_scanner.resources").joinpath("default.yaml")
    with default.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            config = _deep_merge(config, yaml.safe_load(handle) or {})
    return config


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value
