"""Shared path handling for inference and downstream evaluation.

Inside the production container ``/app/data`` is a bind mount of
``/mnt/storage/d.tanyushkina/transactions`` on the host.  Local smoke tests
can point the same code at a small fixture by setting ``XBANK_DATA_ROOT``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


CONTAINER_DATA_ROOT = Path("/app/data")


def data_root() -> Path:
    return Path(os.environ.get("XBANK_DATA_ROOT", str(CONTAINER_DATA_ROOT)))


def resolve_data_path(value: str | Path) -> Path:
    """Map an ``/app/data`` path to the configured local data root."""
    path = Path(value)
    try:
        relative = path.relative_to(CONTAINER_DATA_ROOT)
    except ValueError:
        return path
    return data_root() / relative


def load_data_config(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)
    cfg["paths"] = {
        key: str(resolve_data_path(value)) for key, value in cfg["paths"].items()
    }
    return cfg


def evaluation_name(data_cfg: Dict[str, Any]) -> str:
    """Stable output namespace for the corpus used at inference time."""
    return str(data_cfg.get("evaluation_name") or data_cfg["name"])


def embedding_dir(
    base_dir: str | Path,
    eval_name: str,
    checkpoint_source: str,
    model: str,
) -> Path:
    source = "zero_shot" if model == "chronos2" else f"{checkpoint_source}_source"
    return resolve_data_path(base_dir) / eval_name / source / model


def downstream_dir(
    base_dir: str | Path,
    eval_name: str,
    checkpoint_source: str,
    model: str,
) -> Path:
    source = "zero_shot" if model == "chronos2" else f"{checkpoint_source}_source"
    return resolve_data_path(base_dir) / eval_name / source / model


def checkpoint_dir(checkpoint_source: str, model: str) -> Path:
    return data_root() / "checkpoints" / f"{checkpoint_source}_source" / model
