"""YAML configuration loading with the small set of pilot invariants."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 17,
    "data": {
        "history_length": 4,
        "align_past_trajectories": True,
        "max_frame_gap": 1,
        "validate_on_load": False,
        "require_deep_audit": True,
    },
    "adapter": {
        "bev_compressed_channels": 64,
        "bev_pooled_size": 8,
        "bev_token_dim": 128,
        "hidden_dim": 128,
        "query_dim": 128,
        "dropout": 0.0,
        "learned_gate": True,
    },
    "training": {
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "residual_weight": 0.05,
        "gradient_clip_norm": 5.0,
        "num_workers": 0,
        "torch_num_threads": 4,
    },
    "evaluation": {"worst_fraction": 0.2},
}


def _merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    if int(config["data"]["history_length"]) < 1:
        raise ValueError("data.history_length must be positive")
    max_frame_gap = config["data"].get("max_frame_gap")
    if max_frame_gap is not None and int(max_frame_gap) < 1:
        raise ValueError("data.max_frame_gap must be null or positive")
    for name in (
        "bev_compressed_channels",
        "bev_pooled_size",
        "bev_token_dim",
        "hidden_dim",
        "query_dim",
    ):
        if int(config["adapter"][name]) < 1:
            raise ValueError(f"adapter.{name} must be positive")
    dropout = float(config["adapter"]["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("adapter.dropout must be in [0,1)")
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be positive")
    if int(config["training"]["batch_size"]) < 1:
        raise ValueError("training.batch_size must be positive")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(config["training"]["weight_decay"]) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if float(config["training"]["residual_weight"]) < 0:
        raise ValueError("training.residual_weight must be non-negative")
    if float(config["training"]["gradient_clip_norm"]) < 0:
        raise ValueError("training.gradient_clip_norm must be non-negative")
    if int(config["training"]["num_workers"]) < 0:
        raise ValueError("training.num_workers must be non-negative")
    if int(config["training"]["torch_num_threads"]) < 1:
        raise ValueError("training.torch_num_threads must be positive")
    if not 0.0 < float(config["evaluation"]["worst_fraction"]) <= 1.0:
        raise ValueError("evaluation.worst_fraction must be in (0,1]")


def load_config(path: str | Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration root must be a mapping")
    config = _merge(DEFAULTS, loaded)
    validate_config(config)
    return config
