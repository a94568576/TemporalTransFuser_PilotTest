"""Non-invasive hooks for the locally released TransFuser++ model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def select_base_path(outputs: Any) -> tuple[torch.Tensor, str]:
    """Select the prediction compatible with either old TF or released TF++."""

    if not isinstance(outputs, (tuple, list)):
        raise TypeError("TransFuser output must be a tuple/list")
    if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
        return outputs[0], "pred_wp"
    if len(outputs) > 2 and isinstance(outputs[2], torch.Tensor):
        return outputs[2], "pred_checkpoint"
    raise RuntimeError("model returned neither pred_wp nor pred_checkpoint")


@dataclass
class FeatureCapture:
    """Capture one feature tensor and remove the hook on context exit."""

    model: torch.nn.Module
    source: str = "backbone_bev"
    cache_spatial_size: int | None = None

    def __post_init__(self) -> None:
        if self.source == "backbone_bev":
            module = self.model.backbone
        elif self.source == "planner_grid":
            if not hasattr(self.model, "change_channel"):
                raise ValueError("planner_grid hook requires model.change_channel")
            module = self.model.change_channel
        else:
            raise ValueError("source must be 'backbone_bev' or 'planner_grid'")
        self._feature: torch.Tensor | None = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        feature = output[0] if self.source == "backbone_bev" else output
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise RuntimeError(f"captured {self.source} is not [B,C,H,W]")
        feature = feature.detach()
        if self.cache_spatial_size is not None:
            if self.cache_spatial_size < 1:
                raise ValueError("cache_spatial_size must be positive")
            if self.cache_spatial_size > min(feature.shape[-2:]):
                raise ValueError(
                    f"refusing to upsample {tuple(feature.shape[-2:])} feature to "
                    f"{self.cache_spatial_size}x{self.cache_spatial_size}"
                )
            feature = F.adaptive_avg_pool2d(feature, (self.cache_spatial_size, self.cache_spatial_size))
        self._feature = feature

    def pop(self) -> torch.Tensor:
        if self._feature is None:
            raise RuntimeError("feature hook did not fire")
        feature, self._feature = self._feature, None
        return feature

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "FeatureCapture":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
