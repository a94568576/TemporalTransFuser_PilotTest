"""Pilot objective: trajectory L1 plus residual magnitude regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def residual_adapter_loss(
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    residual_weight: float,
) -> dict[str, torch.Tensor]:
    trajectory = F.l1_loss(prediction["trajectory"], target)
    residual = prediction["delta"].abs().mean()
    total = trajectory + float(residual_weight) * residual
    return {"loss": total, "trajectory_l1": trajectory, "residual_l1": residual}
