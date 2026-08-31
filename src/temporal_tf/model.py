"""Small GRU residual adapter for cached frozen-planner trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterVariant:
    """Capacity and input contract for one adapter control.

    The model only needs to know whether a modality has a temporal sequence.
    ``trajectory_history`` and ``bev_history`` additionally describe where the
    engine must obtain that sequence, which makes controls auditable instead of
    overloading a pair of booleans.
    """

    name: str
    trajectory_history: Literal["none", "past", "repeat_current"]
    bev_history: Literal["none", "past", "repeat_current", "shuffled_past"]
    capacity_target: str | None = None

    @property
    def use_past_trajectory(self) -> bool:
        """Whether the capacity-matched model has a trajectory sequence input."""

        return self.trajectory_history != "none"

    @property
    def use_past_bev(self) -> bool:
        """Whether the capacity-matched model has a BEV sequence input."""

        return self.bev_history != "none"

    def input_semantics(self) -> dict[str, str | None]:
        """Return the exact, serializable input semantics locked in checkpoints."""

        return {
            "current_trajectory": "query",
            "trajectory_history": self.trajectory_history,
            "bev_history": self.bev_history,
            "capacity_target": self.capacity_target,
            "history_scope": (
                "none"
                if self.trajectory_history == "none" and self.bev_history == "none"
                else "same_sample_window"
            ),
            "shuffle_algorithm": (
                "sha256_seed_sample_id_index_v1_non_identity"
                if self.bev_history == "shuffled_past"
                else None
            ),
        }


VARIANTS = {
    "current_only": AdapterVariant("current_only", "none", "none"),
    "trajectory_only": AdapterVariant("trajectory_only", "past", "none"),
    "bev_only": AdapterVariant("bev_only", "none", "past"),
    # Preferred research-facing name. ``bev_only`` remains a backwards-
    # compatible alias for historical synthetic artifacts.
    "past_bev": AdapterVariant("past_bev", "none", "past"),
    "combined": AdapterVariant("combined", "past", "past"),
    # No history or BEV is supplied.  Its current-query/fusion widths are chosen
    # by the engine to approximate the trainable parameter count of ``past_bev``.
    "current_only_matched": AdapterVariant(
        "current_only_matched", "none", "none", capacity_target="past_bev"
    ),
    # Exact architecture/state-shape match to ``combined``.  Only current-time
    # tensors are repeated, so this control has capacity but no past signal.
    "repeat_current": AdapterVariant(
        "repeat_current", "repeat_current", "repeat_current", capacity_target="combined"
    ),
    # Exact architecture/state-shape match to ``past_bev`` while withholding
    # all past information and preserving the same recurrent unroll length.
    "current_bev": AdapterVariant(
        "current_bev", "none", "repeat_current", capacity_target="past_bev"
    ),
    # Same capacity and data window as ``past_bev``; only temporal order changes.
    "shuffled_past_bev": AdapterVariant(
        "shuffled_past_bev", "none", "shuffled_past", capacity_target="past_bev"
    ),
}


def current_only_parameter_count(
    *,
    num_waypoints: int,
    query_dim: int,
    hidden_dim: int,
    learned_gate: bool,
) -> int:
    """Exact parameter count for this module's no-history architecture."""

    trajectory_dim = int(num_waypoints) * 2
    query_dim = int(query_dim)
    hidden_dim = int(hidden_dim)
    # current_query: Linear + LayerNorm; fusion: two Linear layers;
    # delta_head and the optional gate_head.
    total = query_dim * (trajectory_dim + 3 + hidden_dim)
    total += hidden_dim * hidden_dim + hidden_dim * trajectory_dim + 2 * hidden_dim
    total += trajectory_dim
    if learned_gate:
        total += hidden_dim + 1
    return total


def match_current_only_dimensions(
    *,
    num_waypoints: int,
    base_query_dim: int,
    base_hidden_dim: int,
    learned_gate: bool,
    target_parameters: int,
) -> tuple[int, int]:
    """Find widened query/hidden dimensions closest to a capacity target.

    Both dimensions are constrained to be at least the configured base widths.
    The search uses the exact closed-form parameter count above and is fully
    deterministic; it does not instantiate or inspect any data-dependent model.
    """

    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    base_query_dim = int(base_query_dim)
    base_hidden_dim = int(base_hidden_dim)
    search_limit = max(base_hidden_dim, int(target_parameters**0.5) * 2 + 4)
    best: tuple[int, int, int, int] | None = None
    trajectory_dim = int(num_waypoints) * 2
    for hidden_dim in range(base_hidden_dim, search_limit + 1):
        constant = hidden_dim * hidden_dim + hidden_dim * trajectory_dim + 2 * hidden_dim
        constant += trajectory_dim
        if learned_gate:
            constant += hidden_dim + 1
        query_coefficient = trajectory_dim + 3 + hidden_dim
        estimated_query = round((target_parameters - constant) / query_coefficient)
        for query_dim in {
            base_query_dim,
            max(base_query_dim, estimated_query - 1),
            max(base_query_dim, estimated_query),
            max(base_query_dim, estimated_query + 1),
        }:
            count = current_only_parameter_count(
                num_waypoints=num_waypoints,
                query_dim=query_dim,
                hidden_dim=hidden_dim,
                learned_gate=learned_gate,
            )
            candidate = (abs(count - target_parameters), query_dim + hidden_dim, query_dim, hidden_dim)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("failed to find current-only capacity-matched dimensions")
    return best[2], best[3]


class BEVCompressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        compressed_channels: int = 64,
        pooled_size: int = 8,
        token_dim: int = 128,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, compressed_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
            nn.Flatten(),
            nn.Linear(compressed_channels * pooled_size * pooled_size, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.network(bev)


class TemporalResidualAdapter(nn.Module):
    """Predict ``T' = T + sigmoid(g) * delta`` from past frozen outputs."""

    def __init__(
        self,
        *,
        num_waypoints: int,
        bev_channels: int,
        use_past_trajectory: bool,
        use_past_bev: bool,
        bev_compressed_channels: int = 64,
        bev_pooled_size: int = 8,
        bev_token_dim: int = 128,
        hidden_dim: int = 128,
        query_dim: int = 128,
        dropout: float = 0.0,
        learned_gate: bool = True,
    ) -> None:
        super().__init__()
        self.num_waypoints = int(num_waypoints)
        self.trajectory_dim = self.num_waypoints * 2
        self.use_past_trajectory = bool(use_past_trajectory)
        self.use_past_bev = bool(use_past_bev)
        self.learned_gate = bool(learned_gate)

        frame_dim = 0
        if self.use_past_bev:
            self.bev_compressor = BEVCompressor(
                bev_channels,
                compressed_channels=bev_compressed_channels,
                pooled_size=bev_pooled_size,
                token_dim=bev_token_dim,
            )
            frame_dim += bev_token_dim
        else:
            self.bev_compressor = None
        if self.use_past_trajectory:
            frame_dim += self.trajectory_dim

        self.temporal_encoder = (
            nn.GRU(frame_dim, hidden_dim, batch_first=True) if frame_dim > 0 else None
        )
        self.current_query = nn.Sequential(
            nn.Linear(self.trajectory_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.ReLU(inplace=True),
        )
        fusion_dim = (hidden_dim if self.temporal_encoder is not None else 0) + query_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(hidden_dim, self.trajectory_dim)
        self.gate_head = nn.Linear(hidden_dim, 1) if self.learned_gate else None

        # Start as an exact identity adapter.  This prevents an untrained module
        # from degrading the frozen planner before it sees data.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self,
        *,
        current_trajectory: torch.Tensor,
        past_trajectory: torch.Tensor | None = None,
        past_bev: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if current_trajectory.ndim != 3 or current_trajectory.shape[-2:] != (self.num_waypoints, 2):
            raise ValueError(
                f"current_trajectory must be [B,{self.num_waypoints},2], "
                f"got {tuple(current_trajectory.shape)}"
            )
        batch_size = current_trajectory.shape[0]
        frame_tokens: list[torch.Tensor] = []

        if self.use_past_bev:
            if past_bev is None or past_bev.ndim != 5:
                raise ValueError("past_bev must be [B,T,C,H,W] for this variant")
            batch, history, channels, height, width = past_bev.shape
            if batch != batch_size:
                raise ValueError("past_bev batch size does not match current trajectory")
            bev_tokens = self.bev_compressor(past_bev.reshape(batch * history, channels, height, width))
            frame_tokens.append(bev_tokens.reshape(batch, history, -1))

        if self.use_past_trajectory:
            if past_trajectory is None or past_trajectory.ndim != 4:
                raise ValueError("past_trajectory must be [B,T,N,2] for this variant")
            if past_trajectory.shape[0] != batch_size or past_trajectory.shape[-2:] != (self.num_waypoints, 2):
                raise ValueError("past_trajectory shape does not match current trajectory")
            frame_tokens.append(past_trajectory.flatten(start_dim=2))

        query = self.current_query(current_trajectory.flatten(start_dim=1))
        if self.temporal_encoder is None:
            fusion_input = query
        else:
            history_lengths = {token.shape[1] for token in frame_tokens}
            if len(history_lengths) != 1:
                raise ValueError("past modalities have different history lengths")
            temporal_input = torch.cat(frame_tokens, dim=-1)
            _, hidden = self.temporal_encoder(temporal_input)
            fusion_input = torch.cat((hidden[-1], query), dim=-1)
        fused = self.fusion(fusion_input)
        delta = self.delta_head(fused).reshape(batch_size, self.num_waypoints, 2)
        if self.gate_head is None:
            gate = torch.ones((batch_size, 1, 1), dtype=delta.dtype, device=delta.device)
        else:
            gate = torch.sigmoid(self.gate_head(fused)).reshape(batch_size, 1, 1)
        refined = current_trajectory + gate * delta
        return {"trajectory": refined, "delta": delta, "gate": gate}
