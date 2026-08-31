"""Observational action-influence pilot over frozen TF++ BEV latents.

This module deliberately does not call TransFuser++ a world model.  It joins
already frozen perception latents to the low-level controls saved by the CARLA
data agent, constructs route-safe future transitions, and provides the small
passive/FiLM models needed by the staged proxy experiment.

The logged dataset contains one realized action per state.  Consequently these
utilities can test observational action utility and action sensitivity, but not
causal counterfactual accuracy.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from .cache import load_index


ACTION_CANDIDATE_NAMES = (
    "true",
    "shuffled",
    "zero",
    "hold",
    "reverse",
    "other",
)


def deterministic_derangement(size: int, seed: int) -> torch.Tensor:
    """Return a deterministic permutation with no fixed points when possible.

    A non-zero cyclic shift is sufficient, reproducible across torch versions,
    and avoids the retry/fallback ambiguity of random derangement algorithms.
    """

    size = int(size)
    if size < 2:
        raise ValueError("a derangement requires at least two elements")
    shift = abs(int(seed)) % (size - 1) + 1
    return (torch.arange(size, dtype=torch.long) + shift) % size


def _validate_permutation(name: str, value: torch.Tensor, batch_size: int) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(f"{name} must contain integer indices")
    value = value.to(dtype=torch.long)
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must be [{batch_size}], got {tuple(value.shape)}")
    if torch.any(value < 0) or torch.any(value >= batch_size):
        raise ValueError(f"{name} contains an out-of-range index")
    if torch.unique(value).numel() != batch_size:
        raise ValueError(f"{name} must be a permutation")
    return value


def build_action_candidates(
    actions: torch.Tensor,
    shuffle_indices: torch.Tensor,
    other_indices: torch.Tensor,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Build the six fixed action controls used by the pilot.

    ``actions`` is kept in physical units here.  In particular, ``zero`` means
    literal ``steer=throttle=brake=0`` rather than a zero standardized vector.
    Callers should normalize the returned candidate tensor afterwards.
    """

    actions = torch.as_tensor(actions)
    if actions.ndim != 3 or actions.shape[-1] != 3:
        raise ValueError(f"actions must be [B,H,3], got {tuple(actions.shape)}")
    if not actions.is_floating_point():
        raise TypeError("actions must be floating point")
    batch_size, horizon, _ = actions.shape
    if horizon < 1:
        raise ValueError("action horizon must be positive")
    shuffled = _validate_permutation("shuffle_indices", shuffle_indices, batch_size)
    other = _validate_permutation("other_indices", other_indices, batch_size)
    identity = torch.arange(batch_size, dtype=torch.long, device=shuffled.device)
    if batch_size > 1 and (torch.any(shuffled == identity) or torch.any(other == identity)):
        raise ValueError("shuffled and other controls must have no fixed points")
    if batch_size > 2 and torch.equal(shuffled, other):
        raise ValueError("other_indices must differ from shuffle_indices")
    zero = torch.zeros_like(actions)
    hold = actions[:, :1].expand(-1, horizon, -1)
    reverse = actions.flip(dims=(1,))
    candidates = torch.stack(
        (actions, actions[shuffled], zero, hold, reverse, actions[other]), dim=1
    )
    return candidates, ACTION_CANDIDATE_NAMES


def center_action_residuals(residuals: torch.Tensor) -> torch.Tensor:
    """Subtract the same-state candidate mean from ``[B,K,C,H,W]`` residuals."""

    if residuals.ndim != 5:
        raise ValueError(
            "residuals must be [B,K,C,H,W], " f"got {tuple(residuals.shape)}"
        )
    if residuals.shape[1] < 1:
        raise ValueError("residuals must contain at least one action candidate")
    return residuals - residuals.mean(dim=1, keepdim=True)


def _route_measurement_directories(
    index: dict[str, Any], measurements_root: Path | None
) -> dict[str, Path]:
    """Resolve route basenames to their ``measurements`` directories."""

    route_names = {
        str(entry["route_id"]).rsplit("/", 1)[-1] for entry in index["records"]
    }
    if measurements_root is not None:
        root = measurements_root.resolve()
        resolved: dict[str, Path] = {}
        for route_name in route_names:
            direct = root / route_name
            if direct.name == "measurements":
                candidate = direct
            elif (direct / "measurements").is_dir():
                candidate = direct / "measurements"
            elif (root / "measurements").is_dir() and len(route_names) == 1:
                candidate = root / "measurements"
            else:
                candidate = direct / "measurements"
            resolved[route_name] = candidate
        return resolved

    source = index.get("source")
    if not isinstance(source, dict):
        raise ValueError("cache index has no source provenance for measurement lookup")
    raw_sources = source.get("raw_sources")
    if not isinstance(raw_sources, list):
        raise ValueError(
            "cache index source has no raw_sources; pass measurements_root explicitly"
        )
    resolved = {}
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            continue
        for value in raw_source.get("route_directory_paths", []):
            route_path = Path(value)
            name = route_path.name
            if name in resolved and resolved[name] != route_path / "measurements":
                raise ValueError(f"ambiguous raw measurement route basename: {name}")
            resolved[name] = route_path / "measurements"
    missing = sorted(route_names - resolved.keys())
    if missing:
        raise FileNotFoundError(f"raw measurement provenance is missing routes: {missing}")
    return {name: resolved[name] for name in route_names}


def _load_action(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing action measurement: {path}")
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        value = json.load(stream)
    try:
        steer = float(value["steer"])
        throttle = float(value["throttle"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid steer/throttle measurement: {path}") from exc
    if "brake" not in value and "control_brake" not in value:
        raise ValueError(f"measurement has neither brake nor control_brake: {path}")
    brake_values = [value[key] for key in ("brake", "control_brake") if key in value]
    if any(item not in (False, True, 0, 1, 0.0, 1.0) for item in brake_values):
        raise ValueError(f"brake values must be boolean or 0/1: {path}")
    if not -1.0 <= steer <= 1.0 or not 0.0 <= throttle <= 1.0:
        raise ValueError(f"action is outside CARLA control bounds: {path}")
    brake = float(any(bool(item) for item in brake_values))
    action = torch.tensor((steer, throttle, brake), dtype=torch.float32)
    if not torch.isfinite(action).all():
        raise ValueError(f"non-finite action measurement: {path}")
    return action


class ActionTransitionDataset(Dataset):
    """Route-safe ``(Z_t, A_t:t+h, Z_t+h)`` frozen-latent transitions.

    Test access is denied by default so development runs cannot silently open
    the existing cache's test partition.
    """

    def __init__(
        self,
        cache_root: str | Path,
        measurements_root: str | Path | None = None,
        *,
        split: str,
        action_horizon: int = 4,
        horizon: int | None = None,
        allow_test: bool = False,
        max_frame_gap: int | None = 1,
        expected_cadence_seconds: float | None = 0.25,
        preload: bool = False,
    ) -> None:
        if horizon is not None:
            if action_horizon != 4 and int(horizon) != int(action_horizon):
                raise ValueError("horizon and action_horizon disagree")
            action_horizon = int(horizon)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        if split == "test" and not allow_test:
            raise ValueError("test split access requires allow_test=True")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if max_frame_gap is not None and max_frame_gap < 1:
            raise ValueError("max_frame_gap must be positive or None")
        if expected_cadence_seconds is not None and expected_cadence_seconds <= 0.0:
            raise ValueError("expected_cadence_seconds must be positive or None")

        self.cache_root = Path(cache_root)
        self.split = split
        self.action_horizon = int(action_horizon)
        self.max_frame_gap = max_frame_gap
        index = load_index(self.cache_root)
        if int(index.get("schema_version", -1)) != 3:
            raise ValueError("action pilot requires cache schema_version=3")

        route_splits: dict[str, set[str]] = defaultdict(set)
        by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in index["records"]:
            route_id = str(entry["route_id"])
            route_splits[route_id].add(str(entry["split"]))
            if entry["split"] == split:
                by_route[route_id].append(entry)
        leaking = {key: value for key, value in route_splits.items() if len(value) != 1}
        if leaking:
            raise ValueError(f"route split leakage detected: {leaking}")

        measurement_dirs = _route_measurement_directories(
            index, Path(measurements_root) if measurements_root is not None else None
        )
        action_cache: dict[tuple[str, int], torch.Tensor] = {}
        self.transitions: list[
            tuple[dict[str, Any], dict[str, Any], torch.Tensor]
        ] = []
        for route_id, entries in sorted(by_route.items()):
            entries.sort(key=lambda item: (int(item["frame_id"]), float(item["timestamp"])))
            route_name = route_id.rsplit("/", 1)[-1]
            for start in range(0, len(entries) - self.action_horizon):
                window = entries[start : start + self.action_horizon + 1]
                frame_ids = [int(entry["frame_id"]) for entry in window]
                gaps = [right - left for left, right in zip(frame_ids, frame_ids[1:])]
                if any(gap <= 0 for gap in gaps):
                    raise ValueError(f"duplicate/unsorted frames in {route_id}: {frame_ids}")
                if self.max_frame_gap is not None and any(
                    gap > self.max_frame_gap for gap in gaps
                ):
                    continue
                if expected_cadence_seconds is not None:
                    timestamps = [float(entry["timestamp"]) for entry in window]
                    deltas = [
                        right - left for left, right in zip(timestamps, timestamps[1:])
                    ]
                    if any(
                        not math.isclose(
                            delta,
                            expected_cadence_seconds,
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        )
                        for delta in deltas
                    ):
                        continue
                actions = []
                for frame_id in frame_ids[:-1]:
                    key = (route_name, frame_id)
                    if key not in action_cache:
                        action_cache[key] = _load_action(
                            measurement_dirs[route_name] / f"{frame_id:04d}.json.gz"
                        )
                    actions.append(action_cache[key])
                self.transitions.append((window[0], window[-1], torch.stack(actions)))
        if not self.transitions:
            raise ValueError(
                f"split '{split}' has no contiguous transitions for horizon={self.action_horizon}"
            )

        self._preloaded: dict[str, dict[str, Any]] | None = {} if preload else None
        if self._preloaded is not None:
            for current, future, _ in self.transitions:
                self._load_record(current)
                self._load_record(future)

    def __len__(self) -> int:
        return len(self.transitions)

    def _load_record(self, entry: dict[str, Any]) -> dict[str, Any]:
        relative = str(entry["path"])
        if self._preloaded is not None and relative in self._preloaded:
            return self._preloaded[relative]
        path = self.cache_root / relative
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("route_id") != entry["route_id"] or int(
            record.get("frame_id", -1)
        ) != int(entry["frame_id"]):
            raise ValueError(f"index/record identity mismatch: {path}")
        latent = torch.as_tensor(record.get("bev_feature"))
        pose = torch.as_tensor(record.get("ego_pose"))
        if latent.ndim != 3 or not latent.is_floating_point():
            raise ValueError(f"bev_feature must be floating [C,H,W]: {path}")
        if pose.shape != (3,) or not pose.is_floating_point():
            raise ValueError(f"ego_pose must be floating [3]: {path}")
        compact = {
            "bev_feature": latent.float(),
            "ego_pose": pose.double(),
        }
        if self._preloaded is not None:
            self._preloaded[relative] = compact
        return compact

    def __getitem__(self, index: int) -> dict[str, Any]:
        current_entry, future_entry, actions = self.transitions[index]
        current = self._load_record(current_entry)
        future = self._load_record(future_entry)
        return {
            "current_latent": current["bev_feature"],
            "future_latent": future["bev_feature"],
            "current_pose": current["ego_pose"],
            "future_pose": future["ego_pose"],
            "actions": actions.clone(),
            "route_id": str(current_entry["route_id"]),
            "frame_id": int(current_entry["frame_id"]),
            "future_frame_id": int(future_entry["frame_id"]),
            "sample_id": (
                f"{current_entry['route_id']}::{int(current_entry['frame_id'])}"
                f"->{int(future_entry['frame_id'])}"
            ),
            "dataset_index": int(index),
        }

    @property
    def actions(self) -> torch.Tensor:
        """Return every logged sequence in dataset order, in physical units."""

        return torch.stack([value[2] for value in self.transitions])

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(str(value[0]["route_id"]) for value in self.transitions)

    @property
    def sample_shape(self) -> dict[str, tuple[int, ...]]:
        sample = self[0]
        return {
            name: tuple(sample[name].shape)
            for name in ("current_latent", "future_latent", "actions")
        }


class PassiveFuturePredictor(nn.Module):
    """Small action-free residual predictor, initialized as persistence."""

    def __init__(self, channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("channels and hidden_channels must be positive")
        self.channels = int(channels)
        self.network = nn.Sequential(
            nn.Conv2d(self.channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.channels, kernel_size=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, current_latent: torch.Tensor) -> torch.Tensor:
        if current_latent.ndim != 4 or current_latent.shape[1] != self.channels:
            raise ValueError(
                f"current_latent must be [B,{self.channels},H,W], "
                f"got {tuple(current_latent.shape)}"
            )
        return current_latent + self.network(current_latent)


class ActionFiLMResidualAdapter(nn.Module):
    """Same-capacity uncentered/centered spatial FiLM action residual."""

    def __init__(
        self,
        channels: int,
        action_horizon: int,
        hidden_channels: int = 32,
        action_hidden_dim: int = 64,
        centered: bool = True,
        gate_bias: float = -5.0,
    ) -> None:
        super().__init__()
        if min(channels, action_horizon, hidden_channels, action_hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.channels = int(channels)
        self.action_horizon = int(action_horizon)
        self.hidden_channels = int(hidden_channels)
        self.centered = bool(centered)
        self.state_encoder = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_horizon * 3, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, self.hidden_channels * 2),
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1),
        )
        # Preserve the frozen base exactly at initialization while allowing the
        # final projection to bootstrap on the first optimization step.
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_bias)))

    def forward(
        self,
        current_latent: torch.Tensor,
        base_future: torch.Tensor,
        action_candidates: torch.Tensor,
        reference_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if current_latent.ndim != 4 or current_latent.shape[1] != self.channels:
            raise ValueError(
                f"current_latent must be [B,{self.channels},H,W], "
                f"got {tuple(current_latent.shape)}"
            )
        if base_future.shape != current_latent.shape:
            raise ValueError("base_future must match current_latent shape")
        actions = torch.as_tensor(action_candidates)
        if actions.ndim == 3:
            actions = actions.unsqueeze(1)
        if actions.ndim != 4 or actions.shape[0] != current_latent.shape[0]:
            raise ValueError("action_candidates must be [B,K,H,3] or [B,H,3]")
        if tuple(actions.shape[-2:]) != (self.action_horizon, 3):
            raise ValueError(
                f"action candidates must end in [{self.action_horizon},3], "
                f"got {tuple(actions.shape)}"
            )
        if not actions.is_floating_point():
            raise TypeError("action_candidates must be floating point")

        batch_size, candidates = actions.shape[:2]
        if self.centered and reference_actions is None and candidates < 2:
            raise ValueError(
                "a centered single-candidate call requires non-empty reference_actions"
            )
        base_detached = base_future.detach()
        state = self.state_encoder(torch.cat((current_latent, base_detached), dim=1))

        def raw_action_residual(candidate_actions: torch.Tensor) -> torch.Tensor:
            candidate_count = candidate_actions.shape[1]
            encoded = self.action_encoder(
                candidate_actions.reshape(batch_size * candidate_count, -1)
            )
            candidate_gamma, candidate_beta = encoded.chunk(2, dim=-1)
            candidate_gamma = torch.tanh(candidate_gamma).reshape(
                batch_size, candidate_count, self.hidden_channels, 1, 1
            )
            candidate_beta = candidate_beta.reshape(
                batch_size, candidate_count, self.hidden_channels, 1, 1
            )
            candidate_modulated = (
                state[:, None] * (1.0 + candidate_gamma) + candidate_beta
            )
            return self.output_head(
                candidate_modulated.reshape(
                    batch_size * candidate_count,
                    self.hidden_channels,
                    *current_latent.shape[-2:],
                )
            ).reshape(
                batch_size,
                candidate_count,
                self.channels,
                *current_latent.shape[-2:],
            )

        raw_residual = raw_action_residual(actions)
        if reference_actions is not None:
            references = torch.as_tensor(reference_actions)
            if not self.centered:
                raise ValueError("reference_actions are only valid for a centered adapter")
            if references.ndim != 4 or references.shape[0] != batch_size:
                raise ValueError("reference_actions must be [B,R,H,3]")
            if references.shape[1] < 1:
                raise ValueError("reference_actions must contain at least one reference")
            if tuple(references.shape[-2:]) != (self.action_horizon, 3):
                raise ValueError(
                    f"reference actions must end in [{self.action_horizon},3]"
                )
            if not references.is_floating_point():
                raise TypeError("reference_actions must be floating point")
            reference_residual = raw_action_residual(references)
            residual = raw_residual - reference_residual.mean(dim=1, keepdim=True)
        else:
            reference_residual = None
            residual = (
                center_action_residuals(raw_residual) if self.centered else raw_residual
            )
        gate_scalar = torch.sigmoid(self.gate_logit)
        gate = gate_scalar.expand(batch_size, candidates).reshape(
            batch_size, candidates, 1, 1, 1
        )
        prediction = base_detached[:, None] + gate * residual
        return {
            "prediction": prediction,
            "residual": residual,
            "raw_residual": raw_residual,
            "reference_residual": reference_residual,
            "gate": gate,
        }


def train_normalization_statistics(
    dataset: ActionTransitionDataset,
) -> dict[str, torch.Tensor]:
    """Compute channel/action statistics from a training dataset only."""

    if dataset.split != "train":
        raise ValueError("normalization statistics must come from the train split")
    channel_sum: torch.Tensor | None = None
    channel_sq_sum: torch.Tensor | None = None
    pixel_count = 0
    for sample in dataset:
        latent = sample["future_latent"].double()
        values = latent.flatten(start_dim=1)
        current_sum = values.sum(dim=1)
        current_sq_sum = values.square().sum(dim=1)
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_sq_sum = (
            current_sq_sum if channel_sq_sum is None else channel_sq_sum + current_sq_sum
        )
        pixel_count += values.shape[1]
    if channel_sum is None or channel_sq_sum is None or pixel_count < 2:
        raise ValueError("training dataset is empty")
    latent_mean = channel_sum / pixel_count
    latent_var = channel_sq_sum / pixel_count - latent_mean.square()
    latent_std = latent_var.clamp_min(1e-8).sqrt()

    actions = dataset.actions.double()
    action_mean = actions.mean(dim=(0, 1))
    action_std = actions.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    return {
        "latent_mean": latent_mean.float(),
        "latent_std": latent_std.float(),
        "action_mean": action_mean.float(),
        "action_std": action_std.float(),
    }


def normalize_latent(value: torch.Tensor, statistics: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = statistics["latent_mean"].to(device=value.device, dtype=value.dtype)
    std = statistics["latent_std"].to(device=value.device, dtype=value.dtype)
    return (value - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)


def normalize_actions(value: torch.Tensor, statistics: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = statistics["action_mean"].to(device=value.device, dtype=value.dtype)
    std = statistics["action_std"].to(device=value.device, dtype=value.dtype)
    return (value - mean) / std


def denormalize_latent(value: torch.Tensor, statistics: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = statistics["latent_mean"].to(device=value.device, dtype=value.dtype)
    std = statistics["latent_std"].to(device=value.device, dtype=value.dtype)
    return value * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def freeze_module(model: nn.Module) -> nn.Module:
    model.requires_grad_(False).eval()
    return model


def route_macro_mean(values: Sequence[float], route_ids: Sequence[str]) -> float:
    if len(values) != len(route_ids) or len(values) == 0:
        raise ValueError("values and route_ids must have equal non-zero length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, route_id in zip(values, route_ids, strict=True):
        grouped[str(route_id)].append(float(value))
    return float(
        sum(sum(route_values) / len(route_values) for route_values in grouped.values())
        / len(grouped)
    )
