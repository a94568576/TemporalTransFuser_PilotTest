"""Leakage-resistant primitives for MPC-local action grounding.

The module intentionally stops at the pure data/model/loss/metric core.  It
does not mine candidates with an MPC implementation and it never treats a
model rollout as a real counterfactual outcome.  NPZ records are grouped by an
explicit ``state_id`` so ranking pairs and state-macro metrics cannot cross an
initial-state boundary.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TRAIN_SPLIT = 0
VAL_SPLIT = 1
TEST_SPLIT = 2
SPLIT_NAME_TO_CODE = {
    "train": TRAIN_SPLIT,
    "val": VAL_SPLIT,
    "test": TEST_SPLIT,
}
VALID_SPLIT_CODES = frozenset(SPLIT_NAME_TO_CODE.values())
VALID_CEM_ITERATIONS = frozenset((0, 1, 2))
REQUIRED_NPZ_ARRAYS = frozenset(
    (
        "state_id",
        "split_code",
        "cem_iteration",
        "action_params",
        "initial_features",
        "outcome_features",
        "collision",
        "real_cost",
    )
)


def _canonical_state_id(value: Any) -> tuple[str, str]:
    """Return a sortable, type-preserving state identifier."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("state_id bytes must be valid UTF-8") from exc
    if isinstance(value, str):
        if not value:
            raise ValueError("state_id strings must be non-empty")
        return ("str", value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return ("int", str(int(value)))
    raise TypeError("state_id must contain non-empty strings/bytes or integers")


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _require_float_array(name: str, value: np.ndarray, shape: tuple[int | None, ...]) -> None:
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _normalize_split_code(value: int | str) -> int:
    if isinstance(value, str):
        if value not in SPLIT_NAME_TO_CODE:
            raise ValueError(f"unknown split name: {value}")
        return SPLIT_NAME_TO_CODE[value]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("split code must be an integer or train/val/test")
    code = int(value)
    if code not in VALID_SPLIT_CODES:
        raise ValueError(f"unsupported split code: {code}")
    return code


def _normalize_split_codes(values: Iterable[int | str]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(_normalize_split_code(value) for value in values))
    if not result:
        raise ValueError("at least one split must be requested")
    return result


@dataclass(frozen=True)
class MPCGroundingRecords:
    """Validated, immutable candidate records from the NPZ contract."""

    state_id: np.ndarray
    split_code: np.ndarray
    cem_iteration: np.ndarray
    action_params: np.ndarray
    initial_features: np.ndarray
    outcome_features: np.ndarray
    collision: np.ndarray
    real_cost: np.ndarray

    def __post_init__(self) -> None:
        state_id = np.asarray(self.state_id)
        if state_id.ndim != 1 or state_id.size == 0:
            raise ValueError("state_id must be a non-empty [N] array")
        if state_id.dtype.kind == "O":
            raise TypeError("state_id object arrays are forbidden")
        for value in state_id:
            _canonical_state_id(value)

        split_code = np.asarray(self.split_code)
        cem_iteration = np.asarray(self.cem_iteration)
        for name, value in (
            ("split_code", split_code),
            ("cem_iteration", cem_iteration),
        ):
            if value.shape != state_id.shape:
                raise ValueError(f"{name} must have shape {state_id.shape}")
            if value.dtype.kind not in "iu" or value.dtype.kind == "b":
                raise TypeError(f"{name} must have an integer dtype")

        split_code = split_code.astype(np.int64, copy=False)
        cem_iteration = cem_iteration.astype(np.int64, copy=False)
        invalid_splits = sorted(set(split_code.tolist()) - VALID_SPLIT_CODES)
        if invalid_splits:
            raise ValueError(f"split_code contains invalid values: {invalid_splits}")
        invalid_iterations = sorted(
            set(cem_iteration.tolist()) - VALID_CEM_ITERATIONS
        )
        if invalid_iterations:
            raise ValueError(
                f"cem_iteration contains invalid values: {invalid_iterations}"
            )

        size = int(state_id.shape[0])
        action_params = np.asarray(self.action_params)
        initial_features = np.asarray(self.initial_features)
        outcome_features = np.asarray(self.outcome_features)
        collision = np.asarray(self.collision)
        real_cost = np.asarray(self.real_cost)
        _require_float_array("action_params", action_params, (size, 4))
        if initial_features.ndim != 2 or initial_features.shape[0] != size:
            raise ValueError(
                "initial_features must be [N,S], "
                f"got {initial_features.shape} for N={size}"
            )
        if initial_features.shape[1] < 1:
            raise ValueError("initial_features state dimension must be positive")
        _require_float_array(
            "initial_features", initial_features, (size, initial_features.shape[1])
        )
        _require_float_array("outcome_features", outcome_features, (size, 4))
        _require_float_array("collision", collision, (size,))
        if np.any(collision < 0.0):
            raise ValueError("collision values must be non-negative")
        _require_float_array("real_cost", real_cost, (size,))

        # A state is an atomic experimental unit.  Reusing its identifier across
        # splits is direct leakage, and changing its initial feature invalidates
        # the within-state counterfactual interpretation.
        groups: dict[tuple[str, str], list[int]] = {}
        for index, value in enumerate(state_id):
            groups.setdefault(_canonical_state_id(value), []).append(index)
        for key, indices in groups.items():
            state_splits = np.unique(split_code[indices])
            if state_splits.size != 1:
                raise ValueError(
                    f"state_id {key[1]!r} crosses split codes {state_splits.tolist()}"
                )
            reference = initial_features[indices[0]]
            if not np.allclose(
                initial_features[indices], reference[None], rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"state_id {key[1]!r} has inconsistent initial_features"
                )

        object.__setattr__(self, "state_id", _readonly_copy(state_id))
        object.__setattr__(self, "split_code", _readonly_copy(split_code, dtype=np.int64))
        object.__setattr__(
            self, "cem_iteration", _readonly_copy(cem_iteration, dtype=np.int64)
        )
        object.__setattr__(
            self, "action_params", _readonly_copy(action_params, dtype=np.float32)
        )
        object.__setattr__(
            self,
            "initial_features",
            _readonly_copy(initial_features, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "outcome_features",
            _readonly_copy(outcome_features, dtype=np.float32),
        )
        object.__setattr__(
            self, "collision", _readonly_copy(collision, dtype=np.float32)
        )
        object.__setattr__(
            self, "real_cost", _readonly_copy(real_cost, dtype=np.float32)
        )

    def __len__(self) -> int:
        return int(self.state_id.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.initial_features.shape[1])

    def subset(self, indices: Sequence[int] | np.ndarray) -> "MPCGroundingRecords":
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise TypeError("subset indices must be a one-dimensional integer array")
        selected = selected.astype(np.int64, copy=False)
        if selected.size == 0:
            raise ValueError("record subset cannot be empty")
        if np.any(selected < 0) or np.any(selected >= len(self)):
            raise IndexError("record subset index is out of bounds")
        return MPCGroundingRecords(
            state_id=self.state_id[selected],
            split_code=self.split_code[selected],
            cem_iteration=self.cem_iteration[selected],
            action_params=self.action_params[selected],
            initial_features=self.initial_features[selected],
            outcome_features=self.outcome_features[selected],
            collision=self.collision[selected],
            real_cost=self.real_cost[selected],
        )


@dataclass(frozen=True)
class StateGroup:
    state_id: Any
    split_code: int
    indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise ValueError("StateGroup indices must be a non-empty integer vector")
        object.__setattr__(self, "indices", _readonly_copy(indices, dtype=np.int64))


def load_mpc_records(
    path: str | Path,
    *,
    split_codes: Iterable[int | str] = (TRAIN_SPLIT, VAL_SPLIT),
    allow_test: bool = False,
    allow_extra_arrays: bool = False,
) -> MPCGroundingRecords:
    """Load and validate NPZ records, returning only explicitly allowed splits.

    Test rows require both an explicit test request and ``allow_test=True``.
    Object arrays are never loaded because ``allow_pickle`` is fixed to false.
    """

    requested = _normalize_split_codes(split_codes)
    if TEST_SPLIT in requested and not allow_test:
        raise PermissionError("test split access requires allow_test=True")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            missing = sorted(REQUIRED_NPZ_ARRAYS - keys)
            extra = sorted(keys - REQUIRED_NPZ_ARRAYS)
            if missing:
                raise ValueError(f"NPZ is missing required arrays: {missing}")
            if extra and not allow_extra_arrays:
                raise ValueError(f"NPZ contains unexpected arrays: {extra}")
            records = MPCGroundingRecords(
                **{name: archive[name] for name in REQUIRED_NPZ_ARRAYS}
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to load MPC grounding NPZ: {source}") from exc

    present = set(records.split_code.tolist())
    absent = sorted(set(requested) - present)
    if absent:
        raise ValueError(f"requested split codes are absent from NPZ: {absent}")
    selected = np.flatnonzero(np.isin(records.split_code, requested))
    return records.subset(selected)


def state_group_indices(
    records: MPCGroundingRecords,
    *,
    split_codes: Iterable[int | str] = (TRAIN_SPLIT, VAL_SPLIT),
    cem_iterations: Iterable[int] | None = None,
    min_candidates: int = 1,
    allow_test: bool = False,
) -> tuple[StateGroup, ...]:
    """Return deterministically ordered state groups after strict filters."""

    requested = _normalize_split_codes(split_codes)
    if TEST_SPLIT in requested and not allow_test:
        raise PermissionError("test split grouping requires allow_test=True")
    if isinstance(min_candidates, bool) or int(min_candidates) < 1:
        raise ValueError("min_candidates must be a positive integer")
    min_candidates = int(min_candidates)

    iteration_values: tuple[int, ...] | None = None
    if cem_iterations is not None:
        iteration_values = tuple(dict.fromkeys(int(value) for value in cem_iterations))
        if not iteration_values:
            raise ValueError("cem_iterations cannot be empty")
        invalid = sorted(set(iteration_values) - VALID_CEM_ITERATIONS)
        if invalid:
            raise ValueError(f"invalid CEM iterations: {invalid}")

    mask = np.isin(records.split_code, requested)
    if iteration_values is not None:
        mask &= np.isin(records.cem_iteration, iteration_values)
    selected = np.flatnonzero(mask)
    grouped: dict[tuple[str, str], list[int]] = {}
    original_ids: dict[tuple[str, str], Any] = {}
    for index in selected.tolist():
        key = _canonical_state_id(records.state_id[index])
        grouped.setdefault(key, []).append(index)
        original_ids.setdefault(key, records.state_id[index].item())

    result = []
    for key in sorted(grouped):
        indices = grouped[key]
        if len(indices) < min_candidates:
            continue
        # Ordering depends on record content, not dictionary/hash iteration.
        indices.sort(
            key=lambda index: (
                int(records.cem_iteration[index]),
                tuple(float(value) for value in records.action_params[index]),
                float(records.real_cost[index]),
                index,
            )
        )
        splits = np.unique(records.split_code[indices])
        if splits.size != 1:
            raise RuntimeError("validated state group unexpectedly crosses splits")
        result.append(
            StateGroup(
                state_id=original_ids[key],
                split_code=int(splits[0]),
                indices=np.asarray(indices, dtype=np.int64),
            )
        )
    return tuple(result)


def filter_state_groups(
    records: MPCGroundingRecords,
    *,
    split_codes: Iterable[int | str] = (TRAIN_SPLIT, VAL_SPLIT),
    cem_iterations: Iterable[int] | None = None,
    min_candidates: int = 1,
    allow_test: bool = False,
) -> MPCGroundingRecords:
    """Filter atomically by state and return groups in deterministic order."""

    groups = state_group_indices(
        records,
        split_codes=split_codes,
        cem_iterations=cem_iterations,
        min_candidates=min_candidates,
        allow_test=allow_test,
    )
    if not groups:
        raise ValueError("state-group filter selected no records")
    indices = np.concatenate([group.indices for group in groups])
    return records.subset(indices)


@dataclass(frozen=True)
class ZScoreStats:
    """Population z-score statistics fitted exclusively on split code 0."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    outcome_mean: np.ndarray
    outcome_std: np.ndarray
    cost_mean: float
    cost_std: float
    train_records: int

    def __post_init__(self) -> None:
        for name, expected in (
            ("state_mean", None),
            ("state_std", None),
            ("action_mean", 4),
            ("action_std", 4),
            ("outcome_mean", 4),
            ("outcome_std", 4),
        ):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or (expected is not None and value.shape != (expected,)):
                raise ValueError(f"{name} has invalid shape {value.shape}")
            if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite floating values")
            if name.endswith("_std") and np.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, _readonly_copy(value, dtype=np.float32))
        if self.state_mean.shape != self.state_std.shape:
            raise ValueError("state mean/std shapes differ")
        if not math.isfinite(float(self.cost_mean)) or not math.isfinite(
            float(self.cost_std)
        ):
            raise ValueError("cost statistics must be finite")
        if float(self.cost_std) <= 0.0:
            raise ValueError("cost_std must be strictly positive")
        if int(self.train_records) < 1:
            raise ValueError("train_records must be positive")

    @staticmethod
    def _normalize(value: Any, mean: np.ndarray | float, std: np.ndarray | float) -> Any:
        if torch.is_tensor(value):
            mean_tensor = torch.as_tensor(mean, device=value.device, dtype=value.dtype)
            std_tensor = torch.as_tensor(std, device=value.device, dtype=value.dtype)
            return (value - mean_tensor) / std_tensor
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError("values to normalize must be finite floating values")
        return (array - mean) / std

    def normalize_state(self, value: Any) -> Any:
        return self._normalize(value, self.state_mean, self.state_std)

    def normalize_action(self, value: Any) -> Any:
        return self._normalize(value, self.action_mean, self.action_std)

    def normalize_outcome(self, value: Any) -> Any:
        return self._normalize(value, self.outcome_mean, self.outcome_std)

    def normalize_cost(self, value: Any) -> Any:
        return self._normalize(value, float(self.cost_mean), float(self.cost_std))


def fit_train_zscore_stats(
    records: MPCGroundingRecords, *, eps: float = 1e-6
) -> ZScoreStats:
    """Fit all statistics from train rows, ignoring val/test values entirely."""

    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be a finite positive value")
    mask = records.split_code == TRAIN_SPLIT
    if not np.any(mask):
        raise ValueError("cannot fit z-score statistics without train records")

    def moments(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source = value[mask].astype(np.float64)
        mean = source.mean(axis=0)
        std = source.std(axis=0, ddof=0)
        return mean.astype(np.float32), np.maximum(std, eps).astype(np.float32)

    state_mean, state_std = moments(records.initial_features)
    action_mean, action_std = moments(records.action_params)
    outcome_mean, outcome_std = moments(records.outcome_features)
    cost_mean_array, cost_std_array = moments(records.real_cost)
    return ZScoreStats(
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        outcome_mean=outcome_mean,
        outcome_std=outcome_std,
        cost_mean=float(cost_mean_array),
        cost_std=float(cost_std_array),
        train_records=int(mask.sum()),
    )


class LatentDynamicsModel(nn.Module):
    """Small residual latent dynamics model with outcome and inverse heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        outcome_dim: int = 4,
        latent_dim: int = 16,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        dimensions = (state_dim, action_dim, outcome_dim, latent_dim, hidden)
        if any(isinstance(value, bool) or int(value) < 1 for value in dimensions):
            raise ValueError("all model dimensions must be positive integers")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.outcome_dim = int(outcome_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.latent_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.latent_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.latent_dim),
        )
        self.outcome_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.outcome_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.action_dim),
        )

    @staticmethod
    def _validate_input(name: str, value: torch.Tensor, width: int) -> None:
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != width:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise ValueError(f"{name} must be a tensor [B,{width}], got {shape}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    def forward(
        self, initial_features: torch.Tensor, action_params: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self._validate_input("initial_features", initial_features, self.state_dim)
        self._validate_input("action_params", action_params, self.action_dim)
        if initial_features.shape[0] != action_params.shape[0]:
            raise ValueError("state and action batch sizes differ")
        state_latent = self.state_encoder(initial_features)
        action_latent = self.action_encoder(action_params)
        delta_z = self.transition(torch.cat((state_latent, action_latent), dim=-1))
        next_latent = state_latent + delta_z
        predicted_outcome = self.outcome_decoder(next_latent)
        reconstructed_action = self.inverse_head(delta_z)
        return {
            "state_latent": state_latent,
            "action_latent": action_latent,
            "delta_z": delta_z,
            "latent_delta": delta_z,
            "next_latent": next_latent,
            "predicted_outcome": predicted_outcome,
            "outcome": predicted_outcome,
            "reconstructed_action": reconstructed_action,
            "inverse_action": reconstructed_action,
        }


class GroundedOutcomeModel(LatentDynamicsModel):
    """Runner-facing name/signature for :class:`LatentDynamicsModel`."""

    def __init__(
        self,
        initial_dim: int,
        action_dim: int = 4,
        outcome_dim: int = 4,
        latent_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(
            state_dim=initial_dim,
            action_dim=action_dim,
            outcome_dim=outcome_dim,
            latent_dim=latent_dim,
            hidden=hidden_dim,
        )
        self.initial_dim = self.state_dim
        self.hidden_dim = self.hidden


@dataclass(frozen=True)
class PhysicalCostWeights:
    """Authoritative V1 normalized physical-cost coefficients."""

    progress_weight: float = -0.20
    lateral_squared_weight: float = 1.50
    yaw_squared_weight: float = 0.80
    speed_squared_weight: float = 0.40
    steering_squared_weight: float = 0.02
    longitudinal_squared_weight: float = 0.01
    collision_weight: float = 10.0

    def as_tuple(self) -> tuple[float, ...]:
        values = (
            self.progress_weight,
            self.lateral_squared_weight,
            self.yaw_squared_weight,
            self.speed_squared_weight,
            self.steering_squared_weight,
            self.longitudinal_squared_weight,
            self.collision_weight,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("physical cost weights must be finite")
        if self.progress_weight > 0.0:
            raise ValueError("progress_weight must be non-positive")
        if any(float(value) < 0.0 for value in values[1:]):
            raise ValueError("error/action/collision cost weights must be non-negative")
        return tuple(float(value) for value in values)


def _cost_weights(
    value: PhysicalCostWeights | Mapping[str, float] | Sequence[float] | torch.Tensor,
) -> tuple[float, ...]:
    if isinstance(value, PhysicalCostWeights):
        return value.as_tuple()
    if isinstance(value, Mapping):
        allowed = {
            "progress_weight",
            "lateral_squared_weight",
            "yaw_squared_weight",
            "speed_squared_weight",
            "steering_squared_weight",
            "longitudinal_squared_weight",
            "collision_weight",
            # The full cost config may be passed directly; this threshold is
            # consumed by pair construction, not physical_cost.
            "pair_tie_threshold",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown physical cost weights: {sorted(unknown)}")
        required = allowed - {"pair_tie_threshold"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"physical cost config is missing weights: {missing}")
        return PhysicalCostWeights(
            **{key: float(value[key]) for key in required}
        ).as_tuple()
    if torch.is_tensor(value):
        if value.ndim != 1 or value.numel() != 7:
            raise ValueError("weight tensor must contain 7 values")
        if not torch.isfinite(value).all():
            raise ValueError("weight tensor contains non-finite values")
        values = tuple(float(item) for item in value.detach().cpu())
    else:
        values = tuple(float(item) for item in value)
    if len(values) != 7:
        raise ValueError("physical cost weights must contain 7 values")
    return PhysicalCostWeights(*values).as_tuple()


def physical_cost(
    outcome: torch.Tensor,
    action: torch.Tensor,
    collision: torch.Tensor,
    cost_config: PhysicalCostWeights
    | Mapping[str, float]
    | Sequence[float]
    | torch.Tensor,
) -> torch.Tensor:
    """Compute a differentiable normalized driving cost.

    This exactly follows ``configs/mpc_local_grounding_pilot_v1.yaml``:
    normalized progress is rewarded; lateral/yaw/speed errors, the two steering
    values, the two longitudinal values, and collision are separately weighted.
    """

    if not torch.is_tensor(outcome) or outcome.ndim < 1 or outcome.shape[-1] != 4:
        raise ValueError("outcome must be a tensor [...,4]")
    if not torch.is_tensor(action) or action.shape != outcome.shape[:-1] + (4,):
        raise ValueError("action must be a tensor with the same leading shape and [...,4]")
    if not torch.is_tensor(collision) or collision.shape != outcome.shape[:-1]:
        raise ValueError("collision must match outcome leading dimensions")
    if not outcome.is_floating_point() or not action.is_floating_point() or not collision.is_floating_point():
        raise TypeError("outcome, action, and collision must be floating point")
    if (
        not torch.isfinite(outcome).all()
        or not torch.isfinite(action).all()
        or not torch.isfinite(collision).all()
    ):
        raise ValueError("outcome/action/collision contains non-finite values")
    if torch.any(collision < 0.0):
        raise ValueError("collision values must be non-negative")
    (
        progress_weight,
        lateral_weight,
        yaw_weight,
        speed_weight,
        steering_weight,
        longitudinal_weight,
        collision_weight,
    ) = _cost_weights(cost_config)
    result = (
        progress_weight * outcome[..., 0]
        + lateral_weight * outcome[..., 1].square()
        + yaw_weight * outcome[..., 2].square()
        + speed_weight * outcome[..., 3].square()
    )
    result = result + steering_weight * torch.stack(
        (action[..., 0].square(), action[..., 2].square()), dim=-1
    ).mean(dim=-1)
    result = result + longitudinal_weight * torch.stack(
        (action[..., 1].square(), action[..., 3].square()), dim=-1
    ).mean(dim=-1)
    result = result + collision_weight * collision
    return result


@dataclass(frozen=True)
class RankPairs:
    """Unordered within-state pairs and probability that left is better."""

    left_indices: np.ndarray
    right_indices: np.ndarray
    targets: np.ndarray
    state_ids: tuple[Any, ...]

    def __post_init__(self) -> None:
        left = np.asarray(self.left_indices)
        right = np.asarray(self.right_indices)
        targets = np.asarray(self.targets)
        if left.ndim != 1 or right.shape != left.shape or targets.shape != left.shape:
            raise ValueError("pair arrays must have the same one-dimensional shape")
        if left.size == 0:
            raise ValueError("RankPairs cannot be empty")
        if left.dtype.kind not in "iu" or right.dtype.kind not in "iu":
            raise TypeError("pair indices must be integers")
        if not np.issubdtype(targets.dtype, np.floating) or not np.isfinite(targets).all():
            raise TypeError("pair targets must be finite floating values")
        if np.any(left < 0) or np.any(right < 0) or np.any(left == right):
            raise ValueError("pair indices must be non-negative and distinct")
        if not np.isin(targets, (0.0, 0.5, 1.0)).all():
            raise ValueError("pair targets must be 0, 0.5, or 1")
        if len(self.state_ids) != left.size:
            raise ValueError("state_ids length differs from pair count")
        object.__setattr__(self, "left_indices", _readonly_copy(left, dtype=np.int64))
        object.__setattr__(self, "right_indices", _readonly_copy(right, dtype=np.int64))
        object.__setattr__(self, "targets", _readonly_copy(targets, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.left_indices.size)


def _state_seed(seed: int, state_id: Any) -> int:
    key = _canonical_state_id(state_id)
    digest = hashlib.sha256(f"{int(seed)}|{key[0]}|{key[1]}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_within_state_pairs(
    records: MPCGroundingRecords,
    *,
    budget_per_state: int = 64,
    seed: int,
    split_code: int | str = TRAIN_SPLIT,
    cem_iterations: Iterable[int] | None = None,
    tie_tolerance: float = 1e-6,
    allow_test: bool = False,
) -> RankPairs:
    """Build a deterministic per-state pair budget without cross-state pairs.

    The target is ``1`` when the left real cost is lower, ``0`` when the right
    cost is lower, and ``0.5`` for a tie within ``tie_tolerance``.  Per-state
    pair lists are seeded independently and each state contributes at most
    ``budget_per_state`` pairs.  The authoritative V1 budget is 64 per state.
    """

    if isinstance(budget_per_state, bool) or int(budget_per_state) < 1:
        raise ValueError("pair budget per state must be a positive integer")
    budget_per_state = int(budget_per_state)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("pair seed must be an integer")
    if not math.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    code = _normalize_split_code(split_code)
    groups = state_group_indices(
        records,
        split_codes=(code,),
        cem_iterations=cem_iterations,
        min_candidates=2,
        allow_test=allow_test,
    )
    if not groups:
        raise ValueError("no state has at least two candidates after filtering")

    selected: list[tuple[int, int, float, Any]] = []
    for group in groups:
        candidates = []
        indices = group.indices.tolist()
        for position, left in enumerate(indices[:-1]):
            for right in indices[position + 1 :]:
                difference = float(records.real_cost[left] - records.real_cost[right])
                if abs(difference) <= tie_tolerance:
                    target = 0.5
                elif difference < 0.0:
                    target = 1.0
                else:
                    target = 0.0
                candidates.append((left, right, target, group.state_id))
        rng = np.random.default_rng(_state_seed(int(seed), group.state_id))
        order = rng.permutation(len(candidates))
        selected.extend(
            candidates[int(index)] for index in order[:budget_per_state]
        )
    if not selected:
        raise ValueError("no pairs were available")

    left = np.asarray([value[0] for value in selected], dtype=np.int64)
    right = np.asarray([value[1] for value in selected], dtype=np.int64)
    state_ids = tuple(value[3] for value in selected)
    for pair_index, (left_index, right_index) in enumerate(zip(left, right, strict=True)):
        if _canonical_state_id(records.state_id[left_index]) != _canonical_state_id(
            records.state_id[right_index]
        ):
            raise RuntimeError(f"cross-state pair generated at position {pair_index}")
    return RankPairs(
        left_indices=left,
        right_indices=right,
        targets=np.asarray([value[2] for value in selected], dtype=np.float32),
        state_ids=state_ids,
    )


def pairwise_logistic_rank_loss(
    predicted_cost: torch.Tensor,
    pairs: RankPairs,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary-logistic ranking loss with soft target 0.5 for real-cost ties."""

    if not torch.is_tensor(predicted_cost) or predicted_cost.ndim != 1:
        raise ValueError("predicted_cost must be a one-dimensional tensor")
    if not predicted_cost.is_floating_point():
        raise TypeError("predicted_cost must be floating point")
    if not torch.isfinite(predicted_cost).all():
        raise ValueError("predicted_cost contains non-finite values")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")
    # RankPairs deliberately stores immutable NumPy arrays.  Construct owned
    # tensors instead of sharing that read-only storage (PyTorch warns because
    # writes through a shared tensor would otherwise be undefined behavior).
    left = torch.tensor(
        pairs.left_indices.tolist(), dtype=torch.long, device=predicted_cost.device
    )
    right = torch.tensor(
        pairs.right_indices.tolist(), dtype=torch.long, device=predicted_cost.device
    )
    if int(torch.max(torch.cat((left, right)))) >= predicted_cost.numel():
        raise IndexError("rank pair index exceeds predicted_cost length")
    target = torch.tensor(
        pairs.targets.tolist(),
        dtype=predicted_cost.dtype,
        device=predicted_cost.device,
    )
    # Positive logit means left has a lower (better) predicted cost.
    logits = predicted_cost[right] - predicted_cost[left]
    return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)


def tie_aware_logistic_rank_loss(
    pred_cost_i: torch.Tensor,
    pred_cost_j: torch.Tensor,
    true_cost_i: torch.Tensor,
    true_cost_j: torch.Tensor,
    tie_threshold: float,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Runner-facing pair loss with target 0.5 for real-cost ties.

    A positive logit means candidate ``i`` has lower predicted cost than
    candidate ``j``.  Exact/near real-cost ties therefore train toward equal
    predicted costs instead of forcing arbitrary separation.  ``temperature``
    scales the predicted-cost difference before binary cross entropy; its
    default preserves the original unscaled loss.
    """

    tensors = (pred_cost_i, pred_cost_j, true_cost_i, true_cost_j)
    if any(not torch.is_tensor(value) for value in tensors):
        raise TypeError("all rank-loss costs must be tensors")
    if any(value.shape != pred_cost_i.shape for value in tensors[1:]):
        raise ValueError("all rank-loss costs must have identical shapes")
    if pred_cost_i.numel() == 0:
        raise ValueError("rank-loss costs cannot be empty")
    if any(not value.is_floating_point() for value in tensors):
        raise TypeError("all rank-loss costs must be floating point")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("rank-loss costs contain non-finite values")
    if not math.isfinite(float(tie_threshold)) or float(tie_threshold) < 0.0:
        raise ValueError("tie_threshold must be finite and non-negative")
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")

    true_difference = true_cost_i - true_cost_j
    ties = true_difference.abs() <= float(tie_threshold)
    target = torch.where(
        ties,
        torch.full_like(true_difference, 0.5),
        (true_difference < 0.0).to(dtype=true_difference.dtype),
    )
    logits = (pred_cost_j - pred_cost_i) / temperature_value
    return F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)


def _average_ranks(values: np.ndarray, tolerance: float) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        anchor = float(values[order[start]])
        while end < len(order) and abs(float(values[order[end]]) - anchor) <= tolerance:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        result[order[start:end]] = average_rank
        start = end
    return result


def _spearman(values: np.ndarray, targets: np.ndarray, tolerance: float) -> float:
    left = _average_ranks(values, tolerance)
    right = _average_ranks(targets, tolerance)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    # Two constant rankings contain no ordering information.  Treating that
    # degenerate case as perfect agreement can make an action-insensitive model
    # pass a planning-alignment gate, so use the conservative neutral score.
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 0.0
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / (left_norm * right_norm))


def _tie_aware_pair_accuracy(
    predicted: np.ndarray, real: np.ndarray, tolerance: float
) -> tuple[float, int]:
    scores = []
    for left in range(len(real) - 1):
        for right in range(left + 1, len(real)):
            real_difference = float(real[left] - real[right])
            predicted_difference = float(predicted[left] - predicted[right])
            real_tie = abs(real_difference) <= tolerance
            predicted_tie = abs(predicted_difference) <= tolerance
            if real_tie:
                scores.append(1.0 if predicted_tie else 0.0)
            elif predicted_tie:
                scores.append(0.5)
            else:
                scores.append(
                    1.0 if np.sign(real_difference) == np.sign(predicted_difference) else 0.0
                )
    if not scores:
        raise ValueError("tie-aware accuracy requires at least two candidates")
    return float(np.mean(scores)), len(scores)


def state_macro_metrics(
    records: MPCGroundingRecords,
    predicted_cost: np.ndarray | torch.Tensor,
    predicted_outcome: np.ndarray | torch.Tensor,
    *,
    split_code: int | str = VAL_SPLIT,
    cem_iterations: Iterable[int] | None = None,
    tie_tolerance: float = 1e-6,
    allow_test: bool = False,
) -> dict[str, float | int]:
    """Compute equally state-weighted ranking, regret, and outcome metrics."""

    if not math.isfinite(float(tie_tolerance)) or float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    if torch.is_tensor(predicted_cost):
        cost = predicted_cost.detach().cpu().numpy()
    else:
        cost = np.asarray(predicted_cost)
    if torch.is_tensor(predicted_outcome):
        outcome = predicted_outcome.detach().cpu().numpy()
    else:
        outcome = np.asarray(predicted_outcome)
    _require_float_array("predicted_cost", cost, (len(records),))
    _require_float_array("predicted_outcome", outcome, (len(records), 4))

    code = _normalize_split_code(split_code)
    groups = state_group_indices(
        records,
        split_codes=(code,),
        cem_iterations=cem_iterations,
        min_candidates=1,
        allow_test=allow_test,
    )
    if not groups:
        raise ValueError("metric filter selected no states")

    spearman_values = []
    accuracy_values = []
    regret_values = []
    outcome_values = []
    pair_count = 0
    for group in groups:
        indices = group.indices
        predicted_group = cost[indices].astype(np.float64)
        real_group = records.real_cost[indices].astype(np.float64)
        outcome_error = (
            outcome[indices].astype(np.float64)
            - records.outcome_features[indices].astype(np.float64)
        ) ** 2
        outcome_values.append(float(outcome_error.mean()))

        minimum_prediction = float(predicted_group.min())
        selected = np.abs(predicted_group - minimum_prediction) <= tie_tolerance
        selected_real = float(real_group[selected].mean())
        regret_values.append(max(0.0, selected_real - float(real_group.min())))

        if len(indices) >= 2:
            spearman_values.append(
                _spearman(predicted_group, real_group, float(tie_tolerance))
            )
            accuracy, pairs = _tie_aware_pair_accuracy(
                predicted_group, real_group, float(tie_tolerance)
            )
            accuracy_values.append(accuracy)
            pair_count += pairs

    if not spearman_values or not accuracy_values:
        raise ValueError("ranking metrics require at least one state with two candidates")
    result: dict[str, float | int] = {
        "state_macro_spearman": float(np.mean(spearman_values)),
        "state_macro_tie_accuracy": float(np.mean(accuracy_values)),
        "state_macro_selection_regret": float(np.mean(regret_values)),
        "state_macro_outcome_mse": float(np.mean(outcome_values)),
        "states": len(groups),
        "ranking_states": len(spearman_values),
        "pairs": pair_count,
    }
    if not all(
        math.isfinite(float(value)) for value in result.values() if isinstance(value, float)
    ):
        raise RuntimeError("state-macro metric computation produced non-finite output")
    return result


def statewise_softmin_listwise_loss(
    predicted_cost: torch.Tensor,
    true_cost: torch.Tensor,
    candidate_groups: Sequence[Sequence[int] | np.ndarray],
    *,
    temperature: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Decision-aligned top-one listwise loss, averaged equally by state.

    Each supplied group is one same-initial-state candidate set.  Its target is
    the deterministic exact real-cost argmin (the first minimum in the supplied
    order), while the model distribution is a softmin over predicted physical
    costs.  This makes the supervised target match the exact-argmin decision
    used by :func:`decision_state_macro_metrics`; unlike pair sampling, every
    candidate in a queried state participates in one state-macro loss term.
    """

    if not torch.is_tensor(predicted_cost) or not torch.is_tensor(true_cost):
        raise TypeError("predicted_cost and true_cost must be tensors")
    if predicted_cost.ndim != 1 or true_cost.shape != predicted_cost.shape:
        raise ValueError("predicted_cost and true_cost must have identical [N] shape")
    if not predicted_cost.is_floating_point() or not true_cost.is_floating_point():
        raise TypeError("predicted_cost and true_cost must be floating point")
    if predicted_cost.device != true_cost.device:
        raise ValueError("predicted_cost and true_cost must be on the same device")
    if not torch.isfinite(predicted_cost).all() or not torch.isfinite(true_cost).all():
        raise ValueError("predicted_cost or true_cost contains non-finite values")
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")

    losses: list[torch.Tensor] = []
    for group_position, raw_indices in enumerate(candidate_groups):
        indices = np.asarray(raw_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError(
                f"candidate group {group_position} must be a one-dimensional integer array"
            )
        indices = indices.astype(np.int64, copy=False)
        if indices.size < 2:
            raise ValueError("each candidate group must contain at least two candidates")
        if len(set(indices.tolist())) != int(indices.size):
            raise ValueError(f"candidate group {group_position} contains duplicate indices")
        if np.any(indices < 0) or np.any(indices >= predicted_cost.numel()):
            raise IndexError(f"candidate group {group_position} index is out of bounds")
        selected = torch.tensor(
            indices.tolist(), dtype=torch.long, device=predicted_cost.device
        )
        group_true = true_cost[selected]
        # torch.argmin returns the first minimum.  The runner supplies a stable
        # SHA-ordered full candidate list, so exact real ties are reproducible.
        target_position = torch.argmin(group_true).reshape(1)
        logits = (-predicted_cost[selected] / temperature_value).reshape(1, -1)
        losses.append(F.cross_entropy(logits, target_position, reduction="mean"))
    if not losses:
        raise ValueError("candidate_groups cannot be empty")
    stacked = torch.stack(losses)
    if reduction == "none":
        return stacked
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


def decision_state_macro_metrics(
    records: MPCGroundingRecords,
    predicted_cost: np.ndarray | torch.Tensor,
    *,
    split_code: int | str = VAL_SPLIT,
    cem_iterations: Iterable[int] | None = None,
    epsilon: float,
    predicted_tie_tolerance: float = 0.0,
    oracle_tie_tolerance: float = 0.0,
    predicted_outcome: np.ndarray | torch.Tensor | None = None,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Evaluate the deterministic exact-argmin decision equally by state.

    The primary metric is exact-argmin simple regret.  Epsilon-regret is the
    secondary, explicitly tolerance-adjusted value ``max(0, regret-epsilon)``;
    it never changes which action is selected.  Per-state selected indices and
    predicted/oracle tie counts make tie-breaking auditable.
    """

    for name, value in (
        ("epsilon", epsilon),
        ("predicted_tie_tolerance", predicted_tie_tolerance),
        ("oracle_tie_tolerance", oracle_tie_tolerance),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite and non-negative") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    epsilon_value = float(epsilon)
    predicted_tolerance = float(predicted_tie_tolerance)
    oracle_tolerance = float(oracle_tie_tolerance)

    if torch.is_tensor(predicted_cost):
        cost = predicted_cost.detach().cpu().numpy()
    else:
        cost = np.asarray(predicted_cost)
    _require_float_array("predicted_cost", cost, (len(records),))

    outcome: np.ndarray | None = None
    if predicted_outcome is not None:
        if torch.is_tensor(predicted_outcome):
            outcome = predicted_outcome.detach().cpu().numpy()
        else:
            outcome = np.asarray(predicted_outcome)
        _require_float_array("predicted_outcome", outcome, (len(records), 4))

    code = _normalize_split_code(split_code)
    groups = state_group_indices(
        records,
        split_codes=(code,),
        cem_iterations=cem_iterations,
        min_candidates=2,
        allow_test=allow_test,
    )
    if not groups:
        raise ValueError("decision metric filter selected no states")

    details: list[dict[str, Any]] = []
    exact_regrets: list[float] = []
    epsilon_regrets: list[float] = []
    epsilon_successes: list[float] = []
    exact_top1_values: list[float] = []
    spearman_values: list[float] = []
    outcome_errors: list[float] = []
    for group in groups:
        # The V3 protocol resolves an exact predicted/real tie by the SHA-256
        # of canonical raw action bytes, not by NPZ row position.
        indices = np.asarray(
            sorted(
                group.indices.tolist(),
                key=lambda index: (
                    hashlib.sha256(
                        np.asarray(records.action_params[index], dtype="<f4").tobytes()
                    ).digest(),
                    int(index),
                ),
            ),
            dtype=np.int64,
        )
        predicted_group = cost[indices].astype(np.float64)
        real_group = records.real_cost[indices].astype(np.float64)
        selected_position = int(np.argmin(predicted_group))
        selected_index = int(indices[selected_position])
        predicted_minimum = float(predicted_group[selected_position])
        oracle_minimum = float(np.min(real_group))
        raw_regret = float(real_group[selected_position] - oracle_minimum)
        # A tiny negative can only be floating roundoff because oracle_minimum
        # comes from this exact group.
        exact_regret = max(0.0, raw_regret)
        epsilon_regret = max(0.0, exact_regret - epsilon_value)
        predicted_ties = np.flatnonzero(
            np.abs(predicted_group - predicted_minimum) <= predicted_tolerance
        )
        oracle_ties = np.flatnonzero(
            np.abs(real_group - oracle_minimum) <= oracle_tolerance
        )
        # Exact real ties have the same raw-action SHA ordering as predicted
        # ties, so the first oracle position is the single frozen top-1 target.
        exact_top1 = float(selected_position == int(oracle_ties[0]))
        sorted_real = np.sort(real_group)
        best_second_gap = float(sorted_real[1] - sorted_real[0])
        spearman = _spearman(predicted_group, real_group, oracle_tolerance)
        detail: dict[str, Any] = {
            "state_id": str(group.state_id),
            "candidate_count": int(len(indices)),
            "selected_index": selected_index,
            "selected_candidate_position": selected_position,
            "selected_cem_iteration": int(records.cem_iteration[selected_index]),
            "selected_action": records.action_params[selected_index].tolist(),
            "selected_predicted_cost": predicted_minimum,
            "selected_real_cost": float(real_group[selected_position]),
            "oracle_real_cost": oracle_minimum,
            "oracle_indices": [int(indices[position]) for position in oracle_ties],
            "exact_argmin_simple_regret": exact_regret,
            "epsilon_regret": epsilon_regret,
            "within_epsilon": bool(exact_regret <= epsilon_value),
            "exact_top1_correct": bool(exact_top1),
            "spearman": spearman,
            "real_cost_range": float(np.max(real_group) - oracle_minimum),
            "best_second_gap": best_second_gap,
            "predicted_argmin_tie_count": int(len(predicted_ties)),
            "predicted_argmin_tied_indices": [
                int(indices[position]) for position in predicted_ties
            ],
            "oracle_argmin_tie_count": int(len(oracle_ties)),
            "deterministic_tie_break_applied": bool(len(predicted_ties) > 1),
        }
        if outcome is not None:
            error = (
                outcome[indices].astype(np.float64)
                - records.outcome_features[indices].astype(np.float64)
            ) ** 2
            state_mse = float(np.mean(error))
            detail["outcome_mse"] = state_mse
            outcome_errors.append(state_mse)
        details.append(detail)
        exact_regrets.append(exact_regret)
        epsilon_regrets.append(epsilon_regret)
        epsilon_successes.append(float(exact_regret <= epsilon_value))
        exact_top1_values.append(exact_top1)
        spearman_values.append(spearman)

    result: dict[str, Any] = {
        "decision_rule": "deterministic_exact_predicted_argmin",
        "primary_metric": "state_macro_exact_argmin_simple_regret",
        "secondary_metric": "state_macro_epsilon_regret",
        "epsilon": epsilon_value,
        "predicted_tie_tolerance": predicted_tolerance,
        "oracle_tie_tolerance": oracle_tolerance,
        "state_macro_exact_argmin_simple_regret": float(np.mean(exact_regrets)),
        "state_macro_epsilon_regret": float(np.mean(epsilon_regrets)),
        "epsilon_success_rate": float(np.mean(epsilon_successes)),
        "exact_top1_accuracy": float(np.mean(exact_top1_values)),
        "state_macro_spearman": float(np.mean(spearman_values)),
        "state_macro_predicted_argmin_tie_count": float(
            np.mean([item["predicted_argmin_tie_count"] for item in details])
        ),
        "states_with_predicted_argmin_ties": int(
            sum(item["deterministic_tie_break_applied"] for item in details)
        ),
        "states": len(details),
        "state_details": details,
    }
    if outcome is not None:
        result["state_macro_outcome_mse"] = float(np.mean(outcome_errors))
    if not all(
        math.isfinite(float(value))
        for key, value in result.items()
        if key not in {"state_details", "decision_rule", "primary_metric", "secondary_metric"}
        and isinstance(value, (float, int))
    ):
        raise RuntimeError("decision metric computation produced non-finite output")
    return result


__all__ = [
    "TRAIN_SPLIT",
    "VAL_SPLIT",
    "TEST_SPLIT",
    "SPLIT_NAME_TO_CODE",
    "VALID_CEM_ITERATIONS",
    "MPCGroundingRecords",
    "StateGroup",
    "ZScoreStats",
    "RankPairs",
    "PhysicalCostWeights",
    "LatentDynamicsModel",
    "GroundedOutcomeModel",
    "load_mpc_records",
    "state_group_indices",
    "filter_state_groups",
    "fit_train_zscore_stats",
    "physical_cost",
    "build_within_state_pairs",
    "pairwise_logistic_rank_loss",
    "tie_aware_logistic_rank_loss",
    "state_macro_metrics",
]
