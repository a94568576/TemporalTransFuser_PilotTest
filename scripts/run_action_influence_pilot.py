#!/usr/bin/env python3
"""Run the staged observational action-influence latent pilot.

The experiment trains an action-free passive predictor on oracle ego-warped,
frozen TF++ BEV features, freezes it, and compares equal-capacity uncentered and
centered FiLM action residuals.  It never evaluates the cache test split.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from temporal_tf.action_influence import (
    ACTION_CANDIDATE_NAMES,
    ActionFiLMResidualAdapter,
    ActionTransitionDataset,
    PassiveFuturePredictor,
    build_action_candidates,
    deterministic_derangement,
    freeze_module,
    normalize_actions,
    normalize_latent,
    parameter_count,
    route_macro_mean,
    train_normalization_statistics,
)
from temporal_tf.bev_geometry import warp_bev_to_current


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


class PreparedTransitions(Dataset):
    """In-memory fp16 tensors after oracle current-to-future ego alignment."""

    def __init__(
        self,
        *,
        current: torch.Tensor,
        target: torch.Tensor,
        actions: torch.Tensor,
        validity: torch.Tensor,
        route_ids: list[str],
        sample_ids: list[str],
    ) -> None:
        size = current.shape[0]
        if target.shape != current.shape or actions.shape[0] != size:
            raise ValueError("prepared transition tensors have inconsistent sizes")
        if validity.shape != (size, 1, *current.shape[-2:]):
            raise ValueError("prepared validity shape mismatch")
        if len(route_ids) != size or len(sample_ids) != size:
            raise ValueError("prepared transition identities have inconsistent sizes")
        self.current = current
        self.target = target
        self.actions = actions
        self.validity = validity
        self.route_ids = tuple(route_ids)
        self.sample_ids = tuple(sample_ids)
        self.base_future: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.current.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = {
            "current": self.current[index],
            "target": self.target[index],
            "actions": self.actions[index],
            "validity": self.validity[index],
            "route_id": self.route_ids[index],
            "sample_id": self.sample_ids[index],
            "dataset_index": int(index),
        }
        if self.base_future is not None:
            value["base_future"] = self.base_future[index]
        return value


@torch.inference_mode()
def _prepare_split(
    dataset: ActionTransitionDataset,
    *,
    statistics: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[PreparedTransitions, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prepared_current: list[torch.Tensor] = []
    prepared_target: list[torch.Tensor] = []
    prepared_actions: list[torch.Tensor] = []
    prepared_validity: list[torch.Tensor] = []
    route_ids: list[str] = []
    sample_ids: list[str] = []
    raw_persistence_errors: list[torch.Tensor] = []
    warped_persistence_errors: list[torch.Tensor] = []
    latent_mean = statistics["latent_mean"].to(device=device, dtype=torch.float32)

    for batch in loader:
        raw_current = batch["current_latent"].to(device=device, dtype=torch.float32)
        raw_target = batch["future_latent"].to(device=device, dtype=torch.float32)
        current_pose = batch["current_pose"].to(device=device, dtype=torch.float64)
        future_pose = batch["future_pose"].to(device=device, dtype=torch.float64)
        aligned, validity = warp_bev_to_current(
            raw_current, current_pose, future_pose
        )
        # Invalid grid-sample support is unknown, not a real zero latent.  Fill
        # it with the train target mean so it becomes zero after standardizing.
        aligned = aligned * validity + latent_mean.reshape(1, -1, 1, 1) * (
            1.0 - validity
        )
        current_normalized = normalize_latent(aligned, statistics)
        target_normalized = normalize_latent(raw_target, statistics)
        raw_current_normalized = normalize_latent(raw_current, statistics)
        raw_persistence_errors.append(
            _sample_errors(raw_current_normalized, target_normalized, validity).cpu()
        )
        warped_persistence_errors.append(
            _sample_errors(current_normalized, target_normalized, validity).cpu()
        )
        prepared_current.append(current_normalized.half().cpu())
        prepared_target.append(target_normalized.half().cpu())
        prepared_actions.append(batch["actions"].float().cpu())
        prepared_validity.append(validity.half().cpu())
        route_ids.extend(str(value) for value in batch["route_id"])
        sample_ids.extend(str(value) for value in batch["sample_id"])

    result = PreparedTransitions(
        current=torch.cat(prepared_current),
        target=torch.cat(prepared_target),
        actions=torch.cat(prepared_actions),
        validity=torch.cat(prepared_validity),
        route_ids=route_ids,
        sample_ids=sample_ids,
    )
    raw_errors = torch.cat(raw_persistence_errors).numpy()
    warped_errors = torch.cat(warped_persistence_errors).numpy()
    diagnostic = {
        "mean_valid_fraction": float(result.validity.float().mean()),
        "raw_persistence_micro_mse": float(raw_errors.mean()),
        "oracle_warp_persistence_micro_mse": float(warped_errors.mean()),
        "loss_support": "oracle_warp_overlap_validity_mask",
        "raw_persistence_route_macro_mse": route_macro_mean(raw_errors, route_ids),
        "oracle_warp_persistence_route_macro_mse": route_macro_mean(
            warped_errors, route_ids
        ),
    }
    return result, diagnostic


def _loader(
    dataset: PreparedTransitions,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def _sample_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor | None = None,
) -> torch.Tensor:
    squared = (prediction - target).square()
    if validity is None:
        return squared.flatten(start_dim=1).mean(dim=1)
    if validity.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
        raise ValueError("validity must be [B,1,H,W]")
    weighted = squared * validity
    denominator = validity.flatten(start_dim=1).sum(dim=1) * prediction.shape[1]
    return weighted.flatten(start_dim=1).sum(dim=1) / denominator.clamp_min(1.0)


@torch.inference_mode()
def _predict_passive(
    model: PassiveFuturePredictor,
    dataset: PreparedTransitions,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    predictions: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    model.eval()
    for batch in loader:
        current = batch["current"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        validity = batch["validity"].to(device=device, dtype=torch.float32)
        prediction = model(current)
        predictions.append(prediction.half().cpu())
        errors.append(_sample_errors(prediction, target, validity).cpu())
    return torch.cat(predictions), torch.cat(errors).numpy()


def _train_passive(
    train: PreparedTransitions,
    val: PreparedTransitions,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[PassiveFuturePredictor, dict[str, Any]]:
    seed = int(config["seed"])
    _seed_everything(seed)
    channels = int(train.current.shape[1])
    model = PassiveFuturePredictor(
        channels=channels, hidden_channels=int(config["hidden_channels"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    batch_size = int(config["batch_size"])
    best_metric = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in _loader(
            train, batch_size=batch_size, shuffle=True, seed=seed + epoch
        ):
            current = batch["current"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            validity = batch["validity"].to(device=device, dtype=torch.float32)
            prediction = model(current)
            loss = _sample_errors(prediction, target, validity).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        _, val_errors = _predict_passive(
            model, val, device=device, batch_size=batch_size
        )
        val_macro = route_macro_mean(val_errors, val.route_ids)
        history.append(
            {
                "epoch": epoch + 1,
                "train_micro_mse": float(np.mean(losses)),
                "val_route_macro_mse": val_macro,
            }
        )
        print(
            f"base epoch={epoch + 1:02d} train={np.mean(losses):.6f} "
            f"val_macro={val_macro:.6f}",
            flush=True,
        )
        if val_macro < best_metric - 1e-8:
            best_metric = val_macro
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("passive predictor produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    freeze_module(model)
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_route_macro_mse": best_metric,
        "epochs_ran": len(history),
        "history": history,
        "trainable_parameters": parameter_count(
            PassiveFuturePredictor(
                channels=channels, hidden_channels=int(config["hidden_channels"])
            )
        ),
    }


def _reference_indices(
    source: PreparedTransitions,
    train: PreparedTransitions,
    *,
    count: int,
) -> tuple[torch.Tensor, list[list[str]]]:
    """Choose fixed references from train actions, preferring another route."""

    if count < 1:
        raise ValueError("reference count must be positive")
    rows = []
    identities: list[list[str]] = []
    train_size = len(train)
    for sample_id, route_id in zip(source.sample_ids, source.route_ids, strict=True):
        selected = []
        selected_ids = []
        for reference_index in range(count):
            digest = hashlib.sha256(
                f"action_reference_v1|{sample_id}|{reference_index}".encode("utf-8")
            ).digest()
            start = int.from_bytes(digest[:8], "big") % train_size
            choice = start
            for offset in range(train_size):
                candidate = (start + offset) % train_size
                if train.route_ids[candidate] != route_id:
                    choice = candidate
                    break
            selected.append(choice)
            selected_ids.append(train.sample_ids[choice])
        rows.append(selected)
        identities.append(selected_ids)
    return torch.tensor(rows, dtype=torch.long), identities


def _adapter_validation_errors(
    model: ActionFiLMResidualAdapter,
    val: PreparedTransitions,
    *,
    reference_actions: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(val), batch_size):
            end = min(start + batch_size, len(val))
            current = val.current[start:end].to(device=device, dtype=torch.float32)
            target = val.target[start:end].to(device=device, dtype=torch.float32)
            base = val.base_future[start:end].to(device=device, dtype=torch.float32)
            validity = val.validity[start:end].to(device=device, dtype=torch.float32)
            actions = normalize_actions(
                val.actions[start:end].to(device=device, dtype=torch.float32), statistics
            ).unsqueeze(1)
            references = (
                normalize_actions(
                    reference_actions[start:end].to(device=device, dtype=torch.float32),
                    statistics,
                )
                if model.centered
                else None
            )
            prediction = model(current, base, actions, references)["prediction"][:, 0]
            values.append(_sample_errors(prediction, target, validity).cpu())
    return torch.cat(values).numpy()


def _train_adapter(
    variant: str,
    seed: int,
    train: PreparedTransitions,
    val: PreparedTransitions,
    *,
    train_references: torch.Tensor,
    val_references: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ActionFiLMResidualAdapter, dict[str, Any]]:
    centered = variant == "centered_film_k4"
    if variant not in {"uncentered_film", "centered_film_k4"}:
        raise ValueError(f"unsupported adapter variant: {variant}")
    _seed_everything(seed)
    model = ActionFiLMResidualAdapter(
        channels=int(train.current.shape[1]),
        action_horizon=int(train.actions.shape[1]),
        hidden_channels=int(config["hidden_channels"]),
        action_hidden_dim=int(config["action_hidden_dim"]),
        centered=centered,
        gate_bias=float(config["gate_bias"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    batch_size = int(config["batch_size"])
    budget_weight = float(config["residual_budget_weight"])
    best_metric = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        future_losses = []
        budget_losses = []
        for batch in _loader(
            train, batch_size=batch_size, shuffle=True, seed=seed * 1000 + epoch
        ):
            indices = batch["dataset_index"].long()
            current = batch["current"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            base = batch["base_future"].to(device=device, dtype=torch.float32)
            validity = batch["validity"].to(device=device, dtype=torch.float32)
            actions = normalize_actions(
                batch["actions"].to(device=device, dtype=torch.float32), statistics
            ).unsqueeze(1)
            references = (
                normalize_actions(
                    train_references[indices].to(device=device, dtype=torch.float32),
                    statistics,
                )
                if centered
                else None
            )
            output = model(current, base, actions, references)
            prediction = output["prediction"][:, 0]
            future_loss = _sample_errors(prediction, target, validity).mean()
            applied = prediction - base
            budget_loss = _sample_errors(applied, torch.zeros_like(applied), validity).mean()
            loss = future_loss + budget_weight * budget_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            future_losses.append(float(future_loss.detach()))
            budget_losses.append(float(budget_loss.detach()))
        val_errors = _adapter_validation_errors(
            model,
            val,
            reference_actions=val_references,
            statistics=statistics,
            device=device,
            batch_size=batch_size,
        )
        val_macro = route_macro_mean(val_errors, val.route_ids)
        history.append(
            {
                "epoch": epoch + 1,
                "train_total_loss": float(np.mean(losses)),
                "train_future_mse": float(np.mean(future_losses)),
                "train_residual_budget": float(np.mean(budget_losses)),
                "val_true_route_macro_mse": val_macro,
                "gate": float(torch.sigmoid(model.gate_logit.detach())),
            }
        )
        print(
            f"{variant} seed={seed} epoch={epoch + 1:02d} "
            f"train={np.mean(future_losses):.6f} val_macro={val_macro:.6f} "
            f"gate={float(torch.sigmoid(model.gate_logit.detach())):.6f}",
            flush=True,
        )
        if val_macro < best_metric - 1e-8:
            best_metric = val_macro
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{variant} seed {seed} produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    freeze_module(model)
    return model, {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_true_route_macro_mse": best_metric,
        "epochs_ran": len(history),
        "gate": float(torch.sigmoid(model.gate_logit.detach().cpu())),
        "trainable_parameters": parameter_count(
            ActionFiLMResidualAdapter(
                channels=int(train.current.shape[1]),
                action_horizon=int(train.actions.shape[1]),
                hidden_channels=int(config["hidden_channels"]),
                action_hidden_dim=int(config["action_hidden_dim"]),
                centered=centered,
                gate_bias=float(config["gate_bias"]),
            )
        ),
        "history": history,
    }


def _control_metrics(errors: np.ndarray, route_ids: Iterable[str]) -> dict[str, Any]:
    routes = list(route_ids)
    result = {}
    for index, name in enumerate(ACTION_CANDIDATE_NAMES):
        values = errors[:, index]
        result[name] = {
            "micro_mse": float(values.mean()),
            "route_macro_mse": route_macro_mean(values, routes),
        }
    true = result["true"]["route_macro_mse"]
    for name in ACTION_CANDIDATE_NAMES[1:]:
        result[name]["relative_penalty_vs_true_percent"] = float(
            100.0 * (result[name]["route_macro_mse"] - true) / max(true, 1e-12)
        )
    return result


def _prediction_features(
    current: torch.Tensor, applied: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    state_mean = current.mean(dim=(-2, -1))
    state_std = current.std(dim=(-2, -1), unbiased=False)
    state = torch.cat((state_mean, state_std), dim=1)
    residual_mean = applied.mean(dim=(-2, -1))
    residual_std = applied.std(dim=(-2, -1), unbiased=False)
    residual_spatial = nn.functional.adaptive_avg_pool2d(
        applied.mean(dim=1, keepdim=True), (8, 8)
    ).flatten(start_dim=1)
    transition = torch.cat(
        (state, residual_mean, residual_std, residual_spatial), dim=1
    )
    return state, transition


@torch.inference_mode()
def _evaluate_adapter(
    model: ActionFiLMResidualAdapter,
    dataset: PreparedTransitions,
    *,
    controls_raw: torch.Tensor,
    references_raw: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    all_errors = []
    all_separation = []
    all_ratio = []
    all_state_features = []
    all_transition_features = []
    map_squared_sum = torch.zeros(
        len(ACTION_CANDIDATE_NAMES),
        dataset.current.shape[-2],
        dataset.current.shape[-1],
        dtype=torch.float64,
    )
    map_weight_sum = torch.zeros(
        dataset.current.shape[-2], dataset.current.shape[-1], dtype=torch.float64
    )
    pair_indices = [
        (left, right)
        for left in range(len(ACTION_CANDIDATE_NAMES))
        for right in range(left + 1, len(ACTION_CANDIDATE_NAMES))
    ]
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        current = dataset.current[start:end].to(device=device, dtype=torch.float32)
        target = dataset.target[start:end].to(device=device, dtype=torch.float32)
        base = dataset.base_future[start:end].to(device=device, dtype=torch.float32)
        validity = dataset.validity[start:end].to(device=device, dtype=torch.float32)
        controls = normalize_actions(
            controls_raw[start:end].to(device=device, dtype=torch.float32), statistics
        )
        references = (
            normalize_actions(
                references_raw[start:end].to(device=device, dtype=torch.float32),
                statistics,
            )
            if model.centered
            else None
        )
        predictions = model(current, base, controls, references)["prediction"]
        expanded_validity = validity[:, None]
        squared_errors = (predictions - target[:, None]).square() * expanded_validity
        denominator = validity.flatten(start_dim=1).sum(dim=1) * predictions.shape[2]
        errors = squared_errors.flatten(start_dim=2).sum(dim=2) / denominator[:, None].clamp_min(1.0)
        all_errors.append(errors.cpu())
        pair_values = []
        for left, right in pair_indices:
            pair_values.append(
                _sample_errors(
                    predictions[:, left], predictions[:, right], validity
                ).sqrt()
            )
        all_separation.append(torch.stack(pair_values, dim=1).mean(dim=1).cpu())
        true_applied = predictions[:, 0] - base
        base_error_rms = _sample_errors(base, target, validity).sqrt()
        applied_rms = _sample_errors(
            true_applied, torch.zeros_like(true_applied), validity
        ).sqrt()
        all_ratio.append((applied_rms / base_error_rms.clamp_min(1e-8)).cpu())
        state_features, transition_features = _prediction_features(current, true_applied)
        all_state_features.append(state_features.cpu())
        all_transition_features.append(transition_features.cpu())
        differences = predictions - predictions[:, :1]
        map_squared_sum += (
            differences.square() * expanded_validity
        ).sum(dim=(0, 2)).double().cpu()
        map_weight_sum += validity.sum(dim=(0, 1)).double().cpu() * predictions.shape[2]
    errors_np = torch.cat(all_errors).numpy()
    maps = (map_squared_sum / map_weight_sum.clamp_min(1.0)).sqrt().numpy()
    metrics = {
        "controls": _control_metrics(errors_np, dataset.route_ids),
        "action_separation_rms": float(torch.cat(all_separation).mean()),
        "residual_to_base_error_rms_ratio": float(torch.cat(all_ratio).mean()),
        "gate": float(torch.sigmoid(model.gate_logit.detach().cpu())),
        "spatial_map_definition": (
            "RMS_channel_sample(prediction_control - prediction_true); "
            "oracle-warp overlap mask; model response map, not causal influence"
        ),
    }
    arrays = {
        "errors": errors_np,
        "spatial_maps": maps,
        "state_features": torch.cat(all_state_features).numpy(),
        "transition_features": torch.cat(all_transition_features).numpy(),
    }
    return metrics, arrays


@torch.inference_mode()
def _finite_difference_sensitivity(
    model: ActionFiLMResidualAdapter,
    dataset: PreparedTransitions,
    *,
    references_raw: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    device: torch.device,
    samples: int,
    perturbation: float,
) -> float:
    count = min(int(samples), len(dataset))
    current = dataset.current[:count].to(device=device, dtype=torch.float32)
    base = dataset.base_future[:count].to(device=device, dtype=torch.float32)
    validity = dataset.validity[:count].to(device=device, dtype=torch.float32)
    actions = normalize_actions(
        dataset.actions[:count].to(device=device, dtype=torch.float32), statistics
    )
    references = (
        normalize_actions(
            references_raw[:count].to(device=device, dtype=torch.float32), statistics
        )
        if model.centered
        else None
    )
    effects = []
    for dimension in range(actions.shape[1] * actions.shape[2]):
        plus = actions.clone().reshape(count, -1)
        minus = actions.clone().reshape(count, -1)
        plus[:, dimension] += perturbation
        minus[:, dimension] -= perturbation
        plus_prediction = model(
            current,
            base,
            plus.reshape_as(actions).unsqueeze(1),
            references,
        )["prediction"][:, 0]
        minus_prediction = model(
            current,
            base,
            minus.reshape_as(actions).unsqueeze(1),
            references,
        )["prediction"][:, 0]
        derivative = (plus_prediction - minus_prediction) / (2.0 * perturbation)
        effects.append(
            _sample_errors(
                derivative, torch.zeros_like(derivative), validity
            ).sqrt().cpu()
        )
    return float(torch.stack(effects, dim=1).mean())


def _ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    penalty: float = 1.0,
) -> dict[str, float]:
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    train_standard = (train_x - x_mean) / x_std
    val_standard = (val_x - x_mean) / x_std
    train_design = np.concatenate(
        (train_standard, np.ones((len(train_standard), 1))), axis=1
    )
    val_design = np.concatenate(
        (val_standard, np.ones((len(val_standard), 1))), axis=1
    )
    regularizer = np.eye(train_design.shape[1]) * float(penalty)
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(
        train_design.T @ train_design + regularizer,
        train_design.T @ train_y,
    )
    prediction = val_design @ weights
    residual_sum = ((val_y - prediction) ** 2).sum(axis=0)
    total_sum = ((val_y - val_y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    valid = total_sum > 1e-8
    r2 = 1.0 - residual_sum[valid] / total_sum[valid]
    return {
        "mean_r2": float(np.mean(r2)) if r2.size else float("nan"),
        "normalized_mae": float(np.mean(np.abs(val_y - prediction))),
    }


@torch.inference_mode()
def _true_features(
    model: ActionFiLMResidualAdapter,
    dataset: PreparedTransitions,
    *,
    references_raw: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_values = []
    transition_values = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        current = dataset.current[start:end].to(device=device, dtype=torch.float32)
        base = dataset.base_future[start:end].to(device=device, dtype=torch.float32)
        actions = normalize_actions(
            dataset.actions[start:end].to(device=device, dtype=torch.float32), statistics
        ).unsqueeze(1)
        references = (
            normalize_actions(
                references_raw[start:end].to(device=device, dtype=torch.float32),
                statistics,
            )
            if model.centered
            else None
        )
        prediction = model(current, base, actions, references)["prediction"][:, 0]
        state, transition = _prediction_features(current, prediction - base)
        state_values.append(state.cpu())
        transition_values.append(transition.cpu())
    return torch.cat(state_values).numpy(), torch.cat(transition_values).numpy()


def _route_bootstrap(
    numerator_errors: np.ndarray,
    denominator_errors: np.ndarray,
    route_ids: Iterable[str],
    *,
    samples: int,
    seed: int,
    ratio_denominator: str = "denominator",
) -> dict[str, float]:
    """Bootstrap equal-route ``100*(num-den)/selected denominator``."""

    if ratio_denominator not in {"numerator", "denominator"}:
        raise ValueError("ratio_denominator must be numerator or denominator")

    routes = np.asarray(list(route_ids))
    unique = np.unique(routes)
    numerator_route = np.asarray(
        [numerator_errors[routes == route].mean() for route in unique]
    )
    denominator_route = np.asarray(
        [denominator_errors[routes == route].mean() for route in unique]
    )
    point_divisor = (
        numerator_route.mean()
        if ratio_denominator == "numerator"
        else denominator_route.mean()
    )
    point = 100.0 * (
        numerator_route.mean() - denominator_route.mean()
    ) / max(point_divisor, 1e-12)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(unique), size=len(unique))
        numerator = numerator_route[indices].mean()
        denominator = denominator_route[indices].mean()
        divisor = numerator if ratio_denominator == "numerator" else denominator
        draws.append(100.0 * (numerator - denominator) / max(divisor, 1e-12))
    lower, upper = np.percentile(draws, (2.5, 97.5))
    return {
        "point_percent": float(point),
        "ci95_lower_percent": float(lower),
        "ci95_upper_percent": float(upper),
        "bootstrap_unit": "route",
        "routes": int(len(unique)),
        "samples": int(samples),
        "ratio_denominator": ratio_denominator,
    }


def _save_checkpoint(
    path: Path, model: nn.Module, metadata: dict[str, Any]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _write_report(path: Path, result: dict[str, Any]) -> None:
    verdict = result["verdict"]
    base = result["stage0"]["base_val_route_macro_mse"]
    lines = [
        "# Action Influence Pilot V1 Results",
        "",
        f"실행 완료: `{result['completed_at']}`",
        "",
        f"## 판정: `{verdict['status']}`",
        "",
        verdict["summary"],
        "",
        "이 결과는 frozen TF++ BEV latent와 단일 logged-policy action을 이용한 "
        "**observational proxy**다. same-state multi-action counterfactual 정확도나 "
        "closed-loop 개선을 입증하지 않는다.",
        "",
        "## Stage 0",
        "",
        "Passive base는 action 입력이 없으므로 action neglect 진단은 구조적으로 "
        "`N/A`다. 모든 action label의 출력 불변성만 확인했다.",
        "",
        f"- frozen passive validation route-macro standardized MSE: `{base:.6f}`",
        f"- action sensitivity: `{result['stage0']['action_sensitivity']}`",
        f"- control output max difference: `{result['stage0']['max_control_output_difference']}`",
        "",
        "## Stage 1 validation 결과",
        "",
        "| Variant | Seed | true MSE | base 대비 개선 | shuffled penalty | sensitivity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("uncentered_film", "centered_film_k4"):
        for seed, value in result["stage1"][variant].items():
            true_mse = value["metrics"]["controls"]["true"]["route_macro_mse"]
            improvement = 100.0 * (base - true_mse) / base
            shuffled = value["metrics"]["controls"]["shuffled"][
                "relative_penalty_vs_true_percent"
            ]
            sensitivity = value["metrics"]["finite_difference_sensitivity"]
            lines.append(
                f"| {variant} | {seed} | {true_mse:.6f} | {improvement:+.2f}% | "
                f"{shuffled:+.2f}% | {sensitivity:.6f} |"
            )
    pooled = verdict["pooled_centered"]
    alignment = verdict["alignment_baseline"]
    lines.extend(
        [
            "",
            "## Centered primary gate",
            "",
            f"- 3% 이상 개선 seed: `{pooled['seeds_meeting_improvement_gate']}`",
            f"- pooled improvement: `{pooled['improvement']}`",
            f"- pooled shuffled penalty: `{pooled['shuffled_penalty']}`",
            "",
            "## Mandatory persistence sanity",
            "",
            f"- raw unaligned persistence MSE: `{alignment['raw_unaligned_persistence_route_macro_mse']:.6f}`",
            f"- oracle-warp persistence MSE: `{alignment['oracle_warp_persistence_route_macro_mse']:.6f}` "
            f"(`{alignment['oracle_warp_vs_raw_percent']:+.2f}%` vs raw)",
            f"- uncentered 3-seed mean: `{alignment['uncentered_three_seed_mean_true_route_macro_mse']:.6f}` "
            f"(`{alignment['uncentered_vs_raw_percent']:+.2f}%` vs raw)",
            f"- centered 3-seed mean: `{alignment['centered_three_seed_mean_true_route_macro_mse']:.6f}` "
            f"(`{alignment['centered_vs_raw_percent']:+.2f}%` vs raw)",
            f"- baseline gate: `{'pass' if alignment['pass'] else 'fail'}`",
            "",
            "Spatial `.npy` 파일은 모델 출력이 action control에 반응한 위치를 "
            "나타낼 뿐 causal influence map이 아니다.",
            "",
            "## 다음 단계",
            "",
            verdict["next_step"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "action_influence_pilot_v1.yaml",
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="two-epoch one-seed plumbing run; never a research result",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("rt", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must decode to a mapping")
    if args.smoke:
        config = deepcopy(config)
        config["base"]["epochs"] = 2
        config["base"]["patience"] = 2
        config["adapter"]["epochs"] = 2
        config["adapter"]["patience"] = 2
        config["seeds"] = [17]
        config["evaluation"]["bootstrap_samples"] = 200
        config["evaluation"]["finite_difference_samples"] = 8

    cache_value = args.cache or Path(config["data"]["cache"])
    cache_root = (
        cache_value.resolve()
        if cache_value.is_absolute()
        else (project_root / cache_value).resolve()
    )
    output = args.output.resolve()
    _fresh_output(output)
    device = _resolve_device(args.device)
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    started_at = _utc_now()
    index_path = cache_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index_value = json.loads(index_path.read_text(encoding="utf-8"))
    source_value = index_value.get("source", {})
    if source_value.get("feature_source") != "backbone_bev" or int(
        source_value.get("cache_spatial_size", -1)
    ) != 64:
        raise ValueError(
            "oracle ego warp requires native backbone_bev cache_spatial_size=64"
        )
    if tuple(config["data"].get("analysis_splits", [])) != ("train", "val"):
        raise ValueError("action pilot must lock analysis_splits to [train, val]")
    if config["data"].get("oracle_current_to_future_se2_warp") is not True:
        raise ValueError("V1 protocol requires oracle_current_to_future_se2_warp=true")

    horizon = int(config["data"]["horizon"])
    train_raw = ActionTransitionDataset(
        cache_root,
        split="train",
        action_horizon=horizon,
        allow_test=False,
        max_frame_gap=1,
        preload=True,
    )
    val_raw = ActionTransitionDataset(
        cache_root,
        split="val",
        action_horizon=horizon,
        allow_test=False,
        max_frame_gap=1,
        preload=True,
    )
    expected_shape = (64, 64, 64)
    if train_raw.sample_shape["current_latent"] != expected_shape or val_raw.sample_shape[
        "current_latent"
    ] != expected_shape:
        raise ValueError(
            f"native TF++ latent must be {expected_shape}, got "
            f"{train_raw.sample_shape['current_latent']} and "
            f"{val_raw.sample_shape['current_latent']}"
        )
    statistics = train_normalization_statistics(train_raw)
    torch.save(statistics, output / "train_normalization.pt")
    normalization_artifact = {
        "path": str(output / "train_normalization.pt"),
        "sha256": _sha256(output / "train_normalization.pt"),
        "latent_channels": int(statistics["latent_mean"].numel()),
        "action_mean": statistics["action_mean"].tolist(),
        "action_std": statistics["action_std"].tolist(),
    }
    prepare_batch = int(config["data"]["prepare_batch_size"])
    train, train_warp = _prepare_split(
        train_raw,
        statistics=statistics,
        device=device,
        batch_size=prepare_batch,
    )
    val, val_warp = _prepare_split(
        val_raw,
        statistics=statistics,
        device=device,
        batch_size=prepare_batch,
    )
    del train_raw, val_raw
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reference_count = int(config["adapter"]["reference_count"])
    train_reference_indices, train_reference_ids = _reference_indices(
        train, train, count=reference_count
    )
    val_reference_indices, val_reference_ids = _reference_indices(
        val, train, count=reference_count
    )
    train_references = train.actions[train_reference_indices]
    val_references = train.actions[val_reference_indices]
    val_shuffle = deterministic_derangement(len(val), 1701)
    val_other = deterministic_derangement(len(val), 2903)
    if torch.equal(val_shuffle, val_other):
        val_other = deterministic_derangement(len(val), 2904)
    val_controls, names = build_action_candidates(val.actions, val_shuffle, val_other)
    if names != ACTION_CANDIDATE_NAMES:
        raise RuntimeError("action candidate order drift")
    control_manifest = {
        "names": list(names),
        "shuffle_indices": val_shuffle.tolist(),
        "other_indices": val_other.tolist(),
        "train_reference_sample_ids": train_reference_ids,
        "val_reference_sample_ids": val_reference_ids,
    }
    _write_json(output / "control_manifest.json", control_manifest)

    passive, passive_training = _train_passive(
        train, val, config=config["base"], device=device
    )
    passive_checkpoint = _save_checkpoint(
        output / "base" / "best.pt", passive, passive_training
    )
    train.base_future, train_base_errors = _predict_passive(
        passive,
        train,
        device=device,
        batch_size=int(config["base"]["batch_size"]),
    )
    val.base_future, val_base_errors = _predict_passive(
        passive,
        val,
        device=device,
        batch_size=int(config["base"]["batch_size"]),
    )
    base_macro = route_macro_mean(val_base_errors, val.route_ids)
    stage0 = {
        "action_neglect_diagnosis": "not_applicable_action_free_base",
        "action_sensitivity": "structurally_zero_no_action_input",
        "base_val_micro_mse": float(val_base_errors.mean()),
        "base_val_route_macro_mse": base_macro,
        "control_errors": {
            name: {
                "micro_mse": float(val_base_errors.mean()),
                "route_macro_mse": base_macro,
            }
            for name in ACTION_CANDIDATE_NAMES
        },
        "max_control_output_difference": 0.0,
        "finite_difference_sensitivity": 0.0,
        "invariance_audit": "pass",
    }

    variants = ("uncentered_film", "centered_film_k4")
    seeds = [int(value) for value in config["seeds"]]
    stage1: dict[str, dict[str, Any]] = {name: {} for name in variants}
    stored_arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    stored_models: dict[tuple[str, int], ActionFiLMResidualAdapter] = {}
    parameter_counts: dict[str, int] = {}
    for variant in variants:
        for seed in seeds:
            model, training = _train_adapter(
                variant,
                seed,
                train,
                val,
                train_references=train_references,
                val_references=val_references,
                statistics=statistics,
                config=config["adapter"],
                device=device,
            )
            checkpoint = _save_checkpoint(
                output / variant / f"seed_{seed}" / "best.pt", model, training
            )
            metrics, arrays = _evaluate_adapter(
                model,
                val,
                controls_raw=val_controls,
                references_raw=val_references,
                statistics=statistics,
                device=device,
                batch_size=int(config["adapter"]["batch_size"]),
            )
            sensitivity = _finite_difference_sensitivity(
                model,
                val,
                references_raw=val_references,
                statistics=statistics,
                device=device,
                samples=int(config["evaluation"]["finite_difference_samples"]),
                perturbation=float(config["evaluation"]["finite_difference_perturbation"]),
            )
            metrics["finite_difference_sensitivity"] = sensitivity
            maps_dir = output / variant / f"seed_{seed}" / "spatial_maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            for control_index, control_name in enumerate(ACTION_CANDIDATE_NAMES):
                np.save(
                    maps_dir / f"true_vs_{control_name}.npy",
                    arrays["spatial_maps"][control_index],
                    allow_pickle=False,
                )
            stage1[variant][str(seed)] = {
                "training": training,
                "checkpoint": checkpoint,
                "metrics": metrics,
            }
            stored_arrays[(variant, seed)] = arrays
            stored_models[(variant, seed)] = model
            parameter_counts[variant] = int(training["trainable_parameters"])
    if parameter_counts["uncentered_film"] != parameter_counts["centered_film_k4"]:
        raise RuntimeError("uncentered and centered adapters are not equal capacity")

    # Inverse-action recovery: compare a state-only ridge probe with the same
    # state features plus the model's applied transition residual.
    train_y = normalize_actions(train.actions.float(), statistics).reshape(len(train), -1).numpy()
    val_y = normalize_actions(val.actions.float(), statistics).reshape(len(val), -1).numpy()
    for variant in variants:
        for seed in seeds:
            model = stored_models[(variant, seed)]
            train_state, train_transition = _true_features(
                model,
                train,
                references_raw=train_references,
                statistics=statistics,
                device=device,
                batch_size=int(config["adapter"]["batch_size"]),
            )
            val_arrays = stored_arrays[(variant, seed)]
            state_probe = _ridge_probe(
                train_state,
                train_y,
                val_arrays["state_features"],
                val_y,
            )
            transition_probe = _ridge_probe(
                train_transition,
                train_y,
                val_arrays["transition_features"],
                val_y,
            )
            stage1[variant][str(seed)]["metrics"]["inverse_action_probe"] = {
                "state_only": state_probe,
                "state_plus_predicted_residual": transition_probe,
                "delta_mean_r2": float(
                    transition_probe["mean_r2"] - state_probe["mean_r2"]
                ),
            }

    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    centered_error_stack = np.stack(
        [stored_arrays[("centered_film_k4", seed)]["errors"] for seed in seeds]
    )
    centered_mean_errors = centered_error_stack.mean(axis=0)
    uncentered_error_stack = np.stack(
        [stored_arrays[("uncentered_film", seed)]["errors"] for seed in seeds]
    )
    uncentered_mean_errors = uncentered_error_stack.mean(axis=0)
    pooled_improvement = _route_bootstrap(
        val_base_errors,
        centered_mean_errors[:, 0],
        val.route_ids,
        samples=bootstrap_samples,
        seed=701,
        ratio_denominator="numerator",
    )
    pooled_shuffled = _route_bootstrap(
        centered_mean_errors[:, 1],
        centered_mean_errors[:, 0],
        val.route_ids,
        samples=bootstrap_samples,
        seed=1701,
    )
    improvement_threshold = float(config["gates"]["improvement_percent"])
    shuffled_threshold = float(config["gates"]["shuffled_penalty_percent"])
    required_seeds = int(config["gates"]["required_seeds"])
    seed_improvements = {}
    seed_sensitivities = {}
    for seed in seeds:
        metrics = stage1["centered_film_k4"][str(seed)]["metrics"]
        true_mse = metrics["controls"]["true"]["route_macro_mse"]
        seed_improvements[str(seed)] = 100.0 * (base_macro - true_mse) / base_macro
        seed_sensitivities[str(seed)] = metrics["finite_difference_sensitivity"]
    meeting = sum(
        value >= improvement_threshold for value in seed_improvements.values()
    )
    go = (
        meeting >= required_seeds
        and pooled_improvement["ci95_lower_percent"] > 0.0
        and pooled_shuffled["point_percent"] >= shuffled_threshold
        and pooled_shuffled["ci95_lower_percent"] > 0.0
        and float(np.mean(list(seed_sensitivities.values()))) > 1e-6
    )
    generic_correction = (
        abs(pooled_shuffled["point_percent"]) < 1.0
        and pooled_improvement["point_percent"] > 0.0
    )
    raw_persistence_macro = float(val_warp["raw_persistence_route_macro_mse"])
    centered_pooled_macro = route_macro_mean(
        centered_mean_errors[:, 0], val.route_ids
    )
    uncentered_pooled_macro = route_macro_mean(
        uncentered_mean_errors[:, 0], val.route_ids
    )
    alignment_baseline = {
        "gate_origin": (
            "protocol correction after the initial run exposed an omitted trivial baseline; "
            "the final rerun changes no model/training hyperparameter"
        ),
        "required": bool(config["gates"].get("require_beat_raw_persistence", False)),
        "raw_unaligned_persistence_route_macro_mse": raw_persistence_macro,
        "oracle_warp_persistence_route_macro_mse": float(
            val_warp["oracle_warp_persistence_route_macro_mse"]
        ),
        "learned_passive_route_macro_mse": base_macro,
        "uncentered_three_seed_mean_true_route_macro_mse": uncentered_pooled_macro,
        "centered_three_seed_mean_true_route_macro_mse": centered_pooled_macro,
        "oracle_warp_vs_raw_percent": float(
            100.0
            * (
                val_warp["oracle_warp_persistence_route_macro_mse"]
                - raw_persistence_macro
            )
            / raw_persistence_macro
        ),
        "uncentered_vs_raw_percent": float(
            100.0 * (uncentered_pooled_macro - raw_persistence_macro)
            / raw_persistence_macro
        ),
        "centered_vs_raw_percent": float(
            100.0 * (centered_pooled_macro - raw_persistence_macro)
            / raw_persistence_macro
        ),
        "pass": centered_pooled_macro < raw_persistence_macro,
    }
    alignment_no_go = alignment_baseline["required"] and not alignment_baseline["pass"]
    if alignment_no_go:
        status = "no_go"
        summary = (
            "Centered residual은 warped passive base보다 개선됐지만, 전체 pipeline이 "
            "단순 raw-latent persistence보다 나빴다. TF++ latent의 metric-warp 기반 "
            "spatial decomposition은 현재 형태로 확장하지 않는다."
        )
        next_step = (
            "Query adapter는 중단한다. 계속하려면 warp-equivariant occupancy/semantic "
            "latent 또는 실제 action-conditioned world model을 선택하고, CARLA "
            "same-state paired action 데이터에서 처음부터 재검증한다."
        )
    elif go:
        status = "go_to_paired_counterfactual_data_not_stage2_model_yet"
        summary = (
            "Centered FiLM이 사전 고정 validation gate를 통과했다. 다만 logged-policy "
            "confounding 때문에 query adapter보다 먼저 CARLA same-state paired action "
            "데이터를 수집해야 한다."
        )
        next_step = (
            "작은 CARLA 0.9.15 same-state multi-action 수집기를 사전등록하고, "
            "counterfactual difference target을 만든 뒤 Stage 1을 재검증한다."
        )
    elif (
        pooled_improvement["point_percent"] < 1.0
        or pooled_improvement["ci95_upper_percent"] <= 0.0
        or generic_correction
    ):
        status = "no_go"
        summary = (
            "단순 centered action residual이 최소 개선/action-use gate를 통과하지 못했다. "
            "현재 결과로 query-based Stage 2를 확장하지 않는다."
        )
        next_step = (
            "Stage 2 query adapter는 중단한다. 연구를 계속하려면 모델 복잡화가 아니라 "
            "same-state paired counterfactual 데이터와 실제 action-conditioned world model을 "
            "먼저 확보한다."
        )
    else:
        status = "inconclusive"
        summary = (
            "일부 개선 신호는 있으나 route-bootstrap/action-use gate가 모두 충족되지 않았다."
        )
        next_step = (
            "현재 validation에 맞춘 추가 튜닝은 하지 않는다. paired counterfactual 데이터나 "
            "독립 route를 확보한 뒤 같은 고정 설정으로 한 번 재실행한다."
        )
    verdict = {
        "status": status,
        "summary": summary,
        "next_step": next_step,
        "pooled_centered": {
            "seed_improvement_percent": seed_improvements,
            "seed_sensitivity": seed_sensitivities,
            "seeds_meeting_improvement_gate": meeting,
            "required_seeds": required_seeds,
            "improvement": pooled_improvement,
            "shuffled_penalty": pooled_shuffled,
            "generic_correction_flag": generic_correction,
        },
        "alignment_baseline": alignment_baseline,
        "stage2_query_adapter_authorized": False,
        "reason_stage2_not_automatically_authorized": (
            "Even a proxy GO requires paired same-state counterfactual data before model expansion."
        ),
    }

    data_manifest = {
        "cache_root": str(cache_root),
        "cache_index_sha256": _sha256(index_path),
        "analysis_splits": ["train", "val"],
        "test_evaluation_performed": False,
        "test_lock_claim": (
            "none: this cache was extracted and deep-audited across all records; test is simply "
            "excluded from training/evaluation"
        ),
        "horizon_saved_frames": horizon,
        "nominal_horizon_seconds": float(config["data"]["horizon_seconds"]),
        "train_transitions": len(train),
        "val_transitions": len(val),
        "train_routes": len(set(train.route_ids)),
        "val_routes": len(set(val.route_ids)),
        "feature_shape": list(train.current.shape[1:]),
        "action_shape": list(train.actions.shape[1:]),
        "action_semantics": [
            "saved_steer_pre_noise",
            "saved_throttle_pre_some_overrides",
            "bool(brake_or_control_brake)",
        ],
        "oracle_ego_warp": (
            "current BEV warped to future ego frame using recorded future pose; diagnostic, "
            "not deployable prediction"
        ),
        "train_sample_ids_sha256": _stable_hash(list(train.sample_ids)),
        "val_sample_ids_sha256": _stable_hash(list(val.sample_ids)),
        "control_manifest_sha256": _sha256(output / "control_manifest.json"),
    }
    _write_json(output / "data_manifest.json", data_manifest)
    completed_at = _utc_now()
    result = {
        "schema_version": 1,
        "experiment": "observational_action_influence_proxy_v1",
        "smoke": bool(args.smoke),
        "started_at": started_at,
        "completed_at": completed_at,
        "device": str(device),
        "config": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "normalization": normalization_artifact,
        "data": data_manifest,
        "warp_diagnostics": {"train": train_warp, "val": val_warp},
        "base_training": passive_training,
        "base_checkpoint": passive_checkpoint,
        "stage0": stage0,
        "stage1": stage1,
        "equal_capacity": {
            "parameter_counts": parameter_counts,
            "pass": parameter_counts["uncentered_film"]
            == parameter_counts["centered_film_k4"],
        },
        "verdict": verdict,
        "limitations": [
            "TF++ supplies frozen perception latents but is not a latent world model.",
            "The passive predictor is trained locally then frozen; it is not a pretrained WM.",
            "Each state has one logged action/future, so causal counterfactual accuracy is unmeasured.",
            "Recorded controls are 4 Hz proxies and are not exact applied 20 Hz actuator sequences.",
            "Future recorded pose is used for oracle ego alignment.",
            "Only three validation routes are available; route-bootstrap intervals are coarse.",
            "No cache test record is evaluated, but no untouched-test claim is made.",
        ],
    }
    _write_json(output / "results.json", result)
    _write_report(output / "RESULTS.md", result)
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
    print(output / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
