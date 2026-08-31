"""Training and evaluation loop for the three adapter ablations."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# CUDA's deterministic cuBLAS modes are read when the CUDA context is created.
# Set the safe default as early as this module can, and require launch-time
# configuration in the real-study runbook because another import may already
# have initialized CUDA before this module is reached.
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG)

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audit import audit_cache
from .cache import load_index
from .config import validate_config
from .data import TemporalCacheDataset
from .losses import residual_adapter_loss
from .metrics import mean_metrics, per_sample_metrics, worst_fraction_indices
from .model import (
    VARIANTS,
    AdapterVariant,
    TemporalResidualAdapter,
    match_current_only_dimensions,
)
from .route_stats import paired_route_comparison, summarize_route_metrics


SELECTION_MANIFEST = "selection_manifest.json"
TEST_OPEN_MARKER = "test_opened.marker.json"
STUDY_SELECTION_OWNER_KIND = "multiseed_study"
CHECKPOINT_SELECTION_METRIC = {
    "split": "val",
    "metric": "ade",
    "aggregation": "equal_route_macro",
}

# This process-local capability is intentionally unavailable from the generic
# single-selection CLI.  A study-owned child can only be opened by the verified
# study orchestration path, which supplies both this capability and the parent
# identity committed in the child manifest.
_STUDY_FINALIZE_CAPABILITY = object()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_selection_owner(owner: Any) -> dict[str, str] | None:
    if owner is None:
        return None
    if not isinstance(owner, dict) or set(owner) != {"kind", "study_owner_id"}:
        raise ValueError("selection_owner must contain exactly kind and study_owner_id")
    if owner.get("kind") != STUDY_SELECTION_OWNER_KIND:
        raise ValueError("selection_owner kind is unsupported")
    study_owner_id = owner.get("study_owner_id")
    if (
        not isinstance(study_owner_id, str)
        or len(study_owner_id) != 64
        or any(character not in "0123456789abcdef" for character in study_owner_id)
    ):
        raise ValueError("selection_owner study_owner_id must be a lowercase SHA256 digest")
    return {"kind": STUDY_SELECTION_OWNER_KIND, "study_owner_id": study_owner_id}


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_fresh_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix with or overwrite an existing experiment directory: {output_dir}"
        )


def seed_everything(seed: int) -> None:
    # ``warn_only=True`` is deliberate: the adapter currently contains
    # AdaptiveAvgPool2d, whose CUDA backward is documented by PyTorch as lacking
    # a deterministic implementation.  We still request deterministic kernels
    # everywhere PyTorch has one and make any fallback visible as a warning,
    # rather than making the actual CUDA pilot impossible to train.
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_runtime_metadata() -> dict[str, Any]:
    """Return the effective reproducibility policy for result provenance."""

    warn_only_query = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    return {
        "policy": "deterministic_algorithms_warn_only",
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            bool(warn_only_query()) if warn_only_query is not None else None
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cublas_process_start_requirement": (
            "must be exported before process launch / first CUDA context for a "
            "reproducible CUDA study"
        ),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this process")
    return device


def _dataset(cache_root: Path, split: str, config: dict[str, Any]) -> TemporalCacheDataset:
    data = config["data"]
    return TemporalCacheDataset(
        cache_root,
        split=split,
        history_length=int(data["history_length"]),
        align_past_trajectories=bool(data["align_past_trajectories"]),
        max_frame_gap=data.get("max_frame_gap"),
        validate_on_load=bool(data.get("validate_on_load", False)),
    )


def _loader(dataset: TemporalCacheDataset, config: dict[str, Any], *, shuffle: bool) -> DataLoader:
    training = config["training"]
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training["num_workers"]),
        pin_memory=False,
        drop_last=False,
    )


def _model_from_sample(
    sample: dict[str, Any], variant_name: str, config: dict[str, Any]
) -> TemporalResidualAdapter:
    variant = VARIANTS[variant_name]
    adapter = config["adapter"]
    num_waypoints = int(sample["current_trajectory"].shape[-2])
    bev_channels = int(sample["current_bev"].shape[-3])

    def build(
        model_variant: AdapterVariant,
        *,
        query_dim: int = int(adapter["query_dim"]),
        hidden_dim: int = int(adapter["hidden_dim"]),
    ) -> TemporalResidualAdapter:
        return TemporalResidualAdapter(
            num_waypoints=num_waypoints,
            bev_channels=bev_channels,
            use_past_trajectory=model_variant.use_past_trajectory,
            use_past_bev=model_variant.use_past_bev,
            bev_compressed_channels=int(adapter["bev_compressed_channels"]),
            bev_pooled_size=int(adapter["bev_pooled_size"]),
            bev_token_dim=int(adapter["bev_token_dim"]),
            hidden_dim=hidden_dim,
            query_dim=query_dim,
            dropout=float(adapter["dropout"]),
            learned_gate=bool(adapter["learned_gate"]),
        )

    if variant_name == "current_only_matched":
        target_variant_name = variant.capacity_target
        if target_variant_name is None:
            raise RuntimeError("current_only_matched is missing its capacity target")
        target_model = build(VARIANTS[target_variant_name])
        target_parameters = sum(parameter.numel() for parameter in target_model.parameters())
        query_dim, hidden_dim = match_current_only_dimensions(
            num_waypoints=num_waypoints,
            base_query_dim=int(adapter["query_dim"]),
            base_hidden_dim=int(adapter["hidden_dim"]),
            learned_gate=bool(adapter["learned_gate"]),
            target_parameters=target_parameters,
        )
        return build(variant, query_dim=query_dim, hidden_dim=hidden_dim)
    return build(variant)


def _trainable_parameter_count(model: TemporalResidualAdapter) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _capacity_match_metadata(
    model: TemporalResidualAdapter,
    *,
    sample: dict[str, Any],
    variant_name: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    variant = VARIANTS[variant_name]
    if variant.capacity_target is None:
        return None
    target_model = _model_from_sample(sample, variant.capacity_target, config)
    model_count = _trainable_parameter_count(model)
    target_count = _trainable_parameter_count(target_model)
    model_shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    target_shapes = {name: tuple(tensor.shape) for name, tensor in target_model.state_dict().items()}
    return {
        "target_variant": variant.capacity_target,
        "trainable_parameters": model_count,
        "target_trainable_parameters": target_count,
        "parameter_difference": model_count - target_count,
        "relative_parameter_difference": (model_count - target_count) / target_count,
        "exact_state_shape_match": model_shapes == target_shapes,
        "query_dim": int(model.current_query[0].out_features),
        "hidden_dim": int(model.delta_head.in_features),
    }


def deterministic_temporal_permutations(
    sample_ids: Iterable[str], *, history_length: int, seed: int
) -> torch.Tensor:
    """Return a stable, non-identity time permutation for each sample window.

    Every row is derived only from ``seed``, that sample's portable ID, and
    indices already present in its own window.  It is therefore independent of
    loader order/workers and cannot draw a frame from another route or split.
    """

    if history_length < 1:
        raise ValueError("history_length must be positive")
    identity = list(range(history_length))
    rows: list[list[int]] = []
    for sample_id in sample_ids:
        order = sorted(
            identity,
            key=lambda index: hashlib.sha256(
                f"{int(seed)}|shuffled_past_bev|{sample_id}|{index}".encode("utf-8")
            ).digest(),
        )
        if history_length > 1 and order == identity:
            order = order[1:] + order[:1]
        rows.append(order)
    return torch.tensor(rows, dtype=torch.long)


def _variant_batch_inputs(
    batch: dict[str, Any],
    *,
    variant_name: str,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor | None]:
    """Materialize one variant's declared inputs without any external lookup."""

    variant = VARIANTS[variant_name]
    current_trajectory = batch["current_trajectory"].to(device)
    history_length = int(batch["past_trajectory"].shape[1])
    if int(batch["past_bev"].shape[1]) != history_length:
        raise ValueError("past trajectory and BEV histories have different lengths")

    if variant.trajectory_history == "none":
        trajectory_history = None
    elif variant.trajectory_history == "past":
        trajectory_history = batch["past_trajectory"].to(device)
    elif variant.trajectory_history == "repeat_current":
        trajectory_history = current_trajectory.unsqueeze(1).expand(
            -1, history_length, -1, -1
        )
    else:  # pragma: no cover - frozen dataclass literals guard this branch
        raise ValueError(f"unsupported trajectory history: {variant.trajectory_history}")

    if variant.bev_history == "none":
        bev_history = None
    elif variant.bev_history == "past":
        bev_history = batch["past_bev"].to(device)
    elif variant.bev_history == "repeat_current":
        current_bev = batch["current_bev"].to(device)
        bev_history = current_bev.unsqueeze(1).expand(-1, history_length, -1, -1, -1)
    elif variant.bev_history == "shuffled_past":
        past_bev = batch["past_bev"].to(device)
        permutations = deterministic_temporal_permutations(
            list(batch["sample_id"]), history_length=history_length, seed=seed
        ).to(device)
        batch_indices = torch.arange(past_bev.shape[0], device=device).unsqueeze(1)
        bev_history = past_bev[batch_indices, permutations]
    else:  # pragma: no cover - frozen dataclass literals guard this branch
        raise ValueError(f"unsupported BEV history: {variant.bev_history}")

    return {
        "current_trajectory": current_trajectory,
        "past_trajectory": trajectory_history,
        "past_bev": bev_history,
    }


def _adapter_forward(
    model: TemporalResidualAdapter,
    batch: dict[str, Any],
    device: torch.device,
    *,
    variant_name: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    variant = VARIANTS[variant_name]
    if (
        model.use_past_trajectory != variant.use_past_trajectory
        or model.use_past_bev != variant.use_past_bev
    ):
        raise ValueError(f"model capacity does not match input semantics for {variant_name}")
    return model(**_variant_batch_inputs(batch, variant_name=variant_name, device=device, seed=seed))


@torch.inference_mode()
def collect_predictions(
    loader: DataLoader,
    *,
    device: torch.device,
    model: TemporalResidualAdapter | None = None,
    variant_name: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if model is not None:
        if variant_name not in VARIANTS:
            raise ValueError("model evaluation requires a valid variant_name")
        model.eval()
    predictions, targets, gates, residuals, applied_residuals = [], [], [], [], []
    latency_per_sample_ms: list[float] = []
    sample_ids: list[str] = []
    route_ids: list[str] = []
    for batch in loader:
        current = batch["current_trajectory"].to(device)
        if model is None:
            prediction = current
            gate = torch.zeros((current.shape[0], 1, 1), device=device)
            residual = torch.zeros_like(current)
            latency_per_sample_ms.extend([0.0] * int(current.shape[0]))
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = _adapter_forward(
                model, batch, device, variant_name=variant_name, seed=seed
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latency_per_sample_ms.extend(
                [elapsed_ms / int(current.shape[0])] * int(current.shape[0])
            )
            prediction = output["trajectory"]
            gate = output["gate"]
            residual = output["delta"]
        predictions.append(prediction.cpu())
        targets.append(batch["gt_trajectory"].cpu())
        gates.append(gate.reshape(gate.shape[0], -1).mean(dim=1).cpu())
        residuals.append(residual.abs().mean(dim=(-1, -2)).cpu())
        applied_residuals.append((prediction - current).abs().mean(dim=(-1, -2)).cpu())
        sample_ids.extend(list(batch["sample_id"]))
        route_ids.extend(list(batch["route_id"]))
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    return {
        "sample_ids": sample_ids,
        "route_ids": route_ids,
        "prediction": prediction_tensor,
        "target": target_tensor,
        "gate": torch.cat(gates),
        "residual_l1": torch.cat(residuals),
        "applied_residual_l1": torch.cat(applied_residuals),
        "latency_per_sample_ms": torch.tensor(latency_per_sample_ms, dtype=torch.float64),
        "per_sample": per_sample_metrics(prediction_tensor, target_tensor),
    }


def summarize_predictions(
    collected: dict[str, Any], *, worst_indices: torch.Tensor | None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(collected["sample_ids"]),
        "overall": mean_metrics(collected["per_sample"]),
        "mean_gate": float(collected["gate"].mean().item()),
        "mean_raw_residual_l1": float(collected["residual_l1"].mean().item()),
        "mean_applied_residual_l1": float(collected["applied_residual_l1"].mean().item()),
        "mean_latency_ms": float(collected["latency_per_sample_ms"].mean().item()),
        "p95_latency_ms": float(
            torch.quantile(collected["latency_per_sample_ms"], 0.95).item()
        ),
        "latency_semantics": "cached_adapter_forward_plus_device_transfer_per_sample",
    }
    summary.update(
        summarize_route_metrics(collected["per_sample"], collected["route_ids"])
    )
    if worst_indices is not None:
        summary["baseline_worst_slice"] = mean_metrics(collected["per_sample"], worst_indices)
        summary["baseline_worst_slice_count"] = int(worst_indices.numel())
    return summary


def _validation_route_macro_ade(
    collected: dict[str, Any],
    *,
    expected_sample_ids: Sequence[str],
    expected_route_ids: Sequence[str],
) -> float:
    """Return equal-route validation ADE after enforcing the locked ordering."""

    sample_ids = list(collected.get("sample_ids", ()))
    route_ids = list(collected.get("route_ids", ()))
    expected_samples = list(expected_sample_ids)
    expected_routes = list(expected_route_ids)
    if sample_ids != expected_samples:
        raise RuntimeError(
            "validation sample IDs/order differ from the shared baseline order"
        )
    if route_ids != expected_routes:
        raise RuntimeError(
            "validation route IDs/order differ from the shared baseline order"
        )
    if len(sample_ids) != len(route_ids):
        raise RuntimeError("validation sample and route ID counts differ")
    route_summary = summarize_route_metrics(collected["per_sample"], route_ids)
    return float(route_summary["route_macro"]["ade"])


def _train_one_variant(
    variant_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    expected_val_sample_ids: Sequence[str],
    expected_val_route_ids: Sequence[str],
) -> tuple[TemporalResidualAdapter, list[dict[str, float]]]:
    model = _model_from_sample(train_loader.dataset[0], variant_name, config).to(device)
    input_semantics = VARIANTS[variant_name].input_semantics()
    capacity_match = _capacity_match_metadata(
        model,
        sample=train_loader.dataset[0],
        variant_name=variant_name,
        config=config,
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    best_route_macro_ade = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "trajectory_l1": 0.0, "residual_l1": 0.0}
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = _adapter_forward(
                model,
                batch,
                device,
                variant_name=variant_name,
                seed=int(config["seed"]),
            )
            loss_items = residual_adapter_loss(
                output,
                batch["gt_trajectory"].to(device),
                residual_weight=float(training["residual_weight"]),
            )
            loss_items["loss"].backward()
            clip = float(training.get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            batch_size = int(batch["gt_trajectory"].shape[0])
            count += batch_size
            for name in totals:
                totals[name] += float(loss_items[name].detach().item()) * batch_size

        validation = collect_predictions(
            val_loader,
            device=device,
            model=model,
            variant_name=variant_name,
            seed=int(config["seed"]),
        )
        val_route_macro_ade = _validation_route_macro_ade(
            validation,
            expected_sample_ids=expected_val_sample_ids,
            expected_route_ids=expected_val_route_ids,
        )
        val_sample_micro_ade = float(validation["per_sample"]["ade"].mean().item())
        epoch_record = {
            "epoch": float(epoch),
            # Backward-compatible name, now explicitly equal-route macro.
            "val_ade": val_route_macro_ade,
            "val_route_macro_ade": val_route_macro_ade,
            "val_sample_micro_ade": val_sample_micro_ade,
        }
        epoch_record.update({f"train_{name}": value / count for name, value in totals.items()})
        history.append(epoch_record)
        if val_route_macro_ade < best_route_macro_ade:
            best_route_macro_ade = val_route_macro_ade
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant_name,
            "model_state_dict": best_state,
            # ``best_val_ade`` is retained as an alias for old result readers.
            "best_val_ade": best_route_macro_ade,
            "best_val_route_macro_ade": best_route_macro_ade,
            "selection_metric": deepcopy(CHECKPOINT_SELECTION_METRIC),
            "config": deepcopy(config),
            "input_semantics": input_semantics,
            "capacity_match": capacity_match,
        },
        checkpoint_dir / f"{variant_name}.pt",
    )
    return model, history


def run_pilot(
    *,
    cache_root: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    device_name: str = "auto",
    variants: Iterable[str] = ("current_only", "trajectory_only", "bev_only", "combined"),
    evaluation_mode: str = "selection",
    selection_owner: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Train and select on train/val only, producing a locked selection artifact."""

    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    if evaluation_mode != "selection":
        raise ValueError("run_pilot only supports selection; use finalize_selection for locked final testing")
    validate_config(config)
    selection_owner = _validated_selection_owner(selection_owner)
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be a non-empty list without duplicates")
    for variant_name in variants:
        if variant_name not in VARIANTS:
            raise ValueError(f"unknown variant {variant_name}; choose from {sorted(VARIANTS)}")
    _require_fresh_output(output_dir)
    if bool(config["data"].get("require_deep_audit", True)):
        # Test tensors are deliberately not deserialized during selection.
        cache_audit = audit_cache(cache_root, deep=True, deep_splits={"train", "val"})
        if cache_audit["status"] != "pass":
            raise ValueError(f"deep cache audit failed: {cache_audit['errors'][:5]}")
    else:
        cache_audit = {"status": "skipped", "deep": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config["training"].get("torch_num_threads", 4)))

    eval_splits = ("train", "val")
    datasets = {split: _dataset(cache_root, split, config) for split in eval_splits}
    train_loader = _loader(datasets["train"], config, shuffle=True)
    eval_loaders = {split: _loader(dataset, config, shuffle=False) for split, dataset in datasets.items()}

    baseline_collected = {
        split: collect_predictions(loader, device=device) for split, loader in eval_loaders.items()
    }
    worst_fraction = float(config["evaluation"]["worst_fraction"])
    worst_indices = {
        split: worst_fraction_indices(collected["per_sample"]["ade"], worst_fraction)
        for split, collected in baseline_collected.items()
    }
    cache_index = load_index(cache_root)
    results: dict[str, Any] = {
        "status": "completed",
        "evaluation_mode": "selection",
        "test_opened": False,
        "evidence_level": "synthetic_smoke"
        if cache_index.get("source", {}).get("kind") == "synthetic"
        else "offline_cache_evaluation",
        "device": str(device),
        "seed": seed,
        "reproducibility": deterministic_runtime_metadata(),
        "checkpoint_selection_metric": deepcopy(CHECKPOINT_SELECTION_METRIC),
        "cache_source": cache_index.get("source", {}),
        "cache_audit": cache_audit,
        "metric_semantics": (
            {
                "ade": "mean_checkpoint_displacement_m",
                "fde": "last_checkpoint_error_m",
            }
            if cache_index.get("source", {}).get("target_semantics") == "geometric_path"
            else {"ade": "average_displacement_error_m", "fde": "final_displacement_error_m"}
        ),
        "dataset_windows": {split: len(dataset) for split, dataset in datasets.items()},
        "baseline": {
            split: summarize_predictions(collected, worst_indices=worst_indices[split])
            for split, collected in baseline_collected.items()
        },
        "variants": {},
        "configuration": deepcopy(config),
    }
    if selection_owner is not None:
        results["selection_owner"] = deepcopy(selection_owner)

    for variant_name in variants:
        # Reset per variant so initial weights and loader sampling are reproducible.
        seed_everything(seed)
        variant_train_loader = _loader(datasets["train"], config, shuffle=True)
        model, history = _train_one_variant(
            variant_name,
            variant_train_loader,
            eval_loaders["val"],
            config=config,
            device=device,
            checkpoint_dir=output_dir / "checkpoints",
            expected_val_sample_ids=baseline_collected["val"]["sample_ids"],
            expected_val_route_ids=baseline_collected["val"]["route_ids"],
        )
        split_results: dict[str, Any] = {}
        for split, loader in eval_loaders.items():
            collected = collect_predictions(
                loader,
                device=device,
                model=model,
                variant_name=variant_name,
                seed=seed,
            )
            if collected["sample_ids"] != baseline_collected[split]["sample_ids"]:
                raise RuntimeError("variant evaluation order differs from the shared baseline order")
            if collected["route_ids"] != baseline_collected[split]["route_ids"]:
                raise RuntimeError("variant route order differs from the shared baseline order")
            split_results[split] = summarize_predictions(
                collected, worst_indices=worst_indices[split]
            )
            split_results[split]["paired_vs_baseline"] = paired_route_comparison(
                collected["per_sample"],
                baseline_collected[split]["per_sample"],
                collected["route_ids"],
                bootstrap_seed=seed,
            )
            split_results[split]["paired_delta_ade_vs_baseline"] = float(
                (
                    collected["per_sample"]["ade"]
                    - baseline_collected[split]["per_sample"]["ade"]
                ).mean().item()
            )
            split_results[split]["fraction_improved_ade"] = float(
                (
                    collected["per_sample"]["ade"]
                    < baseline_collected[split]["per_sample"]["ade"]
                ).float().mean().item()
            )
        results["variants"][variant_name] = {
            "input_semantics": VARIANTS[variant_name].input_semantics(),
            "trainable_parameters": _trainable_parameter_count(model),
            "capacity_match": _capacity_match_metadata(
                model,
                sample=datasets["train"][0],
                variant_name=variant_name,
                config=config,
            ),
            "splits": split_results,
            "history": history,
        }

    config_hash = _stable_hash(config)
    cache_index_hash = _file_hash(cache_root / "index.json")
    checkpoint_manifest = {}
    for variant_name in variants:
        checkpoint_path = output_dir / "checkpoints" / f"{variant_name}.pt"
        checkpoint_manifest[variant_name] = {
            "path": checkpoint_path.relative_to(output_dir).as_posix(),
            "sha256": _file_hash(checkpoint_path),
        }
    selection_identity = (
        f"{config_hash}:{cache_index_hash}:{','.join(variants)}:"
        f"{_stable_hash(CHECKPOINT_SELECTION_METRIC)}"
    )
    if selection_owner is not None:
        selection_identity += f":{_stable_hash(selection_owner)}"
    selection_id = hashlib.sha256(selection_identity.encode("utf-8")).hexdigest()[:20]
    results["selection_id"] = selection_id
    _write_json_atomic(output_dir / "results.json", results)
    selection_manifest = {
        "schema_version": 2,
        "selection_id": selection_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "selection_complete",
        "test_opened": False,
        "config_sha256": config_hash,
        "cache_index_sha256": cache_index_hash,
        "checkpoint_selection_metric": deepcopy(CHECKPOINT_SELECTION_METRIC),
        "variants": list(variants),
        "variant_specs": {
            variant_name: VARIANTS[variant_name].input_semantics()
            for variant_name in variants
        },
        "checkpoints": checkpoint_manifest,
        "results": "results.json",
    }
    if selection_owner is not None:
        selection_manifest["selection_owner"] = deepcopy(selection_owner)
    _write_json_atomic(output_dir / SELECTION_MANIFEST, selection_manifest)
    return results


def finalize_selection(
    *,
    selection_dir: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    device_name: str = "auto",
    _study_owner_id: str | None = None,
    _study_finalize_capability: object | None = None,
) -> dict[str, Any]:
    """Open test once for a hash-locked selection and reuse its saved checkpoints."""

    selection_dir = Path(selection_dir)
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    _require_fresh_output(output_dir)
    manifest_path = selection_dir / SELECTION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "selection_complete" or manifest.get("test_opened") is not False:
        raise RuntimeError("selection is incomplete or its test split has already been opened")
    marker_path = selection_dir / TEST_OPEN_MARKER
    if marker_path.exists():
        raise RuntimeError(f"selection test has already been opened: {marker_path}")

    selection_owner = _validated_selection_owner(manifest.get("selection_owner"))
    if selection_owner is not None:
        if _study_finalize_capability is not _STUDY_FINALIZE_CAPABILITY:
            raise RuntimeError(
                "selection is owned by a multi-seed study and cannot be finalized standalone"
            )
        if _study_owner_id != selection_owner["study_owner_id"]:
            raise ValueError("study-owned selection parent identity mismatch")
    elif _study_owner_id is not None or _study_finalize_capability is not None:
        raise ValueError("study finalization authorization was supplied for an unowned selection")

    selection_results = json.loads((selection_dir / manifest["results"]).read_text(encoding="utf-8"))
    results_owner = _validated_selection_owner(selection_results.get("selection_owner"))
    if results_owner != selection_owner:
        raise ValueError("selection owner differs between manifest and locked results")
    locked_config = selection_results.get("configuration")
    if not isinstance(locked_config, dict):
        raise ValueError("selection results do not contain a locked configuration")
    config_hash = _stable_hash(locked_config)
    if config is not None and _stable_hash(config) != config_hash:
        raise ValueError("supplied final config does not match the locked selection config hash")
    config = locked_config
    validate_config(config)
    cache_index_hash = _file_hash(cache_root / "index.json")
    if config_hash != manifest.get("config_sha256"):
        raise ValueError("locked selection config does not match its manifest hash")
    if cache_index_hash != manifest.get("cache_index_sha256"):
        raise ValueError("final cache index does not match the locked selection cache hash")
    variants = tuple(manifest.get("variants", ()))
    if not variants or any(variant not in VARIANTS for variant in variants):
        raise ValueError("selection manifest has invalid variants")
    expected_variant_specs = {
        variant_name: VARIANTS[variant_name].input_semantics() for variant_name in variants
    }
    if manifest.get("variant_specs") != expected_variant_specs:
        raise ValueError("selection manifest input semantics do not match this implementation")
    if manifest.get("checkpoint_selection_metric") != CHECKPOINT_SELECTION_METRIC:
        raise ValueError(
            "selection manifest checkpoint metric is not validation equal-route macro ADE"
        )
    locked_checkpoints: dict[str, dict[str, Any]] = {}
    for variant_name in variants:
        checkpoint_info = manifest["checkpoints"][variant_name]
        checkpoint_path = selection_dir / checkpoint_info["path"]
        if _file_hash(checkpoint_path) != checkpoint_info["sha256"]:
            raise ValueError(f"checkpoint hash mismatch for {variant_name}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("variant") != variant_name
            or _stable_hash(checkpoint.get("config")) != config_hash
            or checkpoint.get("input_semantics") != expected_variant_specs[variant_name]
            or checkpoint.get("selection_metric") != CHECKPOINT_SELECTION_METRIC
        ):
            raise ValueError(f"checkpoint metadata mismatch for {variant_name}")
        best_route_macro_ade = checkpoint.get("best_val_route_macro_ade")
        best_ade_alias = checkpoint.get("best_val_ade")
        if (
            isinstance(best_route_macro_ade, bool)
            or not isinstance(best_route_macro_ade, (int, float))
            or isinstance(best_ade_alias, bool)
            or not isinstance(best_ade_alias, (int, float))
            or not np.isfinite(float(best_route_macro_ade))
            or not np.isfinite(float(best_ade_alias))
            or float(best_ade_alias) != float(best_route_macro_ade)
        ):
            raise ValueError(
                f"checkpoint best validation route-macro ADE is invalid for {variant_name}"
            )
        locked_checkpoints[variant_name] = checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "selection_id": manifest["selection_id"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "cache_index_sha256": cache_index_hash,
        "config_sha256": config_hash,
        "final_output": str(output_dir.resolve()),
    }
    # Exclusive creation is the concurrency-safe, permanent test-open gate.
    with marker_path.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    manifest["test_opened"] = True
    manifest["test_opened_at"] = marker["opened_at"]
    manifest["final_output"] = marker["final_output"]
    _write_json_atomic(manifest_path, manifest)

    # From this point onward the test is considered opened even if evaluation fails.
    if bool(config["data"].get("require_deep_audit", True)):
        cache_audit = audit_cache(cache_root, deep=True, deep_splits={"test"})
        if cache_audit["status"] != "pass":
            raise ValueError(f"test cache audit failed after opening: {cache_audit['errors'][:5]}")
    else:
        cache_audit = {"status": "skipped", "deep": False}

    device = resolve_device(device_name)
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config["training"].get("torch_num_threads", 4)))
    test_dataset = _dataset(cache_root, "test", config)
    test_loader = _loader(test_dataset, config, shuffle=False)
    baseline_collected = collect_predictions(test_loader, device=device)
    worst_indices = worst_fraction_indices(
        baseline_collected["per_sample"]["ade"],
        float(config["evaluation"]["worst_fraction"]),
    )
    cache_index = load_index(cache_root)
    results: dict[str, Any] = {
        "status": "completed",
        "evaluation_mode": "final",
        "test_opened": True,
        "selection_id": manifest["selection_id"],
        "selection_run": str(selection_dir.resolve()),
        "evidence_level": "synthetic_smoke"
        if cache_index.get("source", {}).get("kind") == "synthetic"
        else "offline_cache_evaluation",
        "device": str(device),
        "seed": seed,
        "reproducibility": deterministic_runtime_metadata(),
        "checkpoint_selection_metric": deepcopy(CHECKPOINT_SELECTION_METRIC),
        "cache_source": cache_index.get("source", {}),
        "cache_audit": cache_audit,
        "metric_semantics": (
            {"ade": "mean_checkpoint_displacement_m", "fde": "last_checkpoint_error_m"}
            if cache_index.get("source", {}).get("target_semantics") == "geometric_path"
            else {"ade": "average_displacement_error_m", "fde": "final_displacement_error_m"}
        ),
        "dataset_windows": {"test": len(test_dataset)},
        "baseline": {
            "test": summarize_predictions(baseline_collected, worst_indices=worst_indices)
        },
        "variants": {},
        "configuration": deepcopy(config),
    }
    if selection_owner is not None:
        results["selection_owner"] = deepcopy(selection_owner)

    for variant_name in variants:
        checkpoint = locked_checkpoints[variant_name]
        model = _model_from_sample(test_dataset[0], variant_name, config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        collected = collect_predictions(
            test_loader,
            device=device,
            model=model,
            variant_name=variant_name,
            seed=seed,
        )
        if collected["sample_ids"] != baseline_collected["sample_ids"]:
            raise RuntimeError("variant test order differs from the shared baseline order")
        if collected["route_ids"] != baseline_collected["route_ids"]:
            raise RuntimeError("variant test route order differs from the shared baseline order")
        split_result = summarize_predictions(collected, worst_indices=worst_indices)
        split_result["paired_vs_baseline"] = paired_route_comparison(
            collected["per_sample"],
            baseline_collected["per_sample"],
            collected["route_ids"],
            bootstrap_seed=seed,
        )
        split_result["paired_delta_ade_vs_baseline"] = float(
            (collected["per_sample"]["ade"] - baseline_collected["per_sample"]["ade"]).mean().item()
        )
        split_result["fraction_improved_ade"] = float(
            (collected["per_sample"]["ade"] < baseline_collected["per_sample"]["ade"])
            .float()
            .mean()
            .item()
        )
        results["variants"][variant_name] = {
            "input_semantics": expected_variant_specs[variant_name],
            "trainable_parameters": _trainable_parameter_count(model),
            "capacity_match": _capacity_match_metadata(
                model,
                sample=test_dataset[0],
                variant_name=variant_name,
                config=config,
            ),
            # Backward-compatible alias plus explicit scientific aggregation.
            "selected_best_val_ade": float(checkpoint["best_val_route_macro_ade"]),
            "selected_best_val_route_macro_ade": float(
                checkpoint["best_val_route_macro_ade"]
            ),
            "splits": {"test": split_result},
        }

    _write_json_atomic(output_dir / "results.json", results)
    return results
