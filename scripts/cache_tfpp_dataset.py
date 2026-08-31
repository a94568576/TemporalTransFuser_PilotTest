#!/usr/bin/env python3
"""Cache frozen TransFuser++ features and paths without modifying upstream."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import jsonpickle
import torch
from torch.utils.data import DataLoader, Dataset

from temporal_tf.cache import CacheWriter
from temporal_tf.raw_collection import (
    assert_collection_key_matches_route,
    assert_exact_identity_set,
    audit_raw_collection,
    load_collection_root_manifest,
)
from temporal_tf.tfpp_hook import FeatureCapture, select_base_path


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item["_cache_index"] = index
        return item


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_path(value: Any) -> Path:
    if isinstance(value, bytes):
        return Path(value.decode("utf-8"))
    return Path(str(value))


def _raw_source_provenance(
    data_roots: list[Path],
    *,
    collection_keys: list[str],
    require_successful_results: bool,
) -> list[dict[str, Any]]:
    if len(data_roots) != len(collection_keys):
        raise ValueError("collection key/root cardinality mismatch")
    report = audit_raw_collection(data_roots)
    if require_successful_results and report["status"] != "pass":
        details = "; ".join(report["errors"][:12])
        if len(report["errors"]) > 12:
            details += f"; ... {len(report['errors']) - 12} more errors"
        raise ValueError(
            "raw collection preflight failed under the local upstream CARLA_Data policy: "
            + details
        )
    audits_by_root = {
        str(Path(result["result_root"]).resolve()): result for result in report["results"]
    }

    sources: list[dict[str, Any]] = []
    for key, root in zip(collection_keys, data_roots, strict=True):
        result_path = root / "result.json"
        result_status: dict[str, Any] | None = None
        result_hash: str | None = None
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            global_record = result.get("_checkpoint", {}).get("global_record", {})
            score_route = global_record.get("scores_mean", {}).get("score_route")
            exceptions = global_record.get("meta", {}).get("exceptions", [])
            successful = (
                result.get("entry_status") == "Finished"
                and score_route is not None
                and float(score_route) >= 99.999
                and not exceptions
            )
            result_status = {
                "entry_status": result.get("entry_status"),
                "global_status": global_record.get("status"),
                "route_completion": float(score_route) if score_route is not None else None,
                "exceptions": exceptions,
                "successful": successful,
            }
            result_hash = _sha256(result_path)
            if require_successful_results and not successful:
                raise ValueError(f"collection result is not a verified success: {result_path}")
        elif require_successful_results:
            raise FileNotFoundError(f"collection result is missing: {result_path}")

        route_paths = sorted(
            child.resolve()
            for child in root.iterdir()
            if child.is_dir() and (child / "measurements").is_dir()
        )
        if require_successful_results and len(route_paths) != 1:
            raise ValueError(
                f"expected exactly one collected route under {root}, found {len(route_paths)}"
            )
        if len(route_paths) == 1:
            assert_collection_key_matches_route(key, route_paths[0])
        audited = audits_by_root.get(str(root.resolve()))
        sources.append(
            {
                "collection_key": key,
                "root": str(root),
                "result_path": str(result_path) if result_path.is_file() else None,
                "result_sha256": result_hash,
                "result_status": result_status,
                "route_directories": [path.name for path in route_paths],
                "route_directory_paths": [str(path) for path in route_paths],
                "tfpp_loader_acceptance": (
                    audited["tfpp_loader_acceptance"] if audited is not None else None
                ),
                "raw_audit_status": audited["status"] if audited is not None else "fail",
            }
        )
    return sources


def _route_id_for_directory(
    route_dir: Path, *, data_roots: list[Path], dataset_id: str
) -> str:
    route_dir = route_dir.resolve()
    for root_index, data_root in enumerate(data_roots):
        try:
            relative_route = route_dir.relative_to(data_root)
        except ValueError:
            continue
        return f"{dataset_id}/root_{root_index:02d}/{relative_route.as_posix()}"
    raise RuntimeError(f"route {route_dir} is not under a configured data root")


def _dataset_route_directories(dataset: Any) -> set[str]:
    route_directories: set[str] = set()
    for measurement_value in dataset.measurements:
        if getattr(measurement_value, "ndim", 0) > 0:
            measurement_value = measurement_value[0]
        route_directories.add(str(_decode_path(measurement_value).parent.resolve()))
    return route_directories


def _measurement_identity(
    dataset: Any,
    index: int,
    *,
    data_roots: list[Path],
    dataset_id: str,
) -> tuple[str, int, torch.Tensor]:
    measurement_value = dataset.measurements[index]
    if getattr(measurement_value, "ndim", 0) > 0:
        measurement_value = measurement_value[0]
    measurement_dir = _decode_path(measurement_value)
    frame_id = int(dataset.sample_start[index]) + int(dataset.config.seq_len) - 1
    measurement_path = measurement_dir / f"{frame_id:04d}.json.gz"
    with gzip.open(measurement_path, "rt", encoding="utf-8") as stream:
        measurement = jsonpickle.decode(stream.read())
    position = measurement.get("pos_global")
    if position is None:
        matrix = torch.as_tensor(measurement["ego_matrix"], dtype=torch.float64)
        position = [float(matrix[0, 3]), float(matrix[1, 3])]
    pose = torch.tensor(
        [float(position[0]), float(position[1]), float(measurement["theta"])],
        dtype=torch.float64,
    )
    route_dir = measurement_dir.parent.resolve()
    route_id = _route_id_for_directory(
        route_dir, data_roots=data_roots, dataset_id=dataset_id
    )
    return route_id, frame_id, pose


def main() -> int:
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]
    workspace = project_root.parent
    parser.add_argument(
        "--tfpp-root",
        type=Path,
        default=workspace / "transfuser_test" / "carla_garage",
    )
    parser.add_argument("--model-dir", type=Path)
    data_input = parser.add_mutually_exclusive_group(required=True)
    data_input.add_argument("--data-root", type=Path, nargs="+")
    data_input.add_argument(
        "--data-root-manifest",
        type=Path,
        help="explicit LOGICAL_KEY ROOT manifest; relative roots resolve beside the manifest",
    )
    parser.add_argument(
        "--expected-route-count",
        type=int,
        help="fail before and after upstream loading unless this exact route count is preserved",
    )
    parser.add_argument(
        "--dataset-id",
        default="tfpp_dataset",
        help="portable stable ID used in route identities and split provenance",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--feature-source",
        choices=["backbone_bev", "planner_grid"],
        default="backbone_bev",
    )
    parser.add_argument(
        "--cache-spatial-size",
        type=int,
        help="default: 16 for backbone_bev, native 8 for planner_grid; upsampling is rejected",
    )
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.7, 0.15, 0.15),
        help="route-level split ratios; use 0.6 0.2 0.2 for the 15-route pilot",
    )
    parser.add_argument(
        "--require-successful-collection-results",
        action="store_true",
        help=(
            "require every root to pass result/frame integrity and the local upstream "
            "CARLA_Data results.json.gz route filter"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.expected_route_count is not None and args.expected_route_count < 1:
        parser.error("--expected-route-count must be positive")
    cache_spatial_size = args.cache_spatial_size
    if cache_spatial_size is None:
        cache_spatial_size = 16 if args.feature_source == "backbone_bev" else 8
    if cache_spatial_size < 1:
        parser.error("--cache-spatial-size must be positive")
    if not args.dataset_id.strip() or "/" in args.dataset_id or "\\" in args.dataset_id:
        parser.error("--dataset-id must be a non-empty path-free identifier")
    if any(value < 0.0 for value in args.split_ratios) or sum(args.split_ratios) <= 0.0:
        parser.error("--split-ratios must be non-negative and sum to a positive value")
    split_ratios = dict(zip(("train", "val", "test"), args.split_ratios, strict=True))

    tfpp_root = args.tfpp_root.resolve()
    root_manifest: dict[str, Any] | None = None
    if args.data_root_manifest is not None:
        try:
            root_entries = load_collection_root_manifest(args.data_root_manifest)
        except ValueError as exc:
            parser.error(str(exc))
        data_roots = [Path(entry["root"]) for entry in root_entries]
        collection_keys = [entry["key"] for entry in root_entries]
        root_manifest = {
            "path": str(args.data_root_manifest.resolve()),
            "sha256": _sha256(args.data_root_manifest.resolve()),
            "entries": root_entries,
        }
    else:
        data_roots = [path.resolve() for path in args.data_root]
        collection_keys = [path.name for path in data_roots]
    if len(set(data_roots)) != len(data_roots):
        parser.error("configured data roots must be unique")
    if len(set(collection_keys)) != len(collection_keys):
        parser.error("configured collection keys must be unique")
    if args.expected_route_count is not None and len(data_roots) != args.expected_route_count:
        parser.error(
            f"--expected-route-count is {args.expected_route_count}, but input declares "
            f"{len(data_roots)} roots"
        )
    strict_route_contract = (
        args.require_successful_collection_results
        or args.data_root_manifest is not None
        or args.expected_route_count is not None
    )
    if strict_route_contract and args.max_samples is not None:
        parser.error("--max-samples is incompatible with the exact route-set contract")
    try:
        raw_sources = _raw_source_provenance(
            data_roots,
            collection_keys=collection_keys,
            require_successful_results=strict_route_contract,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    expected_route_directories = {
        route_path
        for source in raw_sources
        for route_path in source["route_directory_paths"]
    }
    if strict_route_contract:
        expected_count = args.expected_route_count or len(data_roots)
        if len(expected_route_directories) != expected_count:
            raise RuntimeError(
                f"preflight accepted {len(expected_route_directories)} exact routes; "
                f"expected {expected_count}"
            )
    model_dir = (args.model_dir or tfpp_root / "pretrained_models" / "all_towns").resolve()
    checkpoint = model_dir / "model_0030_0.pth"
    team_code = tfpp_root / "team_code"
    carla_python = workspace / "old_carla" / "PythonAPI" / "carla"
    sys.path.insert(0, str(team_code))
    sys.path.insert(0, str(carla_python))
    os.environ.setdefault("TFPP_TIMM_PRETRAINED_INIT", "0")

    from config import GlobalConfig  # noqa: PLC0415
    from data import CARLA_Data  # noqa: PLC0415
    from model import LidarCenterNet  # noqa: PLC0415

    with (model_dir / "config.json").open("rt", encoding="utf-8") as stream:
        loaded_config = jsonpickle.decode(stream.read())
    config = GlobalConfig()
    config.__dict__.update(loaded_config.__dict__)
    # Frozen cache inference must be deterministic and augmentation-free.
    config.compile = False

    # The frozen model keeps its original architecture flags for strict state
    # loading.  The dataset copy disables every augmentation/auxiliary label so
    # cache extraction reads only the sensor/planning inputs that forward needs.
    dataset_config = copy.deepcopy(config)
    dataset_config.augment = False
    dataset_config.use_color_aug = False
    dataset_config.color_aug_prob = 0.0
    dataset_config.lidar_aug_prob = 0.0
    dataset_config.use_cutout = False
    dataset_config.use_semantic = False
    dataset_config.use_depth = False
    dataset_config.detect_boxes = False
    dataset_config.use_bev_semantic = False
    base_dataset = CARLA_Data(
        root=[str(path) for path in data_roots],
        config=dataset_config,
        shared_dict=None,
        validation=False,
    )
    if len(base_dataset) == 0:
        raise RuntimeError("CARLA_Data found no valid frames under --data-root")
    loaded_route_directories = _dataset_route_directories(base_dataset)
    if strict_route_contract:
        assert_exact_identity_set(
            expected_route_directories,
            loaded_route_directories,
            stage="upstream CARLA_Data post-load",
        )
    expected_route_ids = {
        _route_id_for_directory(Path(route_directory), data_roots=data_roots, dataset_id=args.dataset_id)
        for route_directory in expected_route_directories
    }
    loader = DataLoader(
        IndexedDataset(base_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for tiny diagnostics")
    model = LidarCenterNet(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    # Keep the planning graph, but skip unused auxiliary decoders at inference.
    config.use_semantic = False
    config.use_depth = False
    config.detect_boxes = False
    if args.feature_source == "backbone_bev" and not hasattr(model, "bev_semantic_decoder"):
        raise RuntimeError("checkpoint has no BEV auxiliary path required by backbone_bev capture")
    config.use_bev_semantic = args.feature_source == "backbone_bev"
    model.requires_grad_(False).eval().to(device)
    writer = CacheWriter(args.output)
    written = 0
    written_route_ids: set[str] = set()

    with FeatureCapture(
        model,
        source=args.feature_source,
        cache_spatial_size=cache_spatial_size,
    ) as capture, torch.inference_mode():
        for batch in loader:
            indices = batch.pop("_cache_index")
            rgb = batch["rgb"].to(device=device, dtype=torch.float32)
            lidar_key = "temporal_lidar" if config.lidar_seq_len > 1 else "lidar"
            outputs = model(
                rgb=rgb,
                lidar_bev=batch[lidar_key].to(device=device, dtype=torch.float32),
                target_point=batch["target_point"].to(device=device, dtype=torch.float32),
                ego_vel=batch["speed"].to(device=device, dtype=torch.float32).unsqueeze(1),
                command=batch["command"].to(device=device, dtype=torch.float32),
                target_point_next=(
                    batch["target_point_next"].to(device=device, dtype=torch.float32)
                    if config.two_tp_input
                    else None
                ),
            )
            prediction, representation = select_base_path(outputs)
            feature = capture.pop()
            if representation == "pred_checkpoint":
                target = batch["route"][:, : config.predict_checkpoint_len]
                target_semantics = "geometric_path"
            else:
                if "ego_waypoints" not in batch:
                    raise RuntimeError("pred_wp requires ego_waypoints labels in the dataset")
                target = batch["ego_waypoints"]
                target_semantics = "time_sampled_trajectory"
            if tuple(prediction.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"prediction/label mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
                )

            for batch_index, dataset_index_value in enumerate(indices.tolist()):
                route_id, frame_id, pose = _measurement_identity(
                    base_dataset,
                    int(dataset_index_value),
                    data_roots=data_roots,
                    dataset_id=args.dataset_id,
                )
                writer.add(
                    {
                        "bev_feature": feature[batch_index],
                        "pred_trajectory": prediction[batch_index],
                        "gt_trajectory": target[batch_index],
                        "ego_pose": pose,
                        "route_id": route_id,
                        "frame_id": frame_id,
                        "timestamp": frame_id * config.data_save_freq / config.carla_fps,
                        # Exact causal scalar/vector inputs supplied to the
                        # frozen planner.  Cache schema v3 retains these for
                        # provenance and input-conditioned analysis only; the
                        # temporal adapter dataset does not consume them.
                        "speed_t": batch["speed"][batch_index],
                        "command_t": batch["command"][batch_index],
                        "trajectory_source": "frozen_model_prediction",
                        "metadata": {
                            "representation": representation,
                            "target_semantics": target_semantics,
                            "feature_source": args.feature_source,
                        },
                    }
                )
                written += 1
                written_route_ids.add(route_id)
                if args.max_samples is not None and written >= args.max_samples:
                    break
            if args.max_samples is not None and written >= args.max_samples:
                break

    if strict_route_contract:
        assert_exact_identity_set(
            expected_route_ids,
            written_route_ids,
            stage="cache extraction post-write",
        )
        if written != len(base_dataset):
            raise RuntimeError(
                f"cache extraction wrote {written} records from a {len(base_dataset)}-sample "
                "upstream dataset"
            )

    index = writer.finalize(
        source={
            "kind": "frozen_tfpp",
            "dataset_id": args.dataset_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "model_config_sha256": _sha256(model_dir / "config.json"),
            "extractor_sha256": _sha256(Path(__file__).resolve()),
            "tfpp_root": str(tfpp_root),
            "upstream_carla_data_path": str(team_code / "data.py"),
            "upstream_carla_data_sha256": _sha256(team_code / "data.py"),
            "raw_sources": raw_sources,
            "root_manifest": root_manifest,
            "exact_route_contract": {
                "enabled": strict_route_contract,
                "expected_route_count": len(expected_route_directories),
                "collection_keys": collection_keys,
                "expected_route_directories": sorted(expected_route_directories),
                "loaded_route_directories": sorted(loaded_route_directories),
                "expected_cache_route_ids": sorted(expected_route_ids),
                "written_cache_route_ids": sorted(written_route_ids),
            },
            "feature_source": args.feature_source,
            "cache_spatial_size": cache_spatial_size,
            "prediction_representation": representation,
            "target_semantics": target_semantics,
            "cached_optional_inputs": ["speed_t", "command_t"],
            "speed_t_unit": "m/s",
            "command_t_encoding": "TF++ six-way one-hot",
            "upstream_frozen": True,
        },
        split_ratios=split_ratios,
        split_seed=args.split_seed,
    )
    print(f"cached records: {written}")
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
