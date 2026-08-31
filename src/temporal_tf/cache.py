"""Versioned, route-aware cache format for frozen planner outputs."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


SCHEMA_VERSION = 3
INDEX_FILENAME = "index.json"
REQUIRED_TENSOR_KEYS = ("bev_feature", "pred_trajectory", "gt_trajectory", "ego_pose")
OPTIONAL_INPUT_ANALYSIS_KEYS = ("speed_t", "command_t")
COMMAND_DIM = 6
ALLOWED_RECORD_KEYS = {
    "schema_version",
    "bev_feature",
    "pred_trajectory",
    "gt_trajectory",
    "ego_pose",
    "route_id",
    "frame_id",
    "timestamp",
    "speed_t",
    "command_t",
    "trajectory_source",
    "metadata",
}


def _as_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()
    return tensor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: Mapping[str, Any]) -> None:
    """Reject malformed or leakage-prone records before they enter a cache."""

    unexpected = set(record).difference(ALLOWED_RECORD_KEYS)
    if unexpected:
        raise ValueError(f"cache record contains non-allowlisted keys: {sorted(unexpected)}")
    schema_version = record.get("schema_version")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cache record schema {schema_version}; expected {SCHEMA_VERSION}"
        )
    missing = [key for key in (*REQUIRED_TENSOR_KEYS, "route_id", "frame_id") if key not in record]
    if missing:
        raise ValueError(f"cache record is missing keys: {missing}")

    bev = torch.as_tensor(record["bev_feature"])
    pred = torch.as_tensor(record["pred_trajectory"])
    gt = torch.as_tensor(record["gt_trajectory"])
    pose = torch.as_tensor(record["ego_pose"])
    if bev.ndim != 3:
        raise ValueError(f"bev_feature must be [C,H,W], got {tuple(bev.shape)}")
    if pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError(f"pred_trajectory must be [N,2], got {tuple(pred.shape)}")
    if gt.shape != pred.shape:
        raise ValueError(f"gt_trajectory {tuple(gt.shape)} must match prediction {tuple(pred.shape)}")
    if pose.shape != (3,):
        raise ValueError(f"ego_pose must be [x,y,yaw], got {tuple(pose.shape)}")
    if not str(record["route_id"]).strip():
        raise ValueError("route_id must be non-empty")
    if int(record["frame_id"]) < 0:
        raise ValueError("frame_id must be non-negative")
    if "timestamp" in record and not math.isfinite(float(record["timestamp"])):
        raise ValueError("timestamp must be finite")
    for name, tensor in (("bev_feature", bev), ("pred_trajectory", pred), ("gt_trajectory", gt), ("ego_pose", pose)):
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"{name} contains NaN or Inf")

    # These are causal inputs to the frozen planner, retained only for
    # provenance and later input-conditioned analysis.  TemporalCacheDataset
    # deliberately does not expose them to the adapter.
    if "speed_t" in record:
        speed = torch.as_tensor(record["speed_t"])
        if speed.numel() != 1 or speed.ndim > 1:
            raise ValueError(f"speed_t must be a scalar, got {tuple(speed.shape)}")
        if speed.dtype == torch.bool or speed.is_complex():
            raise ValueError("speed_t must be a real numeric scalar")
        speed_value = speed.float().reshape(())
        if not torch.isfinite(speed_value):
            raise ValueError("speed_t must be finite")
    if "command_t" in record:
        command = torch.as_tensor(record["command_t"])
        if command.shape != (COMMAND_DIM,):
            raise ValueError(
                f"command_t must be a [{COMMAND_DIM}] one-hot vector, got {tuple(command.shape)}"
            )
        if command.dtype == torch.bool or command.is_complex():
            raise ValueError("command_t must be a real numeric one-hot vector")
        command_float = command.float()
        if not torch.isfinite(command_float).all():
            raise ValueError("command_t contains NaN or Inf")
        if not torch.all((command_float == 0.0) | (command_float == 1.0)):
            raise ValueError("command_t must contain only exact zero/one values")
        if int(torch.count_nonzero(command_float).item()) != 1:
            raise ValueError("command_t must contain exactly one active class")
    source = record.get("trajectory_source")
    if source != "frozen_model_prediction":
        raise ValueError(
            "trajectory_source must be exactly 'frozen_model_prediction'; "
            "GT history is forbidden as adapter input"
        )
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping of non-tensor provenance scalars")
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("metadata may contain only string keys and scalar provenance values")


def canonical_record(record: Mapping[str, Any], *, bev_dtype: torch.dtype = torch.float16) -> dict[str, Any]:
    """Return the stable on-disk representation of a validated record."""

    # Validate the caller's provenance claim before canonicalization.  Never
    # "upgrade" an untrusted/GT source merely by rewriting the source string.
    validate_record(record)
    result = dict(record)
    result["schema_version"] = SCHEMA_VERSION
    result["bev_feature"] = _as_cpu_tensor(record["bev_feature"], dtype=bev_dtype)
    result["pred_trajectory"] = _as_cpu_tensor(record["pred_trajectory"], dtype=torch.float32)
    result["gt_trajectory"] = _as_cpu_tensor(record["gt_trajectory"], dtype=torch.float32)
    result["ego_pose"] = _as_cpu_tensor(record["ego_pose"], dtype=torch.float64)
    result["route_id"] = str(record["route_id"])
    result["frame_id"] = int(record["frame_id"])
    result["timestamp"] = float(record.get("timestamp", record["frame_id"]))
    if "speed_t" in record:
        result["speed_t"] = _as_cpu_tensor(record["speed_t"], dtype=torch.float32).reshape(())
    if "command_t" in record:
        result["command_t"] = _as_cpu_tensor(
            record["command_t"], dtype=torch.float32
        ).reshape(COMMAND_DIM)
    result["trajectory_source"] = "frozen_model_prediction"
    validate_record(result)
    return result


def _route_slug(route_id: str) -> str:
    digest = hashlib.sha256(route_id.encode("utf-8")).hexdigest()[:12]
    readable = "".join(char if char.isalnum() or char in "-_" else "_" for char in route_id)[-48:]
    return f"{readable}-{digest}"


def assign_route_splits(
    route_ids: Iterable[str],
    ratios: Mapping[str, float] | None = None,
    *,
    seed: int = 17,
) -> dict[str, str]:
    """Deterministically split whole routes; no frame can cross a boundary."""

    ratios = dict(ratios or {"train": 0.7, "val": 0.15, "test": 0.15})
    expected = {"train", "val", "test"}
    if set(ratios) != expected or any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must contain non-negative train/val/test values")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive number")
    ratios = {key: value / total for key, value in ratios.items()}

    routes = sorted(set(str(route_id) for route_id in route_ids))
    random.Random(seed).shuffle(routes)
    count = len(routes)
    if count == 0:
        return {}

    counts = {name: int(math.floor(ratios[name] * count)) for name in ("train", "val", "test")}
    if count >= 3:
        for name in ("train", "val", "test"):
            if ratios[name] > 0 and counts[name] == 0:
                counts[name] = 1
    while sum(counts.values()) > count:
        candidates = [name for name in counts if counts[name] > (1 if count >= 3 and ratios[name] > 0 else 0)]
        if not candidates:
            break
        counts[max(candidates, key=lambda name: counts[name] - ratios[name] * count)] -= 1
    while sum(counts.values()) < count:
        name = max(counts, key=lambda key: ratios[key] * count - counts[key])
        counts[name] += 1

    result: dict[str, str] = {}
    offset = 0
    for split in ("train", "val", "test"):
        for route_id in routes[offset : offset + counts[split]]:
            result[route_id] = split
        offset += counts[split]
    return result


@dataclass
class CacheWriter:
    """Write immutable per-frame records, then atomically publish an index."""

    root: Path
    bev_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.records_dir = self.root / "records"
        if (self.root / INDEX_FILENAME).exists():
            raise FileExistsError(f"cache already finalized: {self.root / INDEX_FILENAME}")
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []
        self._seen: set[tuple[str, int]] = set()
        self._finalized = False

    def add(self, record: Mapping[str, Any]) -> Path:
        if self._finalized:
            raise RuntimeError("cannot add records after cache finalization")
        normalized = canonical_record(record, bev_dtype=self.bev_dtype)
        identity = (normalized["route_id"], normalized["frame_id"])
        if identity in self._seen:
            raise ValueError(f"duplicate route/frame record: {identity}")
        self._seen.add(identity)

        relative = Path("records") / _route_slug(normalized["route_id"]) / f"{normalized['frame_id']:08d}.pt"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite cache record: {destination}")
        torch.save(normalized, destination)
        self._entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(destination),
                "route_id": normalized["route_id"],
                "frame_id": normalized["frame_id"],
                "timestamp": normalized["timestamp"],
            }
        )
        return destination

    def finalize(
        self,
        *,
        source: Mapping[str, Any] | None = None,
        split_ratios: Mapping[str, float] | None = None,
        split_seed: int = 17,
    ) -> Path:
        if self._finalized or (self.root / INDEX_FILENAME).exists():
            raise FileExistsError(f"cache is already finalized: {self.root / INDEX_FILENAME}")
        if not self._entries:
            raise ValueError("cannot finalize an empty cache")
        route_splits = assign_route_splits(
            (entry["route_id"] for entry in self._entries), split_ratios, seed=split_seed
        )
        entries = sorted(self._entries, key=lambda entry: (entry["route_id"], entry["frame_id"]))
        for entry in entries:
            entry["split"] = route_splits[entry["route_id"]]
        index = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": dict(source or {}),
            "trajectory_history_source": "frozen_model_prediction_only",
            "record_hash_algorithm": "sha256",
            "split_policy": {
                "unit": "route",
                "seed": split_seed,
                "ratios": dict(split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15}),
            },
            "records": entries,
        }
        path = self.root / INDEX_FILENAME
        temporary = self.root / f".{INDEX_FILENAME}.tmp"
        temporary.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        self._finalized = True
        return path


def load_index(cache_root: str | Path) -> dict[str, Any]:
    path = Path(cache_root) / INDEX_FILENAME
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema {index.get('schema_version')}; expected {SCHEMA_VERSION}")
    if index.get("trajectory_history_source") != "frozen_model_prediction_only":
        raise ValueError("cache index does not certify frozen-model-only trajectory history")
    if index.get("record_hash_algorithm") != "sha256":
        raise ValueError("cache index does not commit records with SHA256 hashes")
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("cache index records must be a non-empty list")
    required = {"path", "sha256", "route_id", "frame_id", "timestamp", "split"}
    allowed_splits = {"train", "val", "test"}
    identities: set[tuple[str, int]] = set()
    paths: set[str] = set()
    route_splits: dict[str, set[str]] = {}
    for offset, entry in enumerate(records):
        if not isinstance(entry, Mapping) or not required.issubset(entry):
            raise ValueError(f"malformed cache index entry at offset {offset}")
        split = str(entry["split"])
        if split not in allowed_splits:
            raise ValueError(f"invalid split '{split}' at index offset {offset}")
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe record path at index offset {offset}: {relative}")
        relative_string = relative.as_posix()
        if relative_string in paths:
            raise ValueError(f"duplicate record path: {relative_string}")
        paths.add(relative_string)
        record_hash = str(entry["sha256"])
        if len(record_hash) != 64 or any(character not in "0123456789abcdef" for character in record_hash):
            raise ValueError(f"invalid record SHA256 at index offset {offset}")
        route_id = str(entry["route_id"])
        frame_id = int(entry["frame_id"])
        identity = (route_id, frame_id)
        if identity in identities:
            raise ValueError(f"duplicate route/frame identity: {identity}")
        identities.add(identity)
        route_splits.setdefault(route_id, set()).add(split)
        if not math.isfinite(float(entry["timestamp"])):
            raise ValueError(f"non-finite timestamp at index offset {offset}")
    leaking = {route: splits for route, splits in route_splits.items() if len(splits) != 1}
    if leaking:
        raise ValueError(f"route split leakage detected in index: {leaking}")
    return index
