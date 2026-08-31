"""Temporal windows over route-safe frozen-model cache records."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .cache import load_index, validate_record
from .geometry import transform_trajectory_between_egos


class TemporalCacheDataset(Dataset):
    """Build history windows without ever crossing route or split boundaries."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str,
        history_length: int = 4,
        align_past_trajectories: bool = True,
        max_frame_gap: int | None = 1,
        validate_on_load: bool = False,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {split}")
        self.cache_root = Path(cache_root)
        self.history_length = history_length
        self.align_past_trajectories = align_past_trajectories
        self.max_frame_gap = max_frame_gap
        self.validate_on_load = validate_on_load
        index = load_index(self.cache_root)

        by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
        route_splits: dict[str, set[str]] = defaultdict(set)
        for entry in index["records"]:
            route_splits[entry["route_id"]].add(entry["split"])
            if entry["split"] == split:
                by_route[entry["route_id"]].append(entry)
        leaking = {route: splits for route, splits in route_splits.items() if len(splits) != 1}
        if leaking:
            raise ValueError(f"route split leakage detected: {leaking}")

        self.windows: list[tuple[dict[str, Any], ...]] = []
        for route_id, entries in sorted(by_route.items()):
            entries.sort(key=lambda entry: (entry["frame_id"], entry["timestamp"]))
            for end in range(history_length, len(entries)):
                window = entries[end - history_length : end + 1]
                frame_ids = [int(entry["frame_id"]) for entry in window]
                gaps = [right - left for left, right in zip(frame_ids, frame_ids[1:])]
                if any(gap <= 0 for gap in gaps):
                    raise ValueError(f"duplicate or unsorted frames in route {route_id}: {frame_ids}")
                if max_frame_gap is not None and any(gap > max_frame_gap for gap in gaps):
                    continue
                self.windows.append(tuple(window))
        if not self.windows:
            raise ValueError(
                f"split '{split}' has no valid windows of history_length={history_length}; "
                "collect more sequential frames or adjust max_frame_gap"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, entry: dict[str, Any]) -> dict[str, Any]:
        path = self.cache_root / entry["path"]
        record = torch.load(path, map_location="cpu", weights_only=True)
        if self.validate_on_load:
            validate_record(record)
        if record["route_id"] != entry["route_id"] or int(record["frame_id"]) != int(entry["frame_id"]):
            raise ValueError(f"index/record identity mismatch: {path}")
        return record

    def __getitem__(self, index: int) -> dict[str, Any]:
        entries = self.windows[index]
        records = [self._load(entry) for entry in entries]
        past, current = records[:-1], records[-1]
        past_trajectories = torch.stack([record["pred_trajectory"].float() for record in past])
        past_poses = torch.stack([record["ego_pose"].double() for record in past])
        current_pose = current["ego_pose"].double()
        if self.align_past_trajectories:
            target_poses = current_pose.unsqueeze(0).expand_as(past_poses)
            past_trajectories = transform_trajectory_between_egos(
                past_trajectories, past_poses, target_poses
            ).float()

        return {
            "past_bev": torch.stack([record["bev_feature"].float() for record in past]),
            "current_bev": current["bev_feature"].float(),
            "past_trajectory": past_trajectories,
            "current_trajectory": current["pred_trajectory"].float(),
            "gt_trajectory": current["gt_trajectory"].float(),
            "past_ego_pose": past_poses,
            "current_ego_pose": current_pose,
            "route_id": current["route_id"],
            "frame_id": int(current["frame_id"]),
            "sample_id": f"{current['route_id']}::{int(current['frame_id'])}",
        }

    @property
    def sample_shape(self) -> dict[str, tuple[int, ...]]:
        sample = self[0]
        return {
            key: tuple(sample[key].shape)
            for key in (
                "past_bev",
                "current_bev",
                "past_trajectory",
                "current_trajectory",
                "gt_trajectory",
            )
        }
