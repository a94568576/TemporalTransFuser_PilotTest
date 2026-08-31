#!/usr/bin/env python3
"""Render deterministic, model-free frozen-cache sanity figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from temporal_tf.cache import load_index
from temporal_tf.geometry import transform_trajectory_between_egos
from temporal_tf.sanity import (
    analyze_cache,
    deterministic_sample_entries,
    raw_frame_metrics,
    route_safe_history_entries,
    write_json_artifact,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
GENERATED_FIGURE_PATTERN = re.compile(r"\d{2}_[0-9a-f]{10}_\d{8}\.png")
TEST_MARKER_FILENAMES = {"test_opened.marker.json", "study_test_opened.marker.json"}


def _load_pyplot():
    config_root = Path(tempfile.gettempdir()) / "temporal_tf_matplotlib"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_root))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise RuntimeError(
            "visualization requires matplotlib at runtime; install it in the active "
            "environment (for example: pip install matplotlib)"
        ) from exc
    return pyplot


def _build_rgb_index(raw_root: Path) -> dict[tuple[str, int], list[Path]]:
    lookup: dict[tuple[str, int], list[Path]] = defaultdict(list)
    if not raw_root.is_dir():
        raise ValueError(f"--raw-root is not a directory: {raw_root}")
    for path in sorted(raw_root.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.parent.name == "rgb"
            and len(path.stem) == 4
            and path.stem.isdigit()
        ):
            route_name = path.parent.parent.name
            lookup[(route_name, int(path.stem))].append(path)
    return lookup


def _resolve_rgb(
    lookup: dict[tuple[str, int], list[Path]], route_id: str, frame_id: int
) -> tuple[Path | None, str | None]:
    route_name = route_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
    candidates = sorted(lookup.get((route_name, frame_id), []))
    if not candidates:
        return None, f"raw RGB not found for route={route_id}, frame={frame_id:04d}"
    if len(candidates) > 1:
        return None, (
            f"ambiguous raw RGB candidates for route={route_id}, frame={frame_id:04d}; "
            "overlay disabled to avoid displaying another route"
        )
    return candidates[0], None


def _load_record(cache_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    return torch.load(
        cache_root / entry["path"],
        map_location="cpu",
        weights_only=True,
    )


def _figure_name(selection_index: int, entry: dict[str, Any]) -> str:
    route_id = str(entry["route_id"])
    route_digest = hashlib.sha256(route_id.encode("utf-8")).hexdigest()[:10]
    return f"{selection_index:02d}_{route_digest}_{int(entry['frame_id']):08d}.png"


def render_cache_visualizations(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    max_samples: int = 20,
    history_length: int = 4,
    raw_root: str | Path | None = None,
    splits: set[str] | None = None,
) -> dict[str, Any]:
    """Render selected cache records and return their metadata manifest."""

    if not 1 <= max_samples <= 20:
        raise ValueError("max_samples must be within [1, 20]")
    if history_length < 0:
        raise ValueError("history_length must be non-negative")

    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    selected_splits = set(splits) if splits is not None else None
    selected_scope: list[str] | str = (
        sorted(selected_splits) if selected_splits is not None else "all"
    )
    sanity = analyze_cache(cache_root, splits=selected_splits)
    if sanity["status"] != "pass":
        error = (
            "cache sanity failed; refusing to visualize invalid data: "
            + "; ".join(sanity["errors"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_artifact(
            {
                "schema_version": 1,
                "kind": "temporal_tf_cache_visualization_manifest",
                "status": "fail",
                "cache_root": str(cache_root.resolve()),
                "output_dir": str(output_dir.resolve()),
                "selected_splits": selected_scope,
                "errors": [error],
            },
            output_dir / "manifest.json",
        )
        raise RuntimeError(error)
    index = load_index(cache_root)
    selected = deterministic_sample_entries(
        index,
        max_samples=max_samples,
        splits=selected_splits,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(
        {
            "schema_version": 1,
            "kind": "temporal_tf_cache_visualization_manifest",
            "status": "in_progress",
            "cache_root": str(cache_root.resolve()),
            "output_dir": str(output_dir.resolve()),
            "selected_splits": selected_scope,
        },
        output_dir / "manifest.json",
    )
    try:
        pyplot = _load_pyplot()
    except RuntimeError as exc:
        write_json_artifact(
            {
                "schema_version": 1,
                "kind": "temporal_tf_cache_visualization_manifest",
                "status": "fail",
                "cache_root": str(cache_root.resolve()),
                "output_dir": str(output_dir.resolve()),
                "selected_splits": selected_scope,
                "errors": [str(exc)],
            },
            output_dir / "manifest.json",
        )
        raise
    expected_figure_names = {
        _figure_name(offset, dict(entry)) for offset, entry in enumerate(selected)
    }
    for stale_path in output_dir.glob("*.png"):
        if (
            GENERATED_FIGURE_PATTERN.fullmatch(stale_path.name)
            and stale_path.name not in expected_figure_names
        ):
            stale_path.unlink()

    warnings = list(sanity["warnings"])
    rgb_lookup: dict[tuple[str, int], list[Path]] = {}
    resolved_raw_root = None
    if raw_root is not None:
        resolved_raw_root = Path(raw_root).resolve()
        rgb_lookup = _build_rgb_index(resolved_raw_root)

    figures: list[dict[str, Any]] = []
    for selection_index, entry_mapping in enumerate(selected):
        entry = dict(entry_mapping)
        current = _load_record(cache_root, entry)
        route_id = str(entry["route_id"])
        frame_id = int(entry["frame_id"])
        history_entries = route_safe_history_entries(
            index, entry, history_length=history_length
        )
        aligned_history: list[tuple[dict[str, Any], torch.Tensor]] = []
        for history_entry_mapping in history_entries:
            history_entry = dict(history_entry_mapping)
            history_record = _load_record(cache_root, history_entry)
            aligned = transform_trajectory_between_egos(
                history_record["pred_trajectory"].double(),
                history_record["ego_pose"].double(),
                current["ego_pose"].double(),
            ).float()
            aligned_history.append((history_entry, aligned))

        bev = current["bev_feature"].float()
        energy = torch.sqrt(torch.mean(bev.square(), dim=0)).numpy()
        rgb_path = None
        rgb_warning = None
        rgb_image = None
        if raw_root is not None:
            rgb_path, rgb_warning = _resolve_rgb(rgb_lookup, route_id, frame_id)
            if rgb_warning:
                warnings.append(rgb_warning)
            if rgb_path is not None:
                try:
                    rgb_image = pyplot.imread(rgb_path)
                except Exception as exc:  # image decoder errors vary by backend
                    rgb_warning = f"raw RGB could not be decoded at {rgb_path}: {exc}"
                    warnings.append(rgb_warning)

        column_count = 3 if rgb_image is not None else 2
        figure, axes = pyplot.subplots(
            1,
            column_count,
            figsize=(5.2 * column_count, 4.8),
            squeeze=False,
            constrained_layout=True,
        )
        axes_row = axes[0]
        heatmap_axis = axes_row[0]
        heatmap = heatmap_axis.imshow(energy, origin="lower", cmap="magma")
        heatmap_axis.set_title("BEV channel RMS energy")
        heatmap_axis.set_xlabel("BEV column")
        heatmap_axis.set_ylabel("BEV row")
        figure.colorbar(heatmap, ax=heatmap_axis, fraction=0.046, pad=0.04)

        path_axis = axes_row[1]
        for history_offset, (history_entry, aligned) in enumerate(aligned_history):
            alpha = 0.2 + 0.5 * (history_offset + 1) / max(1, len(aligned_history))
            path_axis.plot(
                aligned[:, 0].numpy(),
                aligned[:, 1].numpy(),
                color="0.45",
                linewidth=1.0,
                alpha=alpha,
                label="aligned past pred" if history_offset == len(aligned_history) - 1 else None,
            )
        prediction = current["pred_trajectory"].float()
        target = current["gt_trajectory"].float()
        path_axis.plot(
            prediction[:, 0].numpy(),
            prediction[:, 1].numpy(),
            "o-",
            color="tab:red",
            linewidth=2.0,
            markersize=3.5,
            label="frozen prediction",
        )
        path_axis.plot(
            target[:, 0].numpy(),
            target[:, 1].numpy(),
            "o-",
            color="tab:green",
            linewidth=2.0,
            markersize=3.5,
            label="GT",
        )
        path_axis.scatter([0.0], [0.0], marker="x", color="black", label="current ego")
        path_axis.set_aspect("equal", adjustable="datalim")
        path_axis.grid(True, alpha=0.25)
        path_axis.set_xlabel("current-ego local coordinate 0")
        path_axis.set_ylabel("current-ego local coordinate 1")
        path_axis.set_title(f"Path comparison, frame {frame_id:04d}")
        path_axis.legend(fontsize=8)

        if rgb_image is not None:
            rgb_axis = axes_row[2]
            rgb_axis.imshow(rgb_image)
            rgb_axis.set_title(f"Raw RGB {rgb_path.name}")
            rgb_axis.axis("off")

        route_label = route_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
        figure.suptitle(
            f"{route_label} | frame={frame_id:04d} | split={entry['split']}",
            fontsize=10,
        )
        figure_name = _figure_name(selection_index, entry)
        figure_path = output_dir / figure_name
        figure.savefig(figure_path, dpi=140, bbox_inches="tight")
        pyplot.close(figure)

        figures.append(
            {
                "sample_id": f"{route_id}::{frame_id}",
                "route_id": route_id,
                "split": str(entry["split"]),
                "frame_id": frame_id,
                "timestamp": float(entry["timestamp"]),
                "record_path": str(entry["path"]),
                "figure_path": figure_name,
                "history_frame_ids": [
                    int(history_entry["frame_id"])
                    for history_entry, _ in aligned_history
                ],
                "history_alignment": "source ego to current ego via geometry.py",
                "rgb_path": str(rgb_path) if rgb_path is not None else None,
                "rgb_displayed": rgb_image is not None,
                "rgb_warning": rgb_warning,
                "bev_energy": {
                    "definition": "sqrt(mean(channel^2))",
                    "shape": list(energy.shape),
                    "min": float(energy.min()),
                    "max": float(energy.max()),
                },
                "raw_metrics": raw_frame_metrics(
                    current["pred_trajectory"], current["gt_trajectory"]
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "temporal_tf_cache_visualization_manifest",
        "status": "pass",
        "cache_root": str(cache_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selection": {
            "policy": "evenly spaced over route/frame sorted index within selected splits",
            "selected_splits": selected_scope,
            "max_samples": max_samples,
            "selected_samples": len(figures),
            "selected_sample_split_counts": dict(
                sorted(Counter(figure["split"] for figure in figures).items())
            ),
            "history_length": history_length,
        },
        "raw_rgb": {
            "enabled": raw_root is not None,
            "raw_root": str(resolved_raw_root) if resolved_raw_root is not None else None,
            "accepted_extensions": sorted(IMAGE_SUFFIXES),
        },
        "warnings": warnings,
        "figures": figures,
    }
    write_json_artifact(manifest, output_dir / "manifest.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        help="limit sanity checks and rendered samples to these splits (default: all)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="optional raw dataset root; 4-digit PNG/JPEG files under route/rgb are displayed",
    )
    args = parser.parse_args()
    if args.output_dir.name in TEST_MARKER_FILENAMES:
        parser.error("--output-dir may not collide with a permanent test marker")
    if not 1 <= args.max_samples <= 20:
        parser.error("--max-samples must be within [1, 20]")
    if args.history_length < 0:
        parser.error("--history-length must be non-negative")
    try:
        manifest = render_cache_visualizations(
            args.cache,
            args.output_dir,
            max_samples=args.max_samples,
            history_length=args.history_length,
            raw_root=args.raw_root,
            splits=set(args.splits) if args.splits is not None else None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(args.output_dir / "manifest.json"),
                "figures": len(manifest["figures"]),
                "warnings": manifest["warnings"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
