#!/usr/bin/env python3
"""Produce a read-only JSON sanity artifact for a frozen-planner cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_tf.sanity import analyze_cache, write_json_artifact


TEST_MARKER_FILENAMES = {"test_opened.marker.json", "study_test_opened.marker.json"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        help="limit tensor checks and metrics to these splits (default: all)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional prior sanity JSON whose aggregate mean/p90/p95 values must match",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="absolute tolerance for every compared reference metric",
    )
    parser.add_argument("--expected-frame-step", type=int, default=1)
    parser.add_argument(
        "--cadence-tolerance",
        type=float,
        default=1e-6,
        help="absolute tolerance in seconds per frame around each route's median cadence",
    )
    args = parser.parse_args()
    if args.output.name in TEST_MARKER_FILENAMES:
        parser.error("--output may not overwrite a permanent test marker")

    report = analyze_cache(
        args.cache,
        reference=args.reference,
        reference_tolerance=args.tolerance,
        expected_frame_step=args.expected_frame_step,
        cadence_tolerance=args.cadence_tolerance,
        splits=set(args.splits) if args.splits is not None else None,
    )
    artifact = write_json_artifact(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "artifact": str(artifact),
                "warnings": report["warnings"],
                "errors": report["errors"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
