#!/usr/bin/env python3
"""Audit schema, split isolation, temporal identity, and forbidden inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_tf.audit import audit_cache


TEST_MARKER_FILENAMES = {"test_opened.marker.json", "study_test_opened.marker.json"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        help="limit entry checks and deep tensor loads to these splits (default: all)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.name in TEST_MARKER_FILENAMES:
        parser.error("--output may not overwrite a permanent test marker")
    report = audit_cache(
        args.cache,
        deep=args.deep,
        splits=set(args.splits) if args.splits is not None else None,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
