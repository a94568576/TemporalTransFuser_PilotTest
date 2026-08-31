#!/usr/bin/env python3
"""Audit raw TF++/CARLA data-agent collections before feature caching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_tf.raw_collection import (
    assert_collection_key_matches_route,
    audit_raw_collection,
    load_collection_root_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively audit result.json metadata and saved sensor-frame alignment. "
            "Each input may be a collection container, a leaf output root, or result.json."
        )
    )
    parser.add_argument("roots", nargs="*", type=Path, help="raw collection root(s)")
    parser.add_argument(
        "--root-manifest",
        type=Path,
        help="LOGICAL_KEY ROOT manifest; mutually exclusive with positional roots",
    )
    parser.add_argument("--output", type=Path, help="optional path for the JSON report")
    args = parser.parse_args()

    if bool(args.roots) == bool(args.root_manifest):
        parser.error("provide either positional roots or --root-manifest, but not both")
    manifest_metadata = None
    if args.root_manifest is not None:
        try:
            entries = load_collection_root_manifest(args.root_manifest)
        except ValueError as exc:
            parser.error(str(exc))
        roots = [Path(entry["root"]) for entry in entries]
        manifest_metadata = {
            "path": str(args.root_manifest.resolve()),
            "entries": entries,
        }
    else:
        roots = args.roots

    report = audit_raw_collection(roots)
    if manifest_metadata is not None:
        results_by_root = {
            str(Path(result["result_root"]).resolve()): result for result in report["results"]
        }
        matched_keys: list[str] = []
        manifest_errors: list[str] = []
        for entry in manifest_metadata["entries"]:
            result = results_by_root.get(str(Path(entry["root"]).resolve()))
            if result is None:
                manifest_errors.append(
                    f"manifest key {entry['key']!r} has no uniquely audited result root"
                )
                continue
            route_directory = result["route_directory"]
            if route_directory is None:
                manifest_errors.append(
                    f"manifest key {entry['key']!r} has no unique route directory"
                )
                continue
            try:
                assert_collection_key_matches_route(entry["key"], route_directory)
            except (ValueError, RuntimeError) as exc:
                manifest_errors.append(str(exc))
                continue
            matched_keys.append(entry["key"])
        if manifest_errors:
            report["errors"].extend(manifest_errors)
            report["status"] = "fail"
        manifest_metadata["expected_route_count"] = len(manifest_metadata["entries"])
        manifest_metadata["matched_route_keys"] = matched_keys
        report["root_manifest"] = manifest_metadata
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "fail",
                        "error": f"cannot write output: {type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                )
            )
            return 1
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
