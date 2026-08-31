#!/usr/bin/env python3
"""Evaluate fixed real-pilot continuation gates without touching selection state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from temporal_tf.decision import (
    DecisionInputError,
    load_and_evaluate,
    write_decision_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a finalized study_results.json and emit fixed Go/No-Go JSON and Markdown. "
            "The evaluator never reads selection directories or test-open markers."
        )
    )
    parser.add_argument("study_results", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="default: GO_NO_GO.json beside study_results.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="default: GO_NO_GO.md beside study_results.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing decision reports (never experiment markers)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = args.study_results.parent
    json_output = args.json_output or parent / "GO_NO_GO.json"
    markdown_output = args.markdown_output or parent / "GO_NO_GO.md"
    try:
        decision, _ = load_and_evaluate(args.study_results)
        write_decision_outputs(
            decision,
            json_output=json_output,
            markdown_output=markdown_output,
            overwrite=args.overwrite,
        )
    except (DecisionInputError, FileExistsError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"decision={decision['status']}")
    print(f"json={json_output}")
    print(f"markdown={markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
