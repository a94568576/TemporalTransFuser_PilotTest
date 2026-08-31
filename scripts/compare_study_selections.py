#!/usr/bin/env python3
"""Choose a residual weight using only locked multi-seed validation results."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_tf.selection_compare import (
    CHOICE_JSON,
    CHOICE_REPORT,
    DEFAULT_PRIMARY_VARIANT,
    write_selection_choice,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more test-closed study selections using mean validation "
            "route-macro ADE. This command never opens test records."
        )
    )
    parser.add_argument(
        "studies",
        nargs="+",
        type=Path,
        help="study directories or study_manifest.json files (at least two)",
    )
    parser.add_argument("--output", type=Path, required=True, help="fresh artifact directory")
    parser.add_argument("--primary-variant", default=DEFAULT_PRIMARY_VARIANT)
    args = parser.parse_args()
    if len(args.studies) < 2:
        parser.error("at least two study selections are required")

    choice = write_selection_choice(
        args.studies,
        output_dir=args.output,
        primary_variant=args.primary_variant,
    )
    print(args.output / CHOICE_JSON)
    print(args.output / CHOICE_REPORT)
    print(choice["chosen"]["study_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

