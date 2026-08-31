#!/usr/bin/env python3
"""Run a locked multi-seed selection or its single final test-open event."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_tf.config import load_config
from temporal_tf.model import VARIANTS
from temporal_tf.study import (
    DEFAULT_STUDY_SEEDS,
    DEFAULT_STUDY_VARIANTS,
    STUDY_MANIFEST,
    STUDY_REPORT,
    STUDY_RESULTS,
    finalize_multiseed_study,
    run_multiseed_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-or-more seed study with one permanent final test-open event."
    )
    parser.add_argument("--evaluation-mode", choices=("selection", "final"), default="selection")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--study-selection",
        type=Path,
        help="optional final-time assertion; must equal the choice's selected study",
    )
    parser.add_argument(
        "--selection-choice",
        type=Path,
        help=(
            "final-only comparator selection_choice.json; its recomputed chosen study "
            "and manifest SHA lock the final"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--residual-weight", type=float)
    args = parser.parse_args()

    if args.evaluation_mode == "selection":
        if args.config is None:
            parser.error("selection requires --config")
        if args.study_selection is not None or args.selection_choice is not None:
            parser.error("--study-selection and --selection-choice are only valid for final")
        config = load_config(args.config)
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.residual_weight is not None:
            config["training"]["residual_weight"] = args.residual_weight
        run_multiseed_selection(
            cache_root=args.cache,
            study_dir=args.output,
            config=config,
            seeds=args.seeds or DEFAULT_STUDY_SEEDS,
            variants=args.variants or DEFAULT_STUDY_VARIANTS,
            device_name=args.device,
        )
        print(args.output / STUDY_MANIFEST)
    else:
        if args.selection_choice is None:
            parser.error("final requires comparator --selection-choice")
        if (
            args.config is not None
            or args.seeds is not None
            or args.variants is not None
            or args.epochs is not None
            or args.residual_weight is not None
        ):
            parser.error(
                "final config, seeds, variants, and hyperparameters are locked by --selection-choice"
            )
        finalize_multiseed_study(
            study_selection=args.study_selection,
            cache_root=args.cache,
            output_dir=args.output,
            device_name=args.device,
            selection_choice=args.selection_choice,
        )
        print(args.output / STUDY_RESULTS)
        print(args.output / STUDY_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
