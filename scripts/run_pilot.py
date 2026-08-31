#!/usr/bin/env python3
"""Run the frozen baseline and temporal adapter ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_tf.config import load_config, validate_config
from temporal_tf.engine import finalize_selection, run_pilot
from temporal_tf.model import VARIANTS
from temporal_tf.report import write_markdown


REAL_PILOT_VARIANTS = [
    "current_only",
    "current_only_matched",
    "trajectory_only",
    "current_bev",
    "past_bev",
    "shuffled_past_bev",
    "combined",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="required for selection; optional final-time verification against the locked config",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--evaluation-mode",
        choices=["selection", "final"],
        default="selection",
        help="selection creates locked checkpoints; final requires --selection-run and never retrains",
    )
    parser.add_argument("--selection-run", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=sorted(VARIANTS),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--residual-weight", type=float)
    args = parser.parse_args()

    if args.evaluation_mode == "selection":
        if args.config is None:
            parser.error("--evaluation-mode selection requires --config")
        if args.selection_run is not None:
            parser.error("--selection-run is only valid with --evaluation-mode final")
        config = load_config(args.config)
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.residual_weight is not None:
            config["training"]["residual_weight"] = args.residual_weight
        validate_config(config)
        variants = args.variants or REAL_PILOT_VARIANTS
        results = run_pilot(
            cache_root=args.cache,
            output_dir=args.output,
            config=config,
            device_name=args.device,
            variants=variants,
        )
    else:
        if args.selection_run is None:
            parser.error("--evaluation-mode final requires --selection-run")
        if args.variants is not None:
            parser.error("final variants are locked by --selection-run; do not pass --variants")
        if args.epochs is not None or args.residual_weight is not None:
            parser.error("final hyperparameters are locked; do not pass override flags")
        config = load_config(args.config) if args.config is not None else None
        results = finalize_selection(
            selection_dir=args.selection_run,
            cache_root=args.cache,
            output_dir=args.output,
            config=config,
            device_name=args.device,
        )
    report = write_markdown(results, args.output / "RESULTS.md")
    print(args.output / "results.json")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
