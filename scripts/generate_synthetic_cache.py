#!/usr/bin/env python3
"""Generate a labeled synthetic cache for pipeline smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_tf.synthetic import generate_synthetic_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes", type=int, default=15)
    parser.add_argument("--frames-per-route", type=int, default=32)
    parser.add_argument("--waypoints", type=int, default=10)
    parser.add_argument("--bev-channels", type=int, default=16)
    parser.add_argument("--bev-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    index = generate_synthetic_cache(
        args.output,
        num_routes=args.routes,
        frames_per_route=args.frames_per_route,
        num_waypoints=args.waypoints,
        bev_channels=args.bev_channels,
        bev_size=args.bev_size,
        seed=args.seed,
    )
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
