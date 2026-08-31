#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
TFPP_ROOT="${TFPP_ROOT:-${WORKSPACE_ROOT}/transfuser_test/carla_garage}"
PYTHON="${TFPP_ROOT}/.venv/bin/python"
ROOT_MANIFEST="${ROOT_MANIFEST:-${PROJECT_ROOT}/configs/real_pilot_cache_roots_town13_v1.txt}"
OUTPUT_CACHE="${1:-${PROJECT_ROOT}/data/cache_real_pilot_town13_v1}"
EXPECTED_ROUTES="${EXPECTED_ROUTES:-15}"

if [[ ! -f "${ROOT_MANIFEST}" ]]; then
  echo "Explicit cache-root manifest is missing: ${ROOT_MANIFEST}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
exec "${PYTHON}" scripts/cache_tfpp_dataset.py \
  --tfpp-root "${TFPP_ROOT}" \
  --data-root-manifest "${ROOT_MANIFEST}" \
  --expected-route-count "${EXPECTED_ROUTES}" \
  --dataset-id carla_town13_real_pilot_v1 \
  --output "${OUTPUT_CACHE}" \
  --device cuda \
  --feature-source backbone_bev \
  --cache-spatial-size 8 \
  --split-seed 17 \
  --split-ratios 0.6 0.2 0.2 \
  --require-successful-collection-results
