#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
TFPP_ROOT="${TFPP_ROOT:-${WORKSPACE_ROOT}/transfuser_test/carla_garage}"
MANIFEST="${1:-${PROJECT_ROOT}/configs/real_pilot_routes_town13_v1.txt}"
OUTPUT_ROOT="${2:-${PROJECT_ROOT}/data/raw_real_pilot_town13_v1}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Route manifest not found: ${MANIFEST}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

while read -r route_relative traffic_seed output_name; do
  if [[ -z "${route_relative}" || "${route_relative:0:1}" == "#" ]]; then
    continue
  fi
  if [[ -z "${traffic_seed}" || -z "${output_name}" ]]; then
    echo "Malformed manifest row: ${route_relative} ${traffic_seed} ${output_name}" >&2
    exit 1
  fi
  route_output="${OUTPUT_ROOT}/${output_name}"
  if [[ -f "${route_output}/result.json" ]]; then
    if "${TFPP_ROOT}/.venv/bin/python" -c \
      'import json, pathlib, sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); r=d["_checkpoint"]["global_record"]; raise SystemExit(0 if d.get("entry_status")=="Finished" and float(r["scores_mean"]["score_route"]) >= 99.999 and not r["meta"].get("exceptions") else 1)' \
      "${route_output}/result.json"; then
      echo "Skipping verified complete route: ${output_name}"
      continue
    fi
    echo "Existing route output is incomplete or failed: ${route_output}" >&2
    exit 1
  fi
  "${PROJECT_ROOT}/scripts/collect_tfpp_route.sh" \
    --route "${TFPP_ROOT}/data/50x36_Town13/${route_relative}" \
    --output "${route_output}" \
    --seed "${traffic_seed}" \
    --town Town13 \
    --repetition 0
done < "${MANIFEST}"
