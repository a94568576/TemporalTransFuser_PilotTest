#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
TFPP_ROOT="${TFPP_ROOT:-${WORKSPACE_ROOT}/transfuser_test/carla_garage}"
CARLA_ROOT="${CARLA_ROOT:-${WORKSPACE_ROOT}/old_carla}"
CARLA_PORT="${CARLA_PORT:-2000}"
TRAFFIC_MANAGER_PORT="${TRAFFIC_MANAGER_PORT:-8000}"
TOWN="${TOWN:-Town13}"
REPETITION="${REPETITION:-0}"
TRAFFIC_SEED="${TRAFFIC_SEED:-17}"
ROUTE_XML=""
OUTPUT_DIR=""
ENVIRONMENT_LOCK=""
ACQUISITION_SCHEDULE=""
SCHEDULE_ORDINAL=""
SPLIT=""
LIFECYCLE_INTENT=""
LIFECYCLE_RESULT=""
ACQUISITION_ORCHESTRATOR="${PROJECT_ROOT}/scripts/run_control_v2_acquisition.py"
GIT_BIN="${GIT_BIN:-${WORKSPACE_ROOT}/transfuser_test/bootstrap/git/usr/bin/git}"
readonly POST_EVALUATOR_COOLDOWN_SECONDS=5

usage() {
  echo "Usage: $0 --route ROUTE.xml --output DIR --environment-lock FILE --schedule FILE --schedule-ordinal N --split train|val|test --lifecycle-intent FILE --lifecycle-result FILE [options]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --route) ROUTE_XML="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --environment-lock) ENVIRONMENT_LOCK="$2"; shift 2 ;;
    --schedule) ACQUISITION_SCHEDULE="$2"; shift 2 ;;
    --schedule-ordinal) SCHEDULE_ORDINAL="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --lifecycle-intent) LIFECYCLE_INTENT="$2"; shift 2 ;;
    --lifecycle-result) LIFECYCLE_RESULT="$2"; shift 2 ;;
    --acquisition-orchestrator) ACQUISITION_ORCHESTRATOR="$2"; shift 2 ;;
    --seed) TRAFFIC_SEED="$2"; shift 2 ;;
    --town) TOWN="$2"; shift 2 ;;
    --repetition) REPETITION="$2"; shift 2 ;;
    --port) CARLA_PORT="$2"; shift 2 ;;
    --traffic-manager-port) TRAFFIC_MANAGER_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${ROUTE_XML}" || -z "${OUTPUT_DIR}" || -z "${ENVIRONMENT_LOCK}" || -z "${ACQUISITION_SCHEDULE}" || -z "${SCHEDULE_ORDINAL}" || -z "${SPLIT}" || -z "${LIFECYCLE_INTENT}" || -z "${LIFECYCLE_RESULT}" ]]; then
  usage
  exit 2
fi
if [[ ! "${SCHEDULE_ORDINAL}" =~ ^[1-9][0-9]*$ || ( "${SPLIT}" != "train" && "${SPLIT}" != "val" && "${SPLIT}" != "test" ) ]]; then
  echo "Invalid schedule ordinal/split" >&2
  exit 2
fi

ROUTE_XML="$(realpath "${ROUTE_XML}")"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
ENVIRONMENT_LOCK="$(realpath -m "${ENVIRONMENT_LOCK}")"
ACQUISITION_SCHEDULE="$(realpath "${ACQUISITION_SCHEDULE}")"
LIFECYCLE_INTENT="$(realpath "${LIFECYCLE_INTENT}")"
LIFECYCLE_RESULT="$(realpath -m "${LIFECYCLE_RESULT}")"
ACQUISITION_ORCHESTRATOR="$(realpath "${ACQUISITION_ORCHESTRATOR}")"
PYTHON="${TFPP_ROOT}/.venv/bin/python"
LEADERBOARD_ROOT="${TFPP_ROOT}/leaderboard_autopilot"
SCENARIO_RUNNER_ROOT="${TFPP_ROOT}/scenario_runner_autopilot"

if [[ ! -f "${ROUTE_XML}" ]]; then
  echo "Route XML not found: ${ROUTE_XML}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "TF++ Python environment not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${GIT_BIN}" ]]; then
  echo "Pinned Git executable not found: ${GIT_BIN}" >&2
  exit 1
fi
if [[ ! -d "${LEADERBOARD_ROOT}" || ! -d "${SCENARIO_RUNNER_ROOT}" ]]; then
  echo "Official autopilot leaderboard/scenario-runner directories are missing under ${TFPP_ROOT}" >&2
  exit 1
fi
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty collection directory: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
COLLECTOR_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
EVALUATOR_WRAPPER="${PROJECT_ROOT}/scripts/run_fixed_tfpp_evaluator.py"
export CARLA_ROOT
export WORK_DIR="${TFPP_ROOT}"
export LEADERBOARD_ROOT
export SCENARIO_RUNNER_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src:${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${TFPP_ROOT}/team_code"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TFPP_TIMM_PRETRAINED_INIT=0
export CHALLENGE_TRACK_CODENAME=MAP
export IS_BENCH2DRIVE=False
export DATAGEN=1
export DEBUG_CHALLENGE=0
export TMP_VISU=0
export HISTOGRAM=0
export TP_STATS=0
export DEACTIVATE_TRAFFIC=0
export SAVE_PATH="${OUTPUT_DIR}"
export TOWN
export REPETITION
export PYGAME_HIDE_SUPPORT_PROMPT=1
export CUDA_VISIBLE_DEVICES=0
export VULKAN_ICD_FILENAMES="${VULKAN_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}"

EVALUATOR_ARGS=(
  --host=localhost
  --port="${CARLA_PORT}"
  --traffic-manager-port="${TRAFFIC_MANAGER_PORT}"
  --traffic-manager-seed="${TRAFFIC_SEED}"
  --routes="${ROUTE_XML}"
  --repetitions=1
  --track=MAP
  --checkpoint="${OUTPUT_DIR}/result.json"
  --debug-checkpoint="${OUTPUT_DIR}/live_results.txt"
  --agent="${TFPP_ROOT}/team_code/data_agent.py"
  --agent-config="${ROUTE_XML}"
  --resume=0
  --debug=0
  --timeout=300
)

"${PYTHON}" -c \
  'import datetime, json, os, pathlib, sys, traceback
from temporal_tf.collection_provenance import hard_process_exit

def main():
    from temporal_tf.collection_provenance import (
        COLLECTION_EFFECTIVE_ENVIRONMENT_KEYS,
        _live_carla_process_identity,
        build_collection_environment,
        ensure_collection_environment_lock,
        write_collection_request,
    )
    output = pathlib.Path(sys.argv[1])
    lock_path = pathlib.Path(sys.argv[12])
    locked_opendrive = None
    if lock_path.is_file():
        locked_environment = json.loads(lock_path.read_text())["environment"]
        locked_live = locked_environment["carla"]["live_server"]
        locked_opendrive = {
            "opendrive_bytes": locked_live["opendrive_bytes"],
            "opendrive_sha256": locked_live["opendrive_sha256"],
        }
    environment = build_collection_environment(
        tfpp_root=sys.argv[9],
        carla_root=sys.argv[10],
        town=sys.argv[4],
        carla_port=int(sys.argv[6]),
        git_executable=sys.argv[11],
        acquisition_schedule_path=sys.argv[13],
        acquisition_orchestrator=sys.argv[18],
        locked_opendrive_identity=locked_opendrive,
    )
    route_process = _live_carla_process_identity(
        server_binary=pathlib.Path(environment["carla"]["server_binary"]["path"]),
        port=int(sys.argv[6]),
        expected_town=sys.argv[4],
    )
    if {
        "executable": route_process["executable"],
        "argv": route_process["argv"],
    } != environment["carla"]["live_process"]:
        raise RuntimeError("route-local CARLA process differs from environment lock")
    lock = ensure_collection_environment_lock(sys.argv[12], environment)
    write_collection_request(
        output / "collection_request.json",
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        route_xml=sys.argv[2],
        traffic_manager_seed=int(sys.argv[3]),
        town=sys.argv[4],
        repetition=int(sys.argv[5]),
        carla_port=int(sys.argv[6]),
        traffic_manager_port=int(sys.argv[7]),
        live_process_instance=route_process["server_instance"],
        acquisition_schedule_path=sys.argv[13],
        acquisition_schedule_ordinal=int(sys.argv[14]),
        split=sys.argv[15],
        lifecycle_intent_path=sys.argv[16],
        lifecycle_result_path=sys.argv[17],
        collector_script=sys.argv[8],
        evaluator_argv=[
            str(pathlib.Path(sys.executable).resolve()),
            str(pathlib.Path(sys.argv[19]).resolve()),
            *sys.argv[20:],
        ],
        effective_environment={
            key: os.environ.get(key, "")
            for key in COLLECTION_EFFECTIVE_ENVIRONMENT_KEYS
        },
        environment=environment,
        environment_lock=lock,
    )
    return 0

status = 1
try:
    status = main()
except BaseException:
    traceback.print_exc()
finally:
    hard_process_exit(status)' \
  "${OUTPUT_DIR}" "${ROUTE_XML}" "${TRAFFIC_SEED}" "${TOWN}" \
  "${REPETITION}" "${CARLA_PORT}" "${TRAFFIC_MANAGER_PORT}" "${COLLECTOR_SCRIPT}" \
  "${TFPP_ROOT}" "${CARLA_ROOT}" "${GIT_BIN}" "${ENVIRONMENT_LOCK}" \
  "${ACQUISITION_SCHEDULE}" "${SCHEDULE_ORDINAL}" "${SPLIT}" \
  "${LIFECYCLE_INTENT}" "${LIFECYCLE_RESULT}" "${ACQUISITION_ORCHESTRATOR}" \
  "${EVALUATOR_WRAPPER}" "${EVALUATOR_ARGS[@]}"

cd "${TFPP_ROOT}"
set +e
"${PYTHON}" "${EVALUATOR_WRAPPER}" "${EVALUATOR_ARGS[@]}"
EVALUATOR_STATUS=$?
set -e
sleep "${POST_EVALUATOR_COOLDOWN_SECONDS}"
if [[ ${EVALUATOR_STATUS} -ne 0 ]]; then
  echo "TF++ evaluator failed with status ${EVALUATOR_STATUS}; cooldown completed" >&2
  exit "${EVALUATOR_STATUS}"
fi

"${PYTHON}" -c \
  'import pathlib, sys, traceback
from temporal_tf.collection_provenance import hard_process_exit

def main():
    from temporal_tf.collection_provenance import (
        validate_collection_request,
        validate_runtime_attestation,
    )
    root = pathlib.Path(sys.argv[1]).resolve()
    validated = validate_collection_request(
        root / "collection_request.json",
        route_xml=sys.argv[2],
        traffic_manager_seed=int(sys.argv[3]),
        split=sys.argv[9],
        town=sys.argv[4],
        repetition=int(sys.argv[5]),
        environment_lock_path=sys.argv[6],
        carla_port=int(sys.argv[7]),
        traffic_manager_port=int(sys.argv[8]),
        acquisition_schedule_path=sys.argv[10],
        acquisition_schedule_ordinal=int(sys.argv[11]),
        verify_current_environment=True,
        verify_current_live_server=False,
        verify_effective_environment=True,
        verify_live_process=True,
    )
    validate_runtime_attestation(
        root / "runtime_attestation.json",
        validated_request=validated,
    )
    return 0

status = 1
try:
    status = main()
except BaseException:
    traceback.print_exc()
finally:
    hard_process_exit(status)' \
  "${OUTPUT_DIR}" "${ROUTE_XML}" "${TRAFFIC_SEED}" "${TOWN}" \
  "${REPETITION}" "${ENVIRONMENT_LOCK}" "${CARLA_PORT}" \
  "${TRAFFIC_MANAGER_PORT}" "${SPLIT}" "${ACQUISITION_SCHEDULE}" \
  "${SCHEDULE_ORDINAL}"

"${PYTHON}" -c \
  'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); r=d["_checkpoint"]["global_record"]; ok=d.get("entry_status")=="Finished" and float(r["scores_mean"]["score_route"]) >= 99.999 and not r["meta"].get("exceptions"); print("collection_status={} route_completion={}".format(r["status"], r["scores_mean"]["score_route"])); raise SystemExit(0 if ok else 1)' \
  "${OUTPUT_DIR}/result.json"
