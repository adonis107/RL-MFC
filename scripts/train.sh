#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-0,1,2,3,4}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
BUDGET_MODE="${BUDGET_MODE:-fair}"
RUN_MODE="${RUN_MODE:-single}"

WORKERS="${WORKERS:-16}"
DEVICE="${DEVICE:-cpu}"

GPU_ENVS="${GPU_ENVS:-}"
GPU_DEVICE="${GPU_DEVICE:-cuda}"
GPU_WORKERS="${GPU_WORKERS:-1}"

CPU_ENVS="${CPU_ENVS:-twostate cybersecurity distribution advertising lq portfolio}"
CPU_DEVICE="${CPU_DEVICE:-cpu}"
CPU_WORKERS="${CPU_WORKERS:-16}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS

AUTO_WORKERS="${AUTO_WORKERS:-0}"
MIN_WORKERS="${MIN_WORKERS:-1}"
TARGET_CPU="${TARGET_CPU:-90}"
TARGET_GPU="${TARGET_GPU:-}"
MIN_FREE_MEMORY_GB="${MIN_FREE_MEMORY_GB:-6}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-60}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

uv sync --frozen
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

common_args=(
  --seeds "${SEEDS}"
  --results-root "${RESULTS_ROOT}"
  --budget-mode "${BUDGET_MODE}"
)

if [[ -n "${LOGS_ROOT:-}" ]]; then
  common_args+=(--logs-root "${LOGS_ROOT}")
fi

if [[ "${AUTO_WORKERS}" == "1" ]]; then
  common_args+=(
    --auto-workers
    --min-workers "${MIN_WORKERS}"
    --target-cpu "${TARGET_CPU}"
    --min-free-memory-gb "${MIN_FREE_MEMORY_GB}"
    --sample-interval "${SAMPLE_INTERVAL}"
  )

  if [[ -n "${TARGET_GPU}" ]]; then
    common_args+=(--target-gpu "${TARGET_GPU}")
  fi
fi

run_env() {
  local env="$1"
  local device="$2"
  local workers="$3"
  local group="$4"
  shift 4

  "${PYTHON}" "${ROOT}/scripts/parallel_train.py" \
    --env "${env}" \
    --device "${device}" \
    --workers "${workers}" \
    --jobs-file "${RESULTS_ROOT}/train_jobs_${group}_${env}.jsonl" \
    "${common_args[@]}" \
    "$@"
}

run_group() {
  local group="$1"
  local device="$2"
  local workers="$3"
  local envs="$4"
  shift 4
  local status=0

  if [[ -z "${envs// /}" ]]; then
    echo "No ${group} environments configured; skipping."
    return 0
  fi

  for env in ${envs}; do
    echo "Running ${env} on ${device} with ${workers} workers."
    run_env "${env}" "${device}" "${workers}" "${group}" "$@" || status=1
  done

  return "${status}"
}

if [[ "${RUN_MODE}" == "single" ]]; then
  args=(
    --env all
    --device "${DEVICE}"
    --workers "${WORKERS}"
    "${common_args[@]}"
  )

  if [[ -n "${JOBS_FILE:-}" ]]; then
    args+=(--jobs-file "${JOBS_FILE}")
  fi

  "${PYTHON}" "${ROOT}/scripts/parallel_train.py" "${args[@]}" "$@"
  exit $?
fi

if [[ "${RUN_MODE}" != "split" ]]; then
  echo "Unknown RUN_MODE=${RUN_MODE}. Use RUN_MODE=split or RUN_MODE=single." >&2
  exit 2
fi

echo "Running split suite: GPU envs [${GPU_ENVS}] on ${GPU_DEVICE}, CPU envs [${CPU_ENVS}] on ${CPU_DEVICE}."
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}; results_root=${RESULTS_ROOT}."

run_group "gpu" "${GPU_DEVICE}" "${GPU_WORKERS}" "${GPU_ENVS}" "$@" &
gpu_pid=$!

run_group "cpu" "${CPU_DEVICE}" "${CPU_WORKERS}" "${CPU_ENVS}" "$@" &
cpu_pid=$!

status=0
wait "${gpu_pid}" || status=1
wait "${cpu_pid}" || status=1
exit "${status}"
