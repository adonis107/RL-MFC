#!/usr/bin/env bash
set -euo pipefail

WORKERS="${WORKERS:-2}"
SEEDS="${SEEDS:-0,1,2,3,4}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
DEVICE="${DEVICE:-cuda}"
BUDGET_MODE="${BUDGET_MODE:-fair}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

uv sync --frozen

ARGS=(
  --env all \
  --seeds "${SEEDS}" \
  --results-root "${RESULTS_ROOT}" \
  --device "${DEVICE}" \
  --budget-mode "${BUDGET_MODE}" \
  --workers "${WORKERS}" \
)

if [[ -n "${JOBS_FILE:-}" ]]; then
  ARGS+=(--jobs-file "${JOBS_FILE}")
fi

if [[ -n "${LOGS_ROOT:-}" ]]; then
  ARGS+=(--logs-root "${LOGS_ROOT}")
fi

uv run python "${ROOT}/scripts/parallel_train.py" "${ARGS[@]}" "$@"
