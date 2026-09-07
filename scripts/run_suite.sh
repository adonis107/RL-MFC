#!/usr/bin/env bash
# Run the full experiment suite, one environment at a time, in the order below.
#
#   scripts/run_suite.sh                    # resume: skip jobs that already have summary.json
#   scripts/run_suite.sh --no-resume        # re-run everything, overwriting existing results
#   WORKERS=4 scripts/run_suite.sh          # override the worker count
#
# Anything passed on the command line is forwarded to parallel_train.py.
set -euo pipefail

cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-8}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
ENVS=(portfolio lq advertising cybersecurity twostate distribution)

# Each job is a separate PyTorch process and torch would otherwise claim about half
# the box per process (torch.get_num_threads() is 8 on a 16-core machine). With
# WORKERS running at once that oversubscribes the CPU several times over, so give
# each worker its own share instead. subprocess.run inherits this environment.
CORES="$(nproc)"
THREADS=$(( CORES / WORKERS ))
(( THREADS < 1 )) && THREADS=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

echo "cores=${CORES} workers=${WORKERS} threads/worker=${THREADS} results=${RESULTS_ROOT}"
echo "order: ${ENVS[*]}"
echo

for env in "${ENVS[@]}"; do
    echo "================ ${env} ================"
    started=$(date +%s)
    uv run python scripts/parallel_train.py \
        --env "${env}" \
        --device cpu \
        --budget-mode fair \
        --workers "${WORKERS}" \
        --results-root "${RESULTS_ROOT}" \
        --logs-root "${RESULTS_ROOT}/logs" \
        "$@"
    echo "${env} finished in $(( ($(date +%s) - started) / 60 )) min"
    echo
done

echo "================ figures and tables ================"
uv run python scripts/plot.py --results-root "${RESULTS_ROOT}" --env all \
    --output-root "${RESULTS_ROOT}/figures"
