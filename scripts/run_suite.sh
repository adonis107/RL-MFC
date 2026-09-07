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

WORKERS="${WORKERS:-7}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
ENVS=(portfolio lq advertising cybersecurity twostate distribution)

# Each job is a separate PyTorch process and torch would otherwise claim about half
# the box per process (torch.get_num_threads() is 8 on a 16-core machine). With
# WORKERS running at once that oversubscribes the CPU several times over, so give
# each worker its own share instead. subprocess.run inherits this environment.
#
# Do NOT use bare `nproc` here: it honours OMP_NUM_THREADS/OMP_THREAD_LIMIT, which
# this script sets itself, so it reports 1 whenever OMP_NUM_THREADS=1 is already in
# the environment. Read the affinity mask directly instead.
DETECTED="$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null \
            || nproc --all 2>/dev/null || echo 1)"
CORES="${CORES:-${DETECTED}}"

# On a cloud container /proc usually shows the whole host, not the slice you were
# allocated, so a large detected count is not something to trust blindly.
if [[ -z "${CORES_CONFIRMED:-}" && "${DETECTED}" -gt 32 ]]; then
    echo "WARNING: the container reports ${DETECTED} CPUs, which is almost certainly the"
    echo "  host's count rather than your allocation. Set CORES to the number of vCPU you"
    echo "  were actually given, e.g. CORES=16 $0"
    echo "  Set CORES_CONFIRMED=1 to silence this."
    echo
fi
if [[ "${CORES}" -lt "${WORKERS}" ]]; then
    echo "WARNING: CORES=${CORES} is below WORKERS=${WORKERS}; each worker gets 1 thread"
    echo "  and the box will be oversubscribed ${WORKERS}:${CORES}."
    echo
fi

THREADS=$(( CORES / WORKERS ))
(( THREADS < 1 )) && THREADS=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

mkdir -p "${RESULTS_ROOT}"
echo "cores=${CORES} (detected ${DETECTED}) workers=${WORKERS}"
echo "threads/worker=${THREADS} -> $(( WORKERS * THREADS )) of ${CORES} cores in use"
echo "results=${RESULTS_ROOT}"
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
