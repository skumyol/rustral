#!/usr/bin/env bash
# Queue all local benchmarks in one background process (sequential jobs).
#
# Usage (from repo root):
#   ./scripts/bench/queue_all_benchmarks.sh
#   RUN_NLP_PAPER=1 ./scripts/bench/queue_all_benchmarks.sh    # adds heavy NLP --paper after harness
#   RUN_CUDA_BENCH=1 ./scripts/bench/queue_all_benchmarks.sh   # adds rustral-cuda (needs CUDA build)
#
# Monitor:
#   tail -f benchmarks/queue_logs/latest/run.log
#   tail -f benchmarks/queue_logs/latest/run.log & watch 'test -f benchmarks/queue_logs/latest/done && echo DONE'
#
# Environment:
#   BENCH_REPEATS   default 5
#   BENCH_WARMUP    default 1
#   RUN_NLP_PAPER   set to 1 to append scripts/eval/run_nlp_real.py --paper --clean
#   RUN_PYTORCH     default 1; set 0 to skip pytorch suite in run_all
#   RUN_CUDA_BENCH  set 1 to append cuda suite to same JSON (requires GPU toolchain)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--worker" ]]; then
  set +e
  failures=0
  # Stdout/stderr of this process are already redirected to run.log by the launcher.
  log() { echo "[$(date -Is)] $*"; }

  cd "$ROOT"
  log "=== queue worker start STAMP=${STAMP} ROOT=${ROOT} ==="

  log "=== [1] run_all.py: rustral + candle -> ${OUT_JSON} ==="
  if python3 scripts/bench/run_all.py \
    --repeats "${BENCH_REPEATS}" --warmup "${BENCH_WARMUP}" \
    --suite rustral --suite candle --out "${OUT_JSON}"; then
    log "[1] ok"
  else
    log "[1] FAILED"
    failures=$((failures + 1))
  fi

  if [[ "${RUN_PYTORCH:-1}" == "1" ]]; then
    log "=== [2] run_all.py: pytorch -> ${OUT_JSON_PYTORCH} ==="
    if python3 scripts/bench/run_all.py \
      --repeats "${BENCH_REPEATS_PYTORCH}" --warmup "${BENCH_WARMUP}" \
      --suite pytorch --out "${OUT_JSON_PYTORCH}"; then
      log "[2] ok"
    else
      log "[2] FAILED or skipped (install torch in this env to succeed)"
      failures=$((failures + 1))
    fi
  else
    log "=== [2] run_all pytorch SKIPPED (RUN_PYTORCH=0) ==="
  fi

  if [[ "${RUN_CUDA_BENCH:-0}" == "1" ]]; then
    log "=== [3] run_all.py: rustral-cuda -> ${OUT_JSON_CUDA} ==="
    if python3 scripts/bench/run_all.py \
      --repeats "${BENCH_REPEATS_CUDA}" --warmup "${BENCH_WARMUP}" \
      --suite rustral-cuda --out "${OUT_JSON_CUDA}"; then
      log "[3] ok"
    else
      log "[3] FAILED (CUDA binary/toolchain missing?)"
      failures=$((failures + 1))
    fi
  else
    log "=== [3] rustral-cuda SKIPPED (set RUN_CUDA_BENCH=1 to enable) ==="
  fi

  if [[ -f "${OUT_JSON}" ]]; then
    log "=== [4] validate_schema.py (${OUT_JSON}) ==="
    if python3 scripts/bench/validate_schema.py "${OUT_JSON}"; then
      log "[4] ok"
    else
      log "[4] FAILED"
      failures=$((failures + 1))
    fi
  fi

  log "=== [5] validate_manifest.py ==="
  if python3 scripts/bench/validate_manifest.py; then
    log "[5] ok"
  else
    log "[5] FAILED"
    failures=$((failures + 1))
  fi

  log "=== [6] regen_index.py --check ==="
  if python3 scripts/bench/regen_index.py --check; then
    log "[6] ok"
  else
    log "[6] FAILED (run python3 scripts/bench/regen_index.py and commit if intentional)"
    failures=$((failures + 1))
  fi

  if [[ "${RUN_NLP_PAPER:-0}" == "1" ]]; then
    log "=== [7] NLP paper (run_nlp_real.py --paper) — long-running ==="
    if python3 scripts/eval/run_nlp_real.py --paper --clean; then
      log "[7] ok"
    else
      log "[7] FAILED"
      failures=$((failures + 1))
    fi
  else
    log "=== [7] NLP paper SKIPPED (RUN_NLP_PAPER=1 to queue) ==="
  fi

  if [[ "$failures" -eq 0 ]]; then
    log "=== queue finished OK ==="
    echo ok > "${LOGDIR}/status.txt"
  else
    log "=== queue finished with ${failures} failing step(s) ==="
    echo "failures=${failures}" > "${LOGDIR}/status.txt"
  fi
  date -Is > "${LOGDIR}/done"
  exit "$failures"
fi

# --- launcher ---
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${ROOT}/benchmarks/queue_logs/${STAMP}"
mkdir -p "${LOGDIR}"
ln -sfn "${STAMP}" "${ROOT}/benchmarks/queue_logs/latest"

export ROOT STAMP LOGDIR
export BENCH_REPEATS="${BENCH_REPEATS:-5}"
export BENCH_WARMUP="${BENCH_WARMUP:-1}"
export BENCH_REPEATS_PYTORCH="${BENCH_REPEATS_PYTORCH:-3}"
export BENCH_REPEATS_CUDA="${BENCH_REPEATS_CUDA:-5}"
export OUT_JSON="${ROOT}/benchmarks/results/queue-${STAMP}.json"
export OUT_JSON_PYTORCH="${ROOT}/benchmarks/results/queue-${STAMP}-pytorch.json"
export OUT_JSON_CUDA="${ROOT}/benchmarks/results/queue-${STAMP}-cuda.json"
export RUN_PYTORCH="${RUN_PYTORCH:-1}"
export RUN_CUDA_BENCH="${RUN_CUDA_BENCH:-0}"
export RUN_NLP_PAPER="${RUN_NLP_PAPER:-0}"

mkdir -p "${ROOT}/benchmarks/results"

nohup env \
  ROOT="$ROOT" STAMP="$STAMP" LOGDIR="$LOGDIR" \
  BENCH_REPEATS="$BENCH_REPEATS" BENCH_WARMUP="$BENCH_WARMUP" \
  BENCH_REPEATS_PYTORCH="$BENCH_REPEATS_PYTORCH" BENCH_REPEATS_CUDA="$BENCH_REPEATS_CUDA" \
  OUT_JSON="$OUT_JSON" OUT_JSON_PYTORCH="$OUT_JSON_PYTORCH" OUT_JSON_CUDA="$OUT_JSON_CUDA" \
  RUN_PYTORCH="$RUN_PYTORCH" RUN_CUDA_BENCH="$RUN_CUDA_BENCH" RUN_NLP_PAPER="$RUN_NLP_PAPER" \
  "$0" --worker > "${LOGDIR}/run.log" 2>&1 &

echo $! > "${LOGDIR}/queue.pid"
echo "Queued background benchmark queue (PID $(cat "${LOGDIR}/queue.pid"))."
echo "  Log file:    ${LOGDIR}/run.log"
echo "  Symlink:     ${ROOT}/benchmarks/queue_logs/latest/run.log"
echo ""
echo "Monitor (follow live output):"
echo "  tail -f ${ROOT}/benchmarks/queue_logs/latest/run.log"
echo ""
echo "Check if finished:"
echo "  test -f ${ROOT}/benchmarks/queue_logs/latest/done && cat ${ROOT}/benchmarks/queue_logs/latest/status.txt"
echo ""
echo "Process still running?"
echo "  ps -p \$(cat ${ROOT}/benchmarks/queue_logs/latest/queue.pid) -o pid,etime,cmd || echo 'not running'"
