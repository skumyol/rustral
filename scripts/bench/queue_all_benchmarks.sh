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
#   RUN_NLP_PAPER    default 1; runs scripts/eval/run_nlp_real.py --paper --clean
#   RUN_NLP_PYTORCH  default 1 with NLP paper: adds --pytorch (needs torch); set 0 to skip
#   RUN_PYTORCH       default 1; set 0 to skip pytorch suite in run_all
#   RUN_PYTORCH_CUDA  default 1; adds pytorch-cuda (skipped if no CUDA device)
#   RUN_CUDA_BENCH    default 1; runs rustral-cuda micro suite (skipped if CUDA toolchain missing)
#   RUN_METAL_BENCH   default 1 on macOS, else 0; runs rustral-metal micro suite when available
#   RUN_STRICT        default 0; when 1, treat optional suite failures as failures
#   RUN_EXTRA_BASELINES default 1; attempt jax/tensorflow/onnxruntime suites (skipped if deps missing)
#   RUSTRAL_PYTHON    optional; default is repo .venv/bin/python or venv/bin/python, else python3

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=lib_rustral_python.sh
source "${ROOT}/scripts/bench/lib_rustral_python.sh"
_rustral_bench_resolve_python "$ROOT"
export RUSTRAL_PYTHON

if [[ "${1:-}" == "--worker" ]]; then
  set +e
  failures=0
  # Stdout/stderr of this process are already redirected to run.log by the launcher.
  log() { echo "[$(date -Is)] $*"; }

  cd "$ROOT"
  log "=== queue worker start STAMP=${STAMP} ROOT=${ROOT} PYTHON=${RUSTRAL_PYTHON} ==="

  log "=== [1] run_all.py: rustral + candle -> ${OUT_JSON} ==="
  if "${RUSTRAL_PYTHON}" scripts/bench/run_all.py \
    --repeats "${BENCH_REPEATS}" --warmup "${BENCH_WARMUP}" \
    --suite rustral --suite candle --out "${OUT_JSON}"; then
    log "[1] ok"
  else
    log "[1] FAILED"
    failures=$((failures + 1))
  fi

  if [[ "${RUN_PYTORCH:-1}" == "1" ]]; then
    log "=== [2] run_all.py: pytorch -> ${OUT_JSON_PYTORCH} ==="
    py_cmd=(
      "${RUSTRAL_PYTHON}" scripts/bench/run_all.py
      --repeats "${BENCH_REPEATS_PYTORCH}" --warmup "${BENCH_WARMUP}"
      --out "${OUT_JSON_PYTORCH}"
      --suite pytorch
    )
    if [[ "${RUN_PYTORCH_CUDA:-0}" == "1" ]]; then
      py_cmd+=(--suite pytorch-cuda)
      log "(including pytorch-cuda when a CUDA device is available)"
    fi
    if "${py_cmd[@]}"; then
      log "[2] ok"
    else
      log "[2] FAILED or skipped (install torch in this env to succeed)"
      failures=$((failures + 1))
    fi
  else
    log "=== [2] run_all pytorch SKIPPED (RUN_PYTORCH=0) ==="
  fi

  if [[ "${RUN_EXTRA_BASELINES:-1}" == "1" ]]; then
    log "=== [2b] extra baselines (jax/tensorflow/onnxruntime) -> ${OUT_JSON_EXTRA} ==="
    if "${RUSTRAL_PYTHON}" scripts/bench/run_all.py \
      --repeats "${BENCH_REPEATS_PYTORCH}" --warmup "${BENCH_WARMUP}" \
      --suite jax --suite jax-gpu --suite tensorflow --suite tensorflow-gpu --suite onnxruntime --suite onnxruntime-cuda \
      --out "${OUT_JSON_EXTRA}"; then
      log "[2b] ok"
    else
      log "[2b] FAILED (install optional deps: jax / tensorflow / onnxruntime / onnx) "
      if [[ "${RUN_STRICT:-0}" == "1" ]]; then
        failures=$((failures + 1))
      else
        log "[2b] treating as SKIPPED (RUN_STRICT=1 to make this fatal)"
      fi
    fi
  else
    log "=== [2b] extra baselines SKIPPED (RUN_EXTRA_BASELINES=0) ==="
  fi

  if [[ "${RUN_CUDA_BENCH:-0}" == "1" ]]; then
    log "=== [3a] run_all.py: rustral-cuda -> ${OUT_JSON_CUDA} ==="
    if "${RUSTRAL_PYTHON}" scripts/bench/run_all.py \
      --repeats "${BENCH_REPEATS_CUDA}" --warmup "${BENCH_WARMUP}" \
      --suite rustral-cuda --out "${OUT_JSON_CUDA}"; then
      log "[3a] ok"
    else
      log "[3a] FAILED (CUDA binary/toolchain missing?)"
      if [[ "${RUN_STRICT:-0}" == "1" ]]; then
        failures=$((failures + 1))
      else
        log "[3a] treating as SKIPPED (RUN_STRICT=1 to make this fatal)"
      fi
    fi
  else
    log "=== [3a] rustral-cuda SKIPPED (set RUN_CUDA_BENCH=1 to enable) ==="
  fi

  if [[ "${RUN_METAL_BENCH:-0}" == "1" ]]; then
    log "=== [3b] run_all.py: rustral-metal -> ${OUT_JSON_METAL} ==="
    if "${RUSTRAL_PYTHON}" scripts/bench/run_all.py \
      --repeats "${BENCH_REPEATS_METAL}" --warmup "${BENCH_WARMUP}" \
      --suite rustral-metal --out "${OUT_JSON_METAL}"; then
      log "[3b] ok"
    else
      log "[3b] FAILED (Metal binary/toolchain missing?)"
      if [[ "${RUN_STRICT:-0}" == "1" ]]; then
        failures=$((failures + 1))
      else
        log "[3b] treating as SKIPPED (RUN_STRICT=1 to make this fatal)"
      fi
    fi
  else
    log "=== [3b] rustral-metal SKIPPED (set RUN_METAL_BENCH=1 to enable) ==="
  fi

  if [[ -f "${OUT_JSON}" ]]; then
    log "=== [4] validate_schema.py (harness JSON outputs) ==="
    schema_fail=0
    for f in "${OUT_JSON}" "${OUT_JSON_PYTORCH}" "${OUT_JSON_EXTRA}" "${OUT_JSON_CUDA}" "${OUT_JSON_METAL}"; do
      [[ -f "${f}" ]] || continue
      log "validate_schema.py (${f})"
      if ! "${RUSTRAL_PYTHON}" scripts/bench/validate_schema.py "${f}"; then
        schema_fail=1
      fi
    done
    if [[ "${schema_fail}" -eq 0 ]]; then
      log "[4] ok"
    else
      log "[4] FAILED"
      failures=$((failures + 1))
    fi
  fi

  log "=== [5] validate_manifest.py ==="
  if "${RUSTRAL_PYTHON}" scripts/bench/validate_manifest.py; then
    log "[5] ok"
  else
    log "[5] FAILED"
    failures=$((failures + 1))
  fi

  log "=== [6] regen_index.py --check ==="
  if "${RUSTRAL_PYTHON}" scripts/bench/regen_index.py --check; then
    log "[6] ok"
  else
    log "[6] FAILED (run ${RUSTRAL_PYTHON} scripts/bench/regen_index.py and commit if intentional)"
    failures=$((failures + 1))
  fi

  if [[ "${RUN_NLP_PAPER:-0}" == "1" ]]; then
    log "=== [7] NLP paper (run_nlp_real.py --paper) — long-running ==="
    NLP_CMD=("${RUSTRAL_PYTHON}" scripts/eval/run_nlp_real.py --paper --clean)
    if [[ "${RUN_NLP_PYTORCH:-1}" == "1" ]]; then
      NLP_CMD+=(--pytorch)
      log "(including --pytorch parity baselines)"
    fi
    if "${NLP_CMD[@]}"; then
      log "[7] ok"
    else
      log "[7] FAILED"
      failures=$((failures + 1))
    fi
  else
    log "=== [7] NLP paper SKIPPED (RUN_NLP_PAPER=1 to queue) ==="
  fi

  log "=== [8] comparative_report.py (Markdown for paper draft) ==="
  mkdir -p "${ROOT}/benchmarks/reports"
  report_extra=()
  [[ -f "${OUT_JSON_PYTORCH}" ]] && report_extra+=(--harness-extra "${OUT_JSON_PYTORCH}")
  [[ -f "${OUT_JSON_EXTRA}" ]] && report_extra+=(--harness-extra "${OUT_JSON_EXTRA}")
  [[ -f "${OUT_JSON_CUDA}" ]] && report_extra+=(--harness-extra "${OUT_JSON_CUDA}")
  [[ -f "${OUT_JSON_METAL}" ]] && report_extra+=(--harness-extra "${OUT_JSON_METAL}")
  if "${RUSTRAL_PYTHON}" scripts/bench/comparative_report.py \
    --harness "${OUT_JSON}" \
    "${report_extra[@]}" \
    --out "${ROOT}/benchmarks/reports/comparative_queue_${STAMP}.md"; then
    log "[8] ok → benchmarks/reports/comparative_queue_${STAMP}.md"
  else
    log "[8] FAILED"
    failures=$((failures + 1))
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
export OUT_JSON_EXTRA="${ROOT}/benchmarks/results/queue-${STAMP}-extra.json"
export OUT_JSON_CUDA="${ROOT}/benchmarks/results/queue-${STAMP}-cuda.json"
export OUT_JSON_METAL="${ROOT}/benchmarks/results/queue-${STAMP}-metal.json"
export RUN_PYTORCH="${RUN_PYTORCH:-1}"
export RUN_PYTORCH_CUDA="${RUN_PYTORCH_CUDA:-1}"
export RUN_CUDA_BENCH="${RUN_CUDA_BENCH:-1}"
if [[ "${RUN_METAL_BENCH:-}" == "" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export RUN_METAL_BENCH=1
  else
    export RUN_METAL_BENCH=0
  fi
fi
export RUN_NLP_PAPER="${RUN_NLP_PAPER:-1}"
export RUN_NLP_PYTORCH="${RUN_NLP_PYTORCH:-1}"
export BENCH_REPEATS_METAL="${BENCH_REPEATS_METAL:-${BENCH_REPEATS_CUDA:-5}}"
export RUN_STRICT="${RUN_STRICT:-0}"
export RUN_EXTRA_BASELINES="${RUN_EXTRA_BASELINES:-1}"

mkdir -p "${ROOT}/benchmarks/results"
mkdir -p "${ROOT}/benchmarks/reports"

nohup env \
  ROOT="$ROOT" STAMP="$STAMP" LOGDIR="$LOGDIR" \
  RUSTRAL_PYTHON="$RUSTRAL_PYTHON" \
  BENCH_REPEATS="$BENCH_REPEATS" BENCH_WARMUP="$BENCH_WARMUP" \
  BENCH_REPEATS_PYTORCH="$BENCH_REPEATS_PYTORCH" BENCH_REPEATS_CUDA="$BENCH_REPEATS_CUDA" \
  BENCH_REPEATS_METAL="$BENCH_REPEATS_METAL" \
  OUT_JSON="$OUT_JSON" OUT_JSON_PYTORCH="$OUT_JSON_PYTORCH" \
  OUT_JSON_EXTRA="$OUT_JSON_EXTRA" \
  OUT_JSON_CUDA="$OUT_JSON_CUDA" OUT_JSON_METAL="$OUT_JSON_METAL" \
  RUN_PYTORCH="$RUN_PYTORCH" RUN_PYTORCH_CUDA="$RUN_PYTORCH_CUDA" \
  RUN_CUDA_BENCH="$RUN_CUDA_BENCH" RUN_METAL_BENCH="$RUN_METAL_BENCH" \
  RUN_NLP_PAPER="$RUN_NLP_PAPER" RUN_NLP_PYTORCH="$RUN_NLP_PYTORCH" RUN_STRICT="$RUN_STRICT" \
  RUN_EXTRA_BASELINES="$RUN_EXTRA_BASELINES" \
  "$0" --worker > "${LOGDIR}/run.log" 2>&1 &

echo $! > "${LOGDIR}/queue.pid"
echo "Queued background benchmark queue (PID $(cat "${LOGDIR}/queue.pid"))."
echo "  Python:      ${RUSTRAL_PYTHON}"
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
echo ""
echo "Full paper-oriented queue (micro + optional GPU + NLP paper + comparative MD):"
echo "  RUN_NLP_PAPER=1 RUN_PYTORCH_CUDA=1 RUN_CUDA_BENCH=1 ./scripts/bench/queue_all_benchmarks.sh"
echo "Apple Silicon GPU micro-suite:"
echo "  RUN_METAL_BENCH=1 ./scripts/bench/queue_all_benchmarks.sh"
