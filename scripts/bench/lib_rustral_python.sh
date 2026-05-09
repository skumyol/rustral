#!/usr/bin/env bash
# Resolve Python for benchmark/eval scripts: use RUSTRAL_PYTHON if set and executable,
# else repo .venv/bin/python, venv/bin/python, else python3 on PATH.
_rustral_bench_resolve_python() {
  local root="$1"
  if [[ -n "${RUSTRAL_PYTHON:-}" ]]; then
    if [[ ! -x "${RUSTRAL_PYTHON}" ]]; then
      echo "error: RUSTRAL_PYTHON is not executable: ${RUSTRAL_PYTHON}" >&2
      return 1
    fi
    return 0
  fi
  if [[ -x "${root}/.venv/bin/python" ]]; then
    RUSTRAL_PYTHON="${root}/.venv/bin/python"
  elif [[ -x "${root}/venv/bin/python" ]]; then
    RUSTRAL_PYTHON="${root}/venv/bin/python"
  else
    RUSTRAL_PYTHON="python3"
  fi
}
