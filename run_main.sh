#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
MAIN_FILE="$SRC_DIR/main.py"
VENV_DIR="$PROJECT_ROOT/.venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
REQUIREMENTS_HASH_FILE="$VENV_DIR/.requirements.sha256"
BOOTSTRAP_PY="${PYTHON_BOOTSTRAP:-python3.13}"
BOOTSTRAP_PY_PATH=""
VENV_PYTHON=""
PYTHON_BIN=""

if [ ! -f "$MAIN_FILE" ]; then
  echo "Unable to find main entrypoint at $MAIN_FILE" >&2
  exit 1
fi

export PYTHONPATH="$SRC_DIR:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

declare -a ARGS=()

ensure_bootstrap_python() {
  if command -v "$BOOTSTRAP_PY" >/dev/null 2>&1; then
    BOOTSTRAP_PY_PATH="$(command -v "$BOOTSTRAP_PY")"
  else
    echo "Could not locate required Python interpreter '$BOOTSTRAP_PY'. Install Python 3.13 or set PYTHON_BOOTSTRAP." >&2
    exit 1
  fi

  local version
  version="$("$BOOTSTRAP_PY_PATH" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
  if [ "$version" != "3.13" ]; then
    echo "Interpreter at $BOOTSTRAP_PY_PATH is Python $version; Python 3.13 is required." >&2
    exit 1
  fi
}

ensure_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating Python 3.13 virtual environment at $VENV_DIR"
    "$BOOTSTRAP_PY_PATH" -m venv "$VENV_DIR"
  fi

  VENV_PYTHON="$VENV_DIR/bin/python"

  local venv_version
  venv_version="$("$VENV_PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
  if [ "$venv_version" != "3.13" ]; then
    echo "Existing virtual environment uses Python $venv_version. Remove $VENV_DIR and rerun to recreate it with Python 3.13." >&2
    exit 1
  fi
}

install_requirements_if_needed() {
  if [ ! -f "$REQUIREMENTS_FILE" ]; then
    return
  fi

  local current_hash previous_hash
  current_hash="$("$VENV_PYTHON" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$REQUIREMENTS_FILE")"
  previous_hash=""
  if [ -f "$REQUIREMENTS_HASH_FILE" ]; then
    previous_hash="$(cat "$REQUIREMENTS_HASH_FILE")"
  fi

  if [ "$current_hash" != "$previous_hash" ]; then
    echo "Installing dependencies from ${REQUIREMENTS_FILE#$PROJECT_ROOT/}"
    "$VENV_PYTHON" -m pip install --upgrade pip >/dev/null
    "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"
    echo "$current_hash" > "$REQUIREMENTS_HASH_FILE"
  fi
}

bootstrap_environment() {
  ensure_bootstrap_python
  ensure_venv
  install_requirements_if_needed
  PYTHON_BIN="${PYTHON_BIN:-$VENV_PYTHON}"
}

parse_cli_args() {
  local label="$1"

  while true; do
    if ! read -r -p "CLI args for ${label} (blank for none): " raw_args; then
      echo
      return 1
    fi

    if [ -z "$raw_args" ]; then
      ARGS=()
      return 0
    fi

    if mapfile -t ARGS < <("$PYTHON_BIN" -c 'import shlex, sys
line = sys.argv[1]
try:
    parts = shlex.split(line)
except ValueError as exc:
    raise SystemExit(str(exc))
for part in parts:
    print(part)
' "$raw_args"); then
      return 0
    else
      echo "Could not parse arguments. Please try again."
    fi
  done
}

bootstrap_environment

if ! parse_cli_args "src/main.py"; then
  echo "Aborted."
  exit 1
fi

echo
echo ">>> Running: $PYTHON_BIN ${MAIN_FILE#$PROJECT_ROOT/} ${ARGS[*]:-}"
echo

"$PYTHON_BIN" "$MAIN_FILE" ${ARGS[@]+"${ARGS[@]}"}

