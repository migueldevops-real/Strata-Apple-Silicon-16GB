#!/usr/bin/env bash
# setup-macos.sh - Apple Silicon setup for the MLX backend (Qwen2.5-7B-Instruct-1M, 256K context).
#
# Creates a venv, installs the server's requirements and mlx-lm, then starts the Strata server with the MLX
# backend.  The model downloads on first start (~4.4 GB).  No sudo.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This backend needs Apple Silicon (native arm64 macOS)." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[strata] creating .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "[strata] installing requirements ..."
pip install -r requirements.txt
echo "[strata] installing mlx-lm and the menu bar helper (rumps) ..."
pip install mlx-lm rumps

exec python -m serve.server --engine mlx --config "${CONFIG:-strata-qwen25-7b-1m.json}" \
  --port "${PORT:-8080}" "$@"
