#!/usr/bin/env bash
# run-macos.sh - start the Strata server with the MLX backend from an existing .venv (see setup-macos.sh).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "no .venv yet: run ./setup-macos.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
exec python -m serve.server --engine mlx --config "${CONFIG:-strata-qwen25-7b-1m.json}" \
  --port "${PORT:-8080}" "$@"
